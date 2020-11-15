#! /usr/bin/env python3

import cpufeature
import copy
from enum import Enum
import traceback
import itertools
import argparse
import signal
import subprocess
import os


def sig_exit(signum, empty):
    print("Exiting on Signal({})".format(str(signum)))
    sys.exit(-1)


signal.signal(signal.SIGINT, sig_exit)

parser = argparse.ArgumentParser(
    description="Export SIMD sorting network implementation as C code")

parser.add_argument("--size",
                    action="store",
                    default="",
                    help="Set size of sorted element")
parser.add_argument("-s",
                    "--signed",
                    action="store_true",
                    default=False,
                    help="Set sign of sorted type")
parser.add_argument("-u",
                    "--unsigned",
                    action="store_true",
                    default=False,
                    help="Set sign of sorted type")
parser.add_argument("-T",
                    "--type",
                    action="store",
                    default="",
                    help="Set type of sorted type")
parser.add_argument("-N",
                    action="store",
                    default="",
                    help="Set number of elements for sort")
parser.add_argument("-a",
                    "--algorithm",
                    action="store",
                    default="",
                    help="Set sorting network generation algorithm")
parser.add_argument("-O",
                    "--optimization",
                    action="store",
                    default="space",
                    help="Set optimization, either \"space\" or \"uop\"")
parser.add_argument("-e",
                    "--extra-memory",
                    action="store_true",
                    default=False,
                    help="Set to enable extra memory load & store")
parser.add_argument(
    "-i",
    "--int-aligned",
    action="store_true",
    default=False,
    help=
    "Set to indicate that memory accesses can be rounded up to nearest sizeof int."
)
parser.add_argument("--aligned",
                    action="store_true",
                    default=False,
                    help="Set to enable aligned load & store")
parser.add_argument("-c",
                    "--constraint",
                    action="store",
                    default="",
                    help="Set to enable aligned load & store")
parser.add_argument(
    "--no-format",
    action="store_false",
    default=True,
    help=
    "Set to not format output. Formatting will first try and pipe through clang-format. If clang-format executable does not exist or throws an error will then use some lazy fallback formatting."
)
parser.add_argument(
    "--clang-format",
    action="store",
    default="clang-format",
    help=
    "Set executable path for clang-format, default will use whats in your PATH"
)
parser.add_argument(
    "--name",
    action="store",
    default="",
    help=
    "set function name for output, the memory version will be \"<name>\", the vec version will be \"<name>_vec\". Defaults to <algorithm_name>_<sort_size>_<type_size><type_sign>"
)

parser.add_argument("--outfile",
                    action="store",
                    default="",
                    help="set output file. default is stdout")
parser.add_argument(
    "--fmode",
    action="store",
    default="",
    help="set to either \"a\" (append) or \"w\" (write) for file mode")
parser.add_argument(
    "--template",
    action="store_true",
    default=False,
    help=
    "set to store output function in specialized template class. If specified template name will be \"func\" and memory sort will be \"sort\" and vec will be \"sort_vec\". The default template name will be \"vsort\""
)


def err_assert(check, msg):
    if check is False:
        print("Error: " + msg)
        traceback.print_stack()
        exit(-1)


class Optimization(Enum):
    SPACE = 0
    UOP = 1


# forward declaraction
SORT_FUNC_NAME = ""
TEMPLATED = False
OUTFILE = ""
FMODE = ""
INSTRUCTION_OPT = Optimization.SPACE
SIMD_RESTRICTIONS = ""
ALIGNED_ACCESS = False
INT_ALIGNED = False
EXTRA_MEMORY = False
DO_FORMAT = False
CLANG_FORMAT_EXE = ""
USER_TYPE = None

MIN_MAX_COUNT = 0


def choose_if(Opt, weight1, weight2):
    if Opt == INSTRUCTION_OPT:
        return weight1
    else:
        return weight2


def arr_to_str(arr):
    arr_str = ""
    for i in range(0, len(arr)):
        arr_str += str(arr[i])
        if i != len(arr) - 1:
            arr_str += "\n"
    return arr_str


def arr_to_csv(arr, HEX=None):
    arr_str = ""
    for i in range(0, len(arr)):
        if i != 0:
            arr_str += ", "
        if HEX is not None and HEX is True:
            arr_str += str(hex(arr[i]))
        else:
            arr_str += str(arr[i])
    return arr_str


def arr_to_padd_str(arr, padding):
    plen = len(padding) - padding.count("\t")

    out = ""
    cur_len = plen
    for a in arr:
        new_line = False
        if len(a) + 2 + cur_len > 64:
            out += "\n"
            new_line = True
            for i in range(0, padding.count("\t")):
                out += "\t"
            for i in range(0, plen):
                out += " "
            cur_len = plen
        if len(out) != 0 and new_line is False:
            out += ", "
        out += a
        cur_len += len(a)
    return out


class Headers():
    def __init__(self):
        self.aliasing_m64 = False
        self.aliasing_int16 = False
        self.xmmintrin = False
        self.immintrin = False
        self.stdint = True

    def reset(self):
        self.aliasing_m64 = False
        self.aliasing_int16 = False
        self.xmmintrin = False
        self.immintrin = False
        self.stdint = True

    def get_headers(self, sort_type):
        ret = ""
        if self.xmmintrin is True:
            ret += "#include <xmmintrin.h>"
            ret += "\n"
        if self.immintrin is True:
            ret += "#include <immintrin.h>"
            ret += "\n"
        if self.stdint is True:
            ret += "#include <stdint.h>"
            ret += "\n"
        ret += "\n"

        if self.aliasing_m64 is True:
            alignment = sort_type.sizeof()
            if ALIGNED_ACCESS is True:
                alignment = 8
            ret += "typedef __m64 _aliasing_m64_ __attribute__((aligned({}), may_alias));".format(
                alignment)
            ret += "\n"
        if self.aliasing_int16 is True:
            alignment = sort_type.sizeof()
            if ALIGNED_ACCESS is True:
                alignment = max(alignment, 2)
            ret += "typedef uint16_t _aliasing_int16_t_ __attribute__((aligned({}), may_alias));".format(
                alignment)
            ret += "\n"
        return ret


header = Headers()

######################################################################
# Instruction Generation
######################################################################


class SIMD_Constraints():
    def __init__(self, constraints):
        self.constraints = copy.deepcopy(constraints)

    def has_support(self):
        for field in self.constraints:
            err_assert(field in cpufeature.CPUFeature,
                       "Testing invalid instruction set: {}".format(field))
            if cpufeature.CPUFeature[field] is False:
                return False
            if SIMD_RESTRICTIONS != "" and SIMD_RESTRICTIONS in field:
                return False
        return True


class SIMD_m64():
    def __init__(self):
        self.SIMD_constraints = SIMD_Constraints(["MMX"])

    def to_string(self):
        return "__m64"

    def prefix(self):
        return "_mm"

    def postfix(self):
        return "si64"

    def sizeof(self):
        return 8

    def has_support(self):
        return self.SIMD_constraints.has_support()


class SIMD_m128():
    def __init__(self):
        self.SIMD_constraints = SIMD_Constraints(["SSE2"])

    def to_string(self):
        return "__m128i"

    def prefix(self):
        return "_mm"

    def postfix(self):
        return "si128"

    def sizeof(self):
        return 16

    def has_support(self):
        return self.SIMD_constraints.has_support()


class SIMD_m256():
    def __init__(self):
        self.SIMD_constraints = SIMD_Constraints(["AVX2"])

    def to_string(self):
        return "__m256i"

    def prefix(self):
        return "_mm256"

    def postfix(self):
        return "si256"

    def sizeof(self):
        return 32

    def has_support(self):
        return self.SIMD_constraints.has_support()


class SIMD_m512():
    def __init__(self):
        self.SIMD_constraints = SIMD_Constraints(["AVX512f"])

    def to_string(self):
        return "__m512i"

    def prefix(self):
        return "_mm512"

    def postfix(self):
        return "si512"

    def sizeof(self):
        return 64

    def has_support(self):
        return self.SIMD_constraints.has_support()


def get_simd_type(sort_bytes):
    if sort_bytes <= SIMD_m64().sizeof():
        return SIMD_m64()
    elif sort_bytes <= SIMD_m128().sizeof():
        return SIMD_m128()
    elif sort_bytes <= SIMD_m256().sizeof():
        return SIMD_m256()
    elif sort_bytes <= SIMD_m512().sizeof():
        return SIMD_m512()
    else:
        err_assert(False, "No matching SIMD type")


class Sign(Enum):
    SIGNED = 0
    UNSIGNED = 1
    NOT_SIGNED = 2


class Sort_Type():
    def __init__(self, size, sign):
        self.size = int(size)
        err_assert(
            self.size == 1 or self.size == 2 or self.size == 4
            or self.size == 8, "Invalid sort_type size")

        self.sign = sign

        self.unsigned_casts = [
            "None", "uint8_t", "uint16_t", "None", "uint32_t", "None", "None",
            "None", "uint64_t"
        ]
        self.signed_casts = [
            "None", "int8_t", "int16_t", "None", "int32_t", "None", "None",
            "None", "int64_t"
        ]

    def sign(self):
        return self.sign

    def sizeof(self):
        return self.size

    def sizeof_bits(self):
        return 8 * self.sizeof()

    def to_string(self):
        if self.sign == Sign.SIGNED:
            return self.signed_casts[self.size]
        elif self.sign == Sign.UNSIGNED:
            return self.unsigned_casts[self.size]
        else:
            err_assert(False, "unable to find type")

    def min_value(self):
        val = 0
        if self.sign == Sign.SIGNED:
            val = (int(1) << (self.sizeof_bits() - 1))
        elif self.sign == Sign.UNSIGNED:
            val = 0

        return "{}({})".format(self.to_string(), str(hex(val)))

    def max_value_int(self):
        val = 0
        if self.sign == Sign.SIGNED:
            val = (int(1) << (self.sizeof_bits() - 1)) - 1
        elif self.sign == Sign.UNSIGNED:
            val = (int(1) << (self.sizeof_bits())) - 1
        return val

    def max_value(self):
        return "{}({})".format(self.to_string(),
                               str(hex(self.max_value_int())))

    def is_valid(self, T):
        if T == "None":
            return False
        if T in self.unsigned_casts:
            return True
        if T in self.signed_casts:
            return True

    def string_to_T(self, T):
        err_assert(self.is_valid(T),
                   "Trying to build sort type from unknown string")
        sign = Sign.SIGNED
        if T in self.unsigned_casts:
            sign = Sign.UNSIGNED

        for i in range(0, len(self.unsigned_casts)):
            if T == self.unsigned_casts[i] or T == self.signed_casts[i]:
                return Sort_Type(i, sign)
        err_assert(False, "was unable to find match")


class Match_Info():
    def __init__(self, sort_type, simd_type, aligned=None, full=None):
        self.simd_type = simd_type
        self.sort_type = sort_type
        self.aligned = aligned
        self.full = full


class SIMD_Instruction():
    def __init__(self, iname, sign, T_size, simd_type, constraints, weight):
        self.sign = sign
        self.T_size = T_size
        self.simd_type = simd_type
        self.iname = iname
        self.SIMD_constraints = SIMD_Constraints(constraints)
        self.constraints = constraints
        self.weight = weight

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type)

    def match_sort_type(self, sort_type):
        if self.sign != Sign.NOT_SIGNED and self.sign != sort_type.sign:
            return False
        if self.T_size != (-1) and self.T_size != sort_type.sizeof():
            return False
        return True

    def match_simd_type(self, simd_type):
        return self.simd_type.sizeof() == simd_type.sizeof()

    def has_support(self):
        return self.SIMD_constraints.has_support()

    def generate_instruction(self):
        return self.iname


######################################################################
# Minimum


# These are all simple instructions. A few fallbacks in the case of
# __m64 instructions and epi64 instructions if AVX512 is not
# present. But instruction selection entirely depends on what
# instructions the CPU has, not based on the input.
######################################################################
class SIMD_Min_Fallback_m64_s8(SIMD_Instruction):
    def __init__(self, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.SIGNED, 1, SIMD_m64(), ["MMX"], weight)

    def generate_instruction(self):
        global MIN_MAX_COUNT

        cmp_tmp = "_cmp_tmp{}".format(int(MIN_MAX_COUNT / 2))
        instruction = ""
        if MIN_MAX_COUNT % 2 == 0:
            instruction = "__m64 [CMP_TMP] = _mm_cmpgt_pi8([V1], [V2]);".replace(
                "[CMP_TMP]", cmp_tmp)
            instruction += "\n"
        instruction += "_mm_or_si64(_mm_and_si64([CMP_TMP], [V2]), _mm_andnot_si64([CMP_TMP], [V1]))".replace(
            "[CMP_TMP]", cmp_tmp)
        MIN_MAX_COUNT += 1
        return instruction


class SIMD_Min_Fallback_m64_u16(SIMD_Instruction):
    def __init__(self, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.UNSIGNED, 2, SIMD_m64(), ["MMX"], weight)

    def generate_instruction(self):
        global MIN_MAX_COUNT

        cmp_tmp = "_cmp_tmp{}".format(int(MIN_MAX_COUNT / 2))
        instruction = ""
        if MIN_MAX_COUNT % 2 == 0:
            instruction = "__m64 [TMP0] = _mm_set1_pi16(1 << 15);"
            instruction += "\n"
            instruction += "__m64 [CMP_TMP] = _mm_cmpgt_pi16(_mm_xor_si64([V1], [TMP0]), _mm_xor_si64([V2], [TMP0]));".replace(
                "[CMP_TMP]", cmp_tmp)
            instruction += "\n"
        instruction += "_mm_or_si64(_mm_and_si64([CMP_TMP], [V2]), _mm_andnot_si64([CMP_TMP], [V1]))".replace(
            "[CMP_TMP]", cmp_tmp)

        MIN_MAX_COUNT += 1
        return instruction


class SIMD_Min_Fallback_s64(SIMD_Instruction):
    def __init__(self, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.SIGNED, 8, simd_type, constraints, weight)

    def generate_instruction(self):
        global MIN_MAX_COUNT

        cmp_tmp = "_cmp_tmp{}".format(int(MIN_MAX_COUNT / 2))
        instruction = ""
        if MIN_MAX_COUNT % 2 == 0:
            instruction = "{} [CMP_TMP] = {}_cmpgt_epi64([V1], [V2]);".format(
                self.simd_type.to_string(),
                self.simd_type.prefix()).replace("[CMP_TMP]", cmp_tmp)
            instruction += "\n"

        instruction += "{}_blendv_epi8([V1], [V2], [CMP_TMP])".format(
            self.simd_type.prefix()).replace("[CMP_TMP]", cmp_tmp)

        MIN_MAX_COUNT += 1
        return instruction


class SIMD_Min_Fallback_u64(SIMD_Instruction):
    def __init__(self, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.UNSIGNED, 8, simd_type, constraints, weight)

    def generate_instruction(self):
        global MIN_MAX_COUNT

        cmp_tmp = "_cmp_tmp{}".format(int(MIN_MAX_COUNT / 2))
        instruction = ""
        if MIN_MAX_COUNT % 2 == 0:
            instruction = "{} [TMP0] = {}_set1_epi64x((1UL) << 63);\n".format(
                self.simd_type.to_string(), self.simd_type.prefix())
            instruction += "{} [CMP_TMP] = {}_cmpgt_epi64({}_xor_{}([V1], [TMP0]), {}_xor_{}([V2], [TMP0]));\n".format(
                self.simd_type.to_string(), self.simd_type.prefix(),
                self.simd_type.prefix(), self.simd_type.postfix(),
                self.simd_type.prefix(),
                self.simd_type.postfix()).replace("[CMP_TMP]", cmp_tmp)

        instruction += "{}_blendv_epi8([V1], [V2], [CMP_TMP])".format(
            self.simd_type.prefix()).replace("[CMP_TMP]", cmp_tmp)
        MIN_MAX_COUNT += 1
        return instruction


class SIMD_Min():
    def __init__(self):
        self.instructions = [
            ###################################################################
            SIMD_Instruction("_mm_min_pu8([V1], [V2])", Sign.UNSIGNED, 1,
                             SIMD_m64(), ["SSE"], 0),
            SIMD_Instruction("_mm_min_pi16([V1], [V2])", Sign.SIGNED, 2,
                             SIMD_m64(), ["SSE"], 0),
            # Fallbacks
            SIMD_Min_Fallback_m64_s8(1),
            SIMD_Min_Fallback_m64_u16(1),
            ###################################################################
            SIMD_Instruction("_mm_min_epi8([V1], [V2])", Sign.SIGNED, 1,
                             SIMD_m128(), ["SSE4.1"], 0),
            SIMD_Instruction("_mm_min_epu8([V1], [V2])", Sign.UNSIGNED, 1,
                             SIMD_m128(), ["SSE2"], 0),
            SIMD_Instruction("_mm_min_epi16([V1], [V2])", Sign.SIGNED, 2,
                             SIMD_m128(), ["SSE2"], 0),
            SIMD_Instruction("_mm_min_epu16([V1], [V2])", Sign.UNSIGNED, 2,
                             SIMD_m128(), ["SSE4.1"], 0),
            SIMD_Instruction("_mm_min_epi32([V1], [V2])", Sign.SIGNED, 4,
                             SIMD_m128(), ["SSE4.1"], 0),
            SIMD_Instruction("_mm_min_epu32([V1], [V2])", Sign.UNSIGNED, 4,
                             SIMD_m128(), ["SSE4.1"], 0),
            SIMD_Instruction("_mm_min_epi64([V1], [V2])", Sign.SIGNED, 8,
                             SIMD_m128(), ["AVX512vl", "AVX512f"], 0),
            SIMD_Instruction("_mm_min_epu64([V1], [V2])", Sign.UNSIGNED, 8,
                             SIMD_m128(), ["AVX512vl", "AVX512f"], 0),
            # Fallbacks
            SIMD_Min_Fallback_s64(SIMD_m128(), ["SSE4.2", "SSE4.1"], 1),
            SIMD_Min_Fallback_u64(SIMD_m128(), ["SSE2", "SSE4.2", "SSE4.1"],
                                  1),
            ###################################################################
            SIMD_Instruction("_mm256_min_epi8([V1], [V2])", Sign.SIGNED, 1,
                             SIMD_m256(), ["AVX2"], 0),
            SIMD_Instruction("_mm256_min_epu8([V1], [V2])", Sign.UNSIGNED, 1,
                             SIMD_m256(), ["AVX2"], 0),
            SIMD_Instruction("_mm256_min_epi16([V1], [V2])", Sign.SIGNED, 2,
                             SIMD_m256(), ["AVX2"], 0),
            SIMD_Instruction("_mm256_min_epu16([V1], [V2])", Sign.UNSIGNED, 2,
                             SIMD_m256(), ["AVX2"], 0),
            SIMD_Instruction("_mm256_min_epi32([V1], [V2])", Sign.SIGNED, 4,
                             SIMD_m256(), ["AVX2"], 0),
            SIMD_Instruction("_mm256_min_epu32([V1], [V2])", Sign.UNSIGNED, 4,
                             SIMD_m256(), ["AVX2"], 0),
            SIMD_Instruction("_mm256_min_epi64([V1], [V2])", Sign.SIGNED, 8,
                             SIMD_m256(), ["AVX512vl", "AVX512f"], 0),
            SIMD_Instruction("_mm256_min_epu64([V1], [V2])", Sign.UNSIGNED, 8,
                             SIMD_m256(), ["AVX512vl", "AVX512f"], 0),
            # Fallbacks
            SIMD_Min_Fallback_s64(SIMD_m256(), ["AVX2"], 1),
            SIMD_Min_Fallback_u64(SIMD_m256(), ["AVX", "AVX2"], 1),
            ###################################################################
            SIMD_Instruction("_mm512_min_epi8([V1], [V2])", Sign.SIGNED, 1,
                             SIMD_m512(), ["AVX512bw"], 0),
            SIMD_Instruction("_mm512_min_epu8([V1], [V2])", Sign.UNSIGNED, 1,
                             SIMD_m512(), ["AVX512bw"], 0),
            SIMD_Instruction("_mm512_min_epi16([V1], [V2])", Sign.SIGNED, 2,
                             SIMD_m512(), ["AVX512bw"], 0),
            SIMD_Instruction("_mm512_min_epu16([V1], [V2])", Sign.UNSIGNED, 2,
                             SIMD_m512(), ["AVX512bw"], 0),
            SIMD_Instruction("_mm512_min_epi32([V1], [V2])", Sign.SIGNED, 4,
                             SIMD_m512(), ["AVX512f"], 0),
            SIMD_Instruction("_mm512_min_epu32([V1], [V2])", Sign.UNSIGNED, 4,
                             SIMD_m512(), ["AVX512f"], 0),
            SIMD_Instruction("_mm512_min_epi64([V1], [V2])", Sign.SIGNED, 8,
                             SIMD_m512(), ["AVX512f"], 0),
            SIMD_Instruction("_mm512_min_epu64([V1], [V2])", Sign.UNSIGNED, 8,
                             SIMD_m512(), ["AVX512f"], 0)
            ###################################################################
        ]


######################################################################
# Maximum


# These are all simple instructions. A few fallbacks in the case of
# __m64 instructions and epi64 instructions if AVX512 is not
# present. But instruction selection entirely depends on what
# instructions the CPU has, not based on the input.
######################################################################
class SIMD_Max_Fallback_m64_s8(SIMD_Instruction):
    def __init__(self, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.SIGNED, 1, SIMD_m64(), ["MMX"], weight)

    def generate_instruction(self):
        global MIN_MAX_COUNT

        cmp_tmp = "_cmp_tmp{}".format(int(MIN_MAX_COUNT / 2))
        instruction = ""
        if MIN_MAX_COUNT % 2 == 0:
            instruction = "__m64 [CMP_TMP] = _mm_cmpgt_pi8([V1], [V2]);".replace(
                "[CMP_TMP]", cmp_tmp)
            instruction += "\n"
        instruction += "_mm_or_si64(_mm_and_si64([CMP_TMP], [V1]), _mm_andnot_si64([CMP_TMP], [V2]))".replace(
            "[CMP_TMP]", cmp_tmp)

        MIN_MAX_COUNT += 1
        return instruction


class SIMD_Max_Fallback_m64_u16(SIMD_Instruction):
    def __init__(self, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.UNSIGNED, 2, SIMD_m64(), ["MMX"], weight)

    def generate_instruction(self):
        global MIN_MAX_COUNT

        cmp_tmp = "_cmp_tmp{}".format(int(MIN_MAX_COUNT / 2))
        instruction = ""
        if MIN_MAX_COUNT % 2 == 0:
            instruction = "__m64 [TMP0] = _mm_set1_pi16(1 << 15);"
            instruction += "\n"
            instruction += "__m64 [CMP_TMP] = _mm_cmpgt_pi16(_mm_xor_si64([V1], [TMP0]), _mm_xor_si64([V2], [TMP0]));".replace(
                "[CMP_TMP]", cmp_tmp)
            instruction += "\n"
        instruction += "_mm_or_si64(_mm_and_si64([CMP_TMP], [V1]), _mm_andnot_si64([CMP_TMP], [V2]))".replace(
            "[CMP_TMP]", cmp_tmp)

        MIN_MAX_COUNT += 1
        return instruction


class SIMD_Max_Fallback_s64(SIMD_Instruction):
    def __init__(self, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.SIGNED, 8, simd_type, constraints, weight)

    def generate_instruction(self):
        global MIN_MAX_COUNT

        cmp_tmp = "_cmp_tmp{}".format(int(MIN_MAX_COUNT / 2))
        instruction = ""
        if MIN_MAX_COUNT % 2 == 0:
            instruction = "{} [CMP_TMP] = {}_cmpgt_epi64([V1], [V2]);".format(
                self.simd_type.to_string(),
                self.simd_type.prefix()).replace("[CMP_TMP]", cmp_tmp)
            instruction += "\n"
        instruction += "{}_blendv_epi8([V2], [V1], [CMP_TMP])".format(
            self.simd_type.prefix()).replace("[CMP_TMP]", cmp_tmp)

        MIN_MAX_COUNT += 1
        return instruction


class SIMD_Max_Fallback_u64(SIMD_Instruction):
    def __init__(self, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.UNSIGNED, 8, simd_type, constraints, weight)

    def generate_instruction(self):
        global MIN_MAX_COUNT

        cmp_tmp = "_cmp_tmp{}".format(int(MIN_MAX_COUNT / 2))
        instruction = ""
        if MIN_MAX_COUNT % 2 == 0:
            instruction = "{} [TMP0] = {}_set1_epi64x((1UL) << 63);\n".format(
                self.simd_type.to_string(), self.simd_type.prefix())
            instruction += "{} [CMP_TMP] = {}_cmpgt_epi64({}_xor_{}([V1], [TMP0]), {}_xor_{}([V2], [TMP0]));\n".format(
                self.simd_type.to_string(), self.simd_type.prefix(),
                self.simd_type.prefix(), self.simd_type.postfix(),
                self.simd_type.prefix(),
                self.simd_type.postfix()).replace("[CMP_TMP]", cmp_tmp)
        instruction += "{}_blendv_epi8([V2], [V1], [CMP_TMP])".format(
            self.simd_type.prefix()).replace("[CMP_TMP]", cmp_tmp)

        MIN_MAX_COUNT += 1
        return instruction


class SIMD_Max():
    def __init__(self):
        self.instructions = [
            ###################################################################
            SIMD_Instruction("_mm_max_pu8([V1], [V2])", Sign.UNSIGNED, 1,
                             SIMD_m64(), ["SSE"], 0),
            SIMD_Instruction("_mm_max_pi16([V1], [V2])", Sign.SIGNED, 2,
                             SIMD_m64(), ["SSE"], 0),
            # Fallbacks
            SIMD_Max_Fallback_m64_s8(1),
            SIMD_Max_Fallback_m64_u16(1),
            ###################################################################
            SIMD_Instruction("_mm_max_epi8([V1], [V2])", Sign.SIGNED, 1,
                             SIMD_m128(), ["SSE4.1"], 0),
            SIMD_Instruction("_mm_max_epu8([V1], [V2])", Sign.UNSIGNED, 1,
                             SIMD_m128(), ["SSE2"], 0),
            SIMD_Instruction("_mm_max_epi16([V1], [V2])", Sign.SIGNED, 2,
                             SIMD_m128(), ["SSE2"], 0),
            SIMD_Instruction("_mm_max_epu16([V1], [V2])", Sign.UNSIGNED, 2,
                             SIMD_m128(), ["SSE4.1"], 0),
            SIMD_Instruction("_mm_max_epi32([V1], [V2])", Sign.SIGNED, 4,
                             SIMD_m128(), ["SSE4.1"], 0),
            SIMD_Instruction("_mm_max_epu32([V1], [V2])", Sign.UNSIGNED, 4,
                             SIMD_m128(), ["SSE4.1"], 0),
            SIMD_Instruction("_mm_max_epi64([V1], [V2])", Sign.SIGNED, 8,
                             SIMD_m128(), ["AVX512vl", "AVX512f"], 0),
            SIMD_Instruction("_mm_max_epu64([V1], [V2])", Sign.UNSIGNED, 8,
                             SIMD_m128(), ["AVX512vl", "AVX512f"], 0),
            # Fallbacks
            SIMD_Max_Fallback_s64(SIMD_m128(), ["SSE4.2", "SSE4.1"], 1),
            SIMD_Max_Fallback_u64(SIMD_m128(), ["SSE2", "SSE4.2", "SSE4.1"],
                                  1),
            ###################################################################
            SIMD_Instruction("_mm256_max_epi8([V1], [V2])", Sign.SIGNED, 1,
                             SIMD_m256(), ["AVX2"], 0),
            SIMD_Instruction("_mm256_max_epu8([V1], [V2])", Sign.UNSIGNED, 1,
                             SIMD_m256(), ["AVX2"], 0),
            SIMD_Instruction("_mm256_max_epi16([V1], [V2])", Sign.SIGNED, 2,
                             SIMD_m256(), ["AVX2"], 0),
            SIMD_Instruction("_mm256_max_epu16([V1], [V2])", Sign.UNSIGNED, 2,
                             SIMD_m256(), ["AVX2"], 0),
            SIMD_Instruction("_mm256_max_epi32([V1], [V2])", Sign.SIGNED, 4,
                             SIMD_m256(), ["AVX2"], 0),
            SIMD_Instruction("_mm256_max_epu32([V1], [V2])", Sign.UNSIGNED, 4,
                             SIMD_m256(), ["AVX2"], 0),
            SIMD_Instruction("_mm256_max_epi64([V1], [V2])", Sign.SIGNED, 8,
                             SIMD_m256(), ["AVX512vl", "AVX512f"], 0),
            SIMD_Instruction("_mm256_max_epu64([V1], [V2])", Sign.UNSIGNED, 8,
                             SIMD_m256(), ["AVX512vl", "AVX512f"], 0),
            # Fallbacks
            SIMD_Max_Fallback_s64(SIMD_m256(), ["AVX2"], 1),
            SIMD_Max_Fallback_u64(SIMD_m256(), ["AVX", "AVX2"], 1),
            ###################################################################
            SIMD_Instruction("_mm512_max_epi8([V1], [V2])", Sign.SIGNED, 1,
                             SIMD_m512(), ["AVX512bw"], 0),
            SIMD_Instruction("_mm512_max_epu8([V1], [V2])", Sign.UNSIGNED, 1,
                             SIMD_m512(), ["AVX512bw"], 0),
            SIMD_Instruction("_mm512_max_epi16([V1], [V2])", Sign.SIGNED, 2,
                             SIMD_m512(), ["AVX512bw"], 0),
            SIMD_Instruction("_mm512_max_epu16([V1], [V2])", Sign.UNSIGNED, 2,
                             SIMD_m512(), ["AVX512bw"], 0),
            SIMD_Instruction("_mm512_max_epi32([V1], [V2])", Sign.SIGNED, 4,
                             SIMD_m512(), ["AVX512f"], 0),
            SIMD_Instruction("_mm512_max_epu32([V1], [V2])", Sign.UNSIGNED, 4,
                             SIMD_m512(), ["AVX512f"], 0),
            SIMD_Instruction("_mm512_max_epi64([V1], [V2])", Sign.SIGNED, 8,
                             SIMD_m512(), ["AVX512f"], 0),
            SIMD_Instruction("_mm512_max_epu64([V1], [V2])", Sign.UNSIGNED, 8,
                             SIMD_m512(), ["AVX512f"], 0)
            ###################################################################
        ]


######################################################################
# Blend

# Minimal optimization logic. Choice order is:
# 1: blend_epi32 -> requires that as movement in can grouped
# 2: blend_epi16 (if epi8/epi16) -> ibid
# 3: mask_mov -> requires AVX512
# 4: blendv_epi8 -> fallback basically.
######################################################################


def blend_mask_lt_T(perm, T_size, T_target):
    N = len(perm)
    T_size = T_size
    T_target = T_target

    scale = int(T_target / T_size)
    err_assert(scale == 1 or scale == 2 or scale == 4, "invalid scale")

    blend_mask = int(0)

    # logic here is basically we build epi32 mask and check that
    # all adjacent epi8/epi16 in a given epi32 map the same
    # way. If they don't return -1 to indicate failure
    run_idx = 0
    run = 0
    for i in range(0, N):
        run_idx += 1
        if perm[(N - 1) - i] > i:
            run += 1
        else:
            run -= 1
        if run_idx == scale:

            # run can only be scale or -scale if
            # all epi8/epi16 mapped the same way
            if run == scale:
                blend_mask |= (int(1) << (int(i / scale)))
            elif run == ((-1) * scale):
                # do nothing
                blend_mask = blend_mask
            else:
                return int(-1)
            run = 0
            run_idx = 0

    return blend_mask


def blend_mask_ge_T(perm, T_size, T_target):
    N = len(perm)
    T_size = T_size
    T_target = T_target

    scale = int(T_size / T_target)
    scaled_mask = (int(1) << scale) - 1
    err_assert(scale == 1 or scale == 2 or scale == 4, "invalid scale")

    blend_mask = int(0)
    for i in range(0, N):
        if perm[(N - 1) - i] > i:
            blend_mask |= scaled_mask << (scale * i)

    return blend_mask


class SIMD_Blend_Generator(SIMD_Instruction):
    def __init__(self, iname, T_size, T_target, perm, simd_type, constraints,
                 weight):
        super().__init__(iname, Sign.NOT_SIGNED, T_size, simd_type,
                         constraints, weight)

        self.T_target = T_target
        self.perm = copy.deepcopy(perm)

    def valid_blend(self, blend_mask):
        if blend_mask == int(-1):
            return False

        N_bits = int((self.simd_type.sizeof() / self.T_target))
        full_mask = (int(1) << N_bits) - 1
        err_assert(
            blend_mask != 0 and blend_mask != full_mask,
            "MISSED OPTIMIZATION AT BLEND: {}\n\tPerm: {}".format(
                str(hex(blend_mask)), str(self.perm)))

        if self.T_target == self.T_size:
            return True
        elif self.T_target == 4:
            return blend_mask != (int(-1))
        else:
            err_assert(False, "invalid blend target")

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and self.valid_blend(self.blend_mask())

    def blend_mask(self):
        N = self.simd_type.sizeof()
        T_size = self.T_size

        err_assert(T_size * len(self.perm) != 8,
                   "Cant do blend_epi8 for __m64")
        err_assert(T_size * len(self.perm) == N, "Invalid permutation map")

        if T_size < self.T_target:
            return blend_mask_lt_T(self.perm, self.T_size, self.T_target)
        else:
            return blend_mask_ge_T(self.perm, self.T_size, self.T_target)

    def generate_instruction(self):
        return self.iname.replace("[BLEND_MASK]", str(hex(self.blend_mask())))


class SIMD_Blend_As_Epi16_Generator(SIMD_Instruction):
    def __init__(self, T_size, perm, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)

        self.perm = copy.deepcopy(perm)

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and self.valid_blend(self.blend_mask())

    def valid_blend(self, blend_mask):
        if blend_mask == int(-1):
            return False

        N_bits = int((self.simd_type.sizeof() / 2))
        full_mask = (int(1) << N_bits) - 1
        err_assert(
            blend_mask != 0 and blend_mask != full_mask,
            "MISSED OPTIMIZATION AT BLEND: {}\n\tPerm: {}".format(
                str(hex(blend_mask)), str(self.perm)))

        lane_size = 16
        ele_per_lane = 8
        lane_mask = 0xff

        N = self.simd_type.sizeof()
        N_lanes = int(N / lane_size)

        lane_0 = blend_mask & lane_mask
        for i in range(1, N_lanes):
            if ((blend_mask >> (ele_per_lane * i)) & lane_mask) != lane_0:
                return False
        return True

    def blend_mask(self):
        N = self.simd_type.sizeof()
        T_size = self.T_size

        err_assert(T_size * len(self.perm) != 8, "Cant do for __m64")
        err_assert(T_size * len(self.perm) == N, "Invalid permutation map")

        if T_size < 2:
            return blend_mask_lt_T(self.perm, self.T_size, 2)
        else:
            return blend_mask_ge_T(self.perm, self.T_size, 2)

    def generate_instruction(self):
        return "{}_blend_epi16([V1], [V2], [BLEND_MASK])".format(
            self.simd_type.prefix()).replace(
                "[BLEND_MASK]", str(hex(self.blend_mask() & 0xff)))


class SIMD_Blend_As_Epi8_Generator(SIMD_Instruction):
    def __init__(self, T_size, perm, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)

        self.perm = copy.deepcopy(perm)

    def blend_vec(self):
        N = len(self.perm)
        T_size = self.T_size

        err_assert(self.simd_type.sizeof() != 8,
                   "Cant use blend_epi8 for __m64")

        n_shift_v = 0
        vec = []
        for i in range(0, N):
            shift_v = 0
            if self.perm[(N - 1) - i] > i:
                shift_v = 1
            n_shift_v += shift_v
            for j in range(0, T_size):
                vec.append(shift_v << 7)

        err_assert(
            n_shift_v != 0 and n_shift_v != N,
            "MISSED OPTIMIZATION AT BLEND_EPI8: {}".format(str(self.perm)))

        ret_vec = []
        for i in range(0, len(vec)):
            ret_vec.append(vec[len(vec) - (i + 1)])
        return ret_vec

    def generate_instruction(self):
        vec = self.blend_vec()
        list_vec = arr_to_csv(vec)

        instruction = "{}_blendv_epi8([V1], [V2], {}_set_epi8([BLEND_VEC]))".format(
            self.simd_type.prefix(),
            self.simd_type.prefix()).replace("[BLEND_VEC]", list_vec)
        return instruction


class SIMD_Blend_m64_Generator(SIMD_Instruction):
    def __init__(self, T_size, perm, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)

        self.perm = copy.deepcopy(perm)

    def blend_mask(self):
        N = len(self.perm)
        T_size = self.T_size

        err_assert(T_size * N == 8, "Blend_m64 used for non __m64 blend")
        err_assert(T_size * N == self.simd_type.sizeof(),
                   "Invalid permutation map")

        scale = 8 * T_size
        scale_mask = (int(1) << scale) - 1

        blend_mask = int(0)
        for i in range(0, N):
            if self.perm[(N - 1) - i] > i:
                blend_mask |= scale_mask << (i * scale)

        return blend_mask

    def generate_instruction(self):
        instruction = "{} [TMP0] = (__m64)([BLEND_MASK]UL);".format(
            self.simd_type.to_string()).replace("[BLEND_MASK]",
                                                str(hex(self.blend_mask())))
        instruction += "\n"
        instruction += "_mm_or_si64(_mm_and_si64([TMP0], [V2]), _mm_andnot_si64([TMP0], [V1]))"
        return instruction


class SIMD_Blend():
    def __init__(self, perm):
        self.instructions = [
            ###################################################################
            SIMD_Blend_m64_Generator(1, perm, SIMD_m64(), ["MMX"], 0),
            SIMD_Blend_m64_Generator(2, perm, SIMD_m64(), ["MMX"], 0),
            ###################################################################
            # __m128i epi8 ordering
            SIMD_Blend_Generator("_mm_blend_epi32([V1], [V2], [BLEND_MASK])",
                                 1, 4, perm, SIMD_m128(), ["AVX2"], 0),
            SIMD_Blend_As_Epi16_Generator(1, perm, SIMD_m128(), ["SSE4.1"], 1),
            SIMD_Blend_Generator("_mm_mask_mov_epi8([V1], [BLEND_MASK], [V2])",
                                 1, 1, perm, SIMD_m128(),
                                 ["AVX512vl", "AVX512bw"], 2),
            SIMD_Blend_As_Epi8_Generator(1, perm, SIMD_m128(), ["SSE4.1"], 3),
            # __m128i epi16 ordering
            SIMD_Blend_Generator("_mm_blend_epi32([V1], [V2], [BLEND_MASK])",
                                 2, 4, perm, SIMD_m128(), ["AVX2"], 0),
            SIMD_Blend_As_Epi16_Generator(2, perm, SIMD_m128(), ["SSE4.1"], 1),
            SIMD_Blend_Generator(
                "_mm_mask_mov_epi16([V1], [BLEND_MASK], [V2])", 2, 2, perm,
                SIMD_m128(), ["AVX512vl", "AVX512bw"], 2),
            SIMD_Blend_As_Epi8_Generator(2, perm, SIMD_m128(), ["SSE4.1"], 3),
            # __m128i epi32 ordering
            SIMD_Blend_Generator("_mm_blend_epi32([V1], [V2], [BLEND_MASK])",
                                 4, 4, perm, SIMD_m128(), ["AVX2"], 0),
            SIMD_Blend_Generator(
                "_mm_mask_mov_epi32([V1], [BLEND_MASK], [V2])", 4, 4, perm,
                SIMD_m128(), ["AVX512vl", "AVX512f"], 1),
            SIMD_Blend_As_Epi8_Generator(4, perm, SIMD_m128(), ["SSE4.1"], 2),
            # __m128i epi64 ordering
            SIMD_Blend_Generator("_mm_blend_epi32([V1], [V2], [BLEND_MASK])",
                                 8, 4, perm, SIMD_m128(), ["AVX2"], 0),
            SIMD_Blend_Generator(
                "_mm_mask_mov_epi64([V1], [BLEND_MASK], [V2])", 8, 8, perm,
                SIMD_m128(), ["AVX512vl", "AVX512f"], 1),
            SIMD_Blend_As_Epi8_Generator(8, perm, SIMD_m128(), ["SSE4.1"], 2),
            ###################################################################
            # __m256i epi8 ordering
            SIMD_Blend_Generator(
                "_mm256_blend_epi32([V1], [V2], [BLEND_MASK])", 1, 4, perm,
                SIMD_m256(), ["AVX2"], 0),
            SIMD_Blend_As_Epi16_Generator(1, perm, SIMD_m256(), ["AVX2"], 1),
            SIMD_Blend_Generator(
                "_mm256_mask_mov_epi8([V1], [BLEND_MASK], [V2])", 1, 1, perm,
                SIMD_m256(), ["AVX512vl", "AVX512bw"], 2),
            SIMD_Blend_As_Epi8_Generator(1, perm, SIMD_m256(), ["AVX2"], 3),
            # __m256i epi16 ordering
            SIMD_Blend_Generator(
                "_mm256_blend_epi32([V1], [V2], [BLEND_MASK])", 2, 4, perm,
                SIMD_m256(), ["AVX2"], 0),
            SIMD_Blend_As_Epi16_Generator(2, perm, SIMD_m256(), ["AVX2"], 1),
            SIMD_Blend_Generator(
                "_mm256_mask_mov_epi16([V1], [BLEND_MASK], [V2])", 2, 2, perm,
                SIMD_m256(), ["AVX512vl", "AVX512bw"], 2),
            SIMD_Blend_As_Epi8_Generator(2, perm, SIMD_m256(), ["AVX2"], 3),
            # __m256i epi32 ordering
            SIMD_Blend_Generator(
                "_mm256_blend_epi32([V1], [V2], [BLEND_MASK])", 4, 4, perm,
                SIMD_m256(), ["AVX2"], 0),
            SIMD_Blend_Generator(
                "_mm256_mask_mov_epi32([V1], [BLEND_MASK], [V2])", 4, 4, perm,
                SIMD_m256(), ["AVX512vl", "AVX512f"], 1),
            SIMD_Blend_As_Epi8_Generator(4, perm, SIMD_m256(), ["AVX2"], 2),
            # __m256i epi64 ordering
            SIMD_Blend_Generator(
                "_mm256_blend_epi32([V1], [V2], [BLEND_MASK])", 8, 4, perm,
                SIMD_m256(), ["AVX2"], 0),
            SIMD_Blend_Generator(
                "_mm256_mask_mov_epi64([V1], [BLEND_MASK], [V2])", 8, 8, perm,
                SIMD_m256(), ["AVX512vl", "AVX512f"], 1),
            SIMD_Blend_As_Epi8_Generator(8, perm, SIMD_m256(), ["AVX2"], 2),
            ###################################################################
            SIMD_Blend_Generator(
                "_mm512_mask_mov_epi8([V1], [BLEND_MASK], [V2])", 1, 1, perm,
                SIMD_m512(), ["AVX512bw"], 0),
            SIMD_Blend_Generator(
                "_mm512_mask_mov_epi16([V1], [BLEND_MASK], [V2])", 2, 2, perm,
                SIMD_m512(), ["AVX512bw"], 0),
            SIMD_Blend_Generator(
                "_mm512_mask_mov_epi32([V1], [BLEND_MASK], [V2])", 4, 4, perm,
                SIMD_m512(), ["AVX512f"], 0),
            SIMD_Blend_Generator(
                "_mm512_mask_mov_epi64([V1], [BLEND_MASK], [V2])", 8, 8, perm,
                SIMD_m512(), ["AVX512f"], 0),
            ###################################################################
        ]


######################################################################
# Permutate

# A fair amount of optimization logic here.  Generally we want to
# avoid permutex instructions as they require storing the indices in
# memory and operate across lanes. Optimization order is generally:

# 1: shuffle_epi32 -> requires elements to be in lane and same pattern
#                     across lanes
# 2: shuffle_epi16 -> uses to instructions but doesnt take global
#                     memory. Might prioritize epi8 above this
#                     depending on benchmarks
# 3: shuffle_epi8  -> uses global memory but Agner Fog says in lane >
#                     across lane (permutex) so we will go with that
# 4: permutex      -> global memory & across lanes so fallback

# Just about every SIMD register size and epi size has exceptions to
# this, however.

# One last note is the __builtin_shuffle and __builtin_shufflevector
# from Clang and GCC respectively both do this for us. Clang does a
# good job at this but GCC does not. I've tried to cover all the
# optimization cases that are relevant but you might find a few
# instances where Clang' builtin outperforms (or does worse). In
# general GCC always does equal or worse.

######################################################################

# Helper Functions


def shuffle_mask_impl(lane_size, ele_per_lane, slot_bits, offset, perm):

    N = len(perm)
    N_lanes = int(N / lane_size)

    err_assert(offset <= ele_per_lane, "invalid offset")

    mask = int(0)
    for i in range(0, N_lanes):
        lane_mask = int(0)
        lower_bound = N - (i * lane_size + ele_per_lane + offset)
        upper_bound = N - (i * lane_size + offset)
        for j in range(0, ele_per_lane):
            p = perm[i * lane_size + j + offset]
            err_assert(
                p >= lower_bound and p < upper_bound,
                "trying to build epi32 shuffle mask across lanes. This should have been checked for earlier"
            )
            idx = p - lower_bound
            err_assert(idx < 4, "invalid idx")
            slot = slot_bits * ((ele_per_lane - 1) - j)

            err_assert((lane_mask & (int(idx) << slot)) == 0,
                       "overlapping indices in shuffle_mask")
            lane_mask |= int(idx) << slot

        # checking for consistent pattern across lanes
        if i != 0 and mask != lane_mask:
            return int(-1)
        mask = lane_mask

    return mask


def grouped_by_lanes(lane_size, T_size, perm):
    N = len(perm)

    ele_per_lane = int(lane_size / T_size)
    N_lanes = int(N / ele_per_lane)

    for i in range(0, N_lanes):
        first_p = perm[i * ele_per_lane]
        target_lane = int(first_p / ele_per_lane)
        for j in range(ele_per_lane * i, ele_per_lane * (i + 1)):
            p = perm[j]
            if int(p / ele_per_lane) == target_lane:
                continue
            else:
                return False
    return True


def in_same_lanes(lane_size, T_size, perm):
    N = len(perm)

    ele_per_lane = int(lane_size / T_size)
    N_lanes = int(N / ele_per_lane)

    for i in range(0, N_lanes):
        lower_bound = N - ((i + 1) * ele_per_lane)
        upper_bound = N - ((i) * ele_per_lane)
        for j in range(ele_per_lane * i, ele_per_lane * (i + 1)):
            p = perm[j]
            if p >= lower_bound and p < upper_bound:
                continue
            else:
                return False
    return True


def can_shrink(from_T, to_T, perm):
    if from_T >= to_T:
        return True

    ele_per_chunk = int(to_T / from_T)
    N_chunks = int(len(perm) / ele_per_chunk)

    for i in range(0, N_chunks):
        base_ele = perm[i * ele_per_chunk]
        if base_ele % ele_per_chunk != (ele_per_chunk - 1):
            return False
        for j in range(0, ele_per_chunk):
            if perm[i * ele_per_chunk + j] != (base_ele - j):
                return False
    return True


def expand_perm(from_T, to_T, perm):
    scale = int(from_T / to_T)
    new_perm = []
    for p in perm:
        for j in range(0, scale):
            new_perm.append(scale * p + ((scale - 1) - (j % scale)))

    err_assert(len(perm) * scale == len(new_perm), "failed to expand perm")
    return new_perm


def shrink_perm(from_T, to_T, perm):
    scale = int(to_T / from_T)
    new_perm = []

    for i in range(0, (int(len(perm) / scale))):
        new_perm.append(int(perm[scale * i] / scale))

    err_assert(
        int(len(perm) / scale) == len(new_perm), "failed to expand perm")
    return new_perm


def scale_perm(from_T, to_T, perm):
    if from_T > to_T:
        return expand_perm(from_T, to_T, perm)
    elif from_T < to_T:
        return shrink_perm(from_T, to_T, perm)
    else:
        return perm


class SIMD_Shuffle_As_Epi64(SIMD_Instruction):
    def __init__(self, T_size, perm, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)
        self.perm = copy.deepcopy(perm)

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and (self.shuffle_mask() != int(-1))

    def shuffle_mask(self):
        err_assert(self.simd_type.sizeof() >= 32,
                   "Shuffle_As_Epi64 only valid for __m256i and __m512i")

        T_size = self.T_size
        if can_shrink(T_size, 8, self.perm) is False:
            return int(-1)

        new_perm = scale_perm(T_size, 8, self.perm)
        err_assert(
            len(new_perm) == int(self.simd_type.sizeof() / 8),
            "Any other value should be impossible")

        if self.simd_type.sizeof() == 64:
            if in_same_lanes(32, 8, new_perm) is False:
                return int(-1)

        mask = shuffle_mask_impl(4, 4, 2, 0, new_perm)
        return mask

    def generate_instruction(self):
        err_assert(self.shuffle_mask() != 0xe4, "Useless shuffle")
        if self.simd_type.sizeof() == 32:
            return "_mm256_permute4x64_epi64([V1], [SHUFFLE_MASK])".replace(
                "[SHUFFLE_MASK]", str(hex(self.shuffle_mask())))
        else:
            return "_mm512_permutex_epi64([V1], [SHUFFLE_MASK])".replace(
                "[SHUFFLE_MASK]", str(hex(self.shuffle_mask())))


class SIMD_Shuffle_As_si128(SIMD_Instruction):
    def __init__(self, T_size, perm, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)
        self.perm = copy.deepcopy(perm)

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and (self.shuffle_mask() != int(-1))

    def shuffle_mask(self):
        # epi64 will cover all of the needs of __m256i
        err_assert(self.simd_type.sizeof() > 32,
                   "Shuffle_As_Epi128 only valid for __m512i")

        T_size = self.T_size
        if can_shrink(T_size, 16, self.perm) is False:
            return int(-1)

        new_perm = scale_perm(T_size, 16, self.perm)
        err_assert(
            len(new_perm) == int(self.simd_type.sizeof() / 16),
            "Any other value should be impossible")

        mask = shuffle_mask_impl(4, 4, 2, 0, new_perm)
        return mask

    def generate_instruction(self):
        err_assert(self.shuffle_mask() != 0xe4, "Useless shuffle")
        return "_mm512_shuffle_i64x2([V1], [V1], [SHUFFLE_MASK])".replace(
            "[SHUFFLE_MASK]", str(hex(self.shuffle_mask())))


class SIMD_Shuffle2_As_PS(SIMD_Instruction):
    def __init__(self, T_size, perm, prev_perm, simd_type, constraints,
                 weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)

        self.shuf_mask = int(-1)
        self.perm = copy.deepcopy(perm)
        self.prev_perm = copy.deepcopy(prev_perm)
        self.vargs = ["", ""]

    def match(self, match_info):
        return super().match(match_info) and (self.shuffle_mask() != int(-1))

    def shuffle_mask(self):
        if len(self.prev_perm) != len(self.perm):
            return int(-1)

        shuf_epi32 = SIMD_Shuffle_As_Epi32(self.T_size, self.perm,
                                           self.simd_type, self.constraints,
                                           self.weight)
        shuf_mask = shuf_epi32.shuffle_mask()
        if shuf_mask == int(-1):
            return int(-1)
        if in_same_lanes(8, self.T_size, self.perm) is False:
            return int(-1)

        blend_mask = int(-1)
        using_perm = None
        if self.T_size < 4:
            # this was already checked in shuffle mask creation but what the hell, right?
            if can_shrink(self.T_size, 4, self.perm) is False:
                return int(-1)

            using_perm = shrink_perm(self.T_size, 4, self.perm)
            blend_mask = blend_mask_lt_T(self.prev_perm, self.T_size, 4)
        else:
            using_perm = expand_perm(self.T_size, 4, self.perm)
            blend_mask = blend_mask_ge_T(self.prev_perm, self.T_size,
                                         4)

        if blend_mask == int(-1):
            return int(-1)

        A_bit = int(-1)
        B_bit = int(-1)

        for i in range(0, len(using_perm), 2):
            blend_bit = ((blend_mask >> (i)) & 0x3)
            if blend_bit != 0x3 and blend_bit != 0:
                return int(-1)
            if i % 4 < 2:
                if A_bit == int(-1):
                    A_bit = blend_bit
                elif A_bit != blend_bit:
                    return int(-1)
            else:
                if B_bit == int(-1):
                    B_bit = blend_bit
                elif B_bit != blend_bit:
                    return int(-1)

            if A_bit == B_bit:
                return int(-1)

        if A_bit == 1:
            self.vargs[0] = "[V1]"
            self.vargs[1] = "[V2]"
        else:
            self.vargs[0] = "[V2]"
            self.vargs[1] = "[V1]"

        self.shuffle_mask = shuf_mask
        return shuf_mask

    def generate_instruction(self):
        err_assert(self.shuffle_mask != int(-1), "no mask for instruction")
        cast_to_ps = "{}_cast{}_ps".format(self.simd_type.prefix(),
                                           self.simd_type.postfix())
        cast_to_i = "{}_castps_{}".format(self.simd_type.prefix(),
                                          self.simd_type.postfix())

        instruction = "[CAST_TO_I]({}_shuffle_ps([CAST_TO_PS]([ARG1]), [CAST_TO_PS]([ARG2]), [SHUFFLE_MASK]))".format(
            self.simd_type.prefix())
        instruction = instruction.replace("[CAST_TO_I]", cast_to_i).replace(
            "[CAST_TO_PS]", cast_to_ps)
        instruction = instruction.replace("[ARG1]", self.vargs[0]).replace(
            "[ARG2]", self.vargs[1])
        instruction = instruction.replace("[SHUFFLE_MASK]",
                                          str(hex(self.shuffle_mask)))
        return instruction

        self.vargs = ["", ""]


class SIMD_Shuffle_As_Epi32(SIMD_Instruction):
    def __init__(self, T_size, perm, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)
        self.perm = copy.deepcopy(perm)

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and (self.shuffle_mask() != int(-1))

    def shuffle_mask(self):
        err_assert(self.simd_type.sizeof() != 8,
                   "shuffle_epi32 not available for __m64")

        T_size = self.T_size

        if in_same_lanes(16, T_size, self.perm) is False:
            return int(-1)

        if T_size < 4:
            return self.shuffle_mask_lt_T()
        else:
            return shuffle_mask_impl(4, 4, 2, 0,
                                     expand_perm(T_size, 4, self.perm))

    def shuffle_mask_lt_T(self):
        T_size = self.T_size

        if can_shrink(T_size, 4, self.perm) is False:
            return int(-1)

        return shuffle_mask_impl(4, 4, 2, 0, shrink_perm(T_size, 4, self.perm))

    def generate_instruction(self):
        err_assert(self.shuffle_mask() != 0xe4, "Useless shuffle")

        mask_t = "uint8_t"
        if self.simd_type.sizeof() == 64:
            mask_t = "_MM_PERM_ENUM"
        return "{}_shuffle_epi32([V1], [MASK_T]([SHUFFLE_MASK]))".format(
            self.simd_type.prefix()).replace("[MASK_T]", mask_t).replace(
                "[SHUFFLE_MASK]", str(hex(self.shuffle_mask())))


class SIMD_Shuffle_As_Epi16(SIMD_Instruction):
    def __init__(self, T_size, perm, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)
        self.perm = copy.deepcopy(perm)
        self.adjusted_weight = False

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and (self.shuffle_mask() != int(-1))

    def shuffle_mask(self):

        T_size = self.T_size

        # __m64 case
        if len(self.perm) <= 4 and T_size == 2 and self.simd_type.sizeof(
        ) == 8:
            return shuffle_mask_impl(4, 4, 2, 0, self.perm)

        err_assert(self.simd_type.sizeof() != 8,
                   "shuffle_epi16 not available for __m64")

        if in_same_lanes(8, T_size, self.perm) is False:
            return int(-1)

        if T_size < 2:
            return self.shuffle_mask_lt_T()
        else:
            return self.build_shuffle_mask(expand_perm(T_size, 2, self.perm))

    def shuffle_mask_lt_T(self):
        T_size = self.T_size

        if can_shrink(T_size, 2, self.perm) is False:
            return int(-1)

        return self.build_shuffle_mask(shrink_perm(T_size, 2, self.perm))

    def build_shuffle_mask(self, new_perm):
        shuffle_mask_lo = shuffle_mask_impl(8, 4, 2, 4, new_perm)
        shuffle_mask_hi = shuffle_mask_impl(8, 4, 2, 0, new_perm)
        if shuffle_mask_lo == int(-1) or shuffle_mask_hi == int(-1):
            return int(-1)

        if shuffle_mask_lo != 0xe4 and shuffle_mask_hi != 0xe4:
            if self.adjusted_weight is False:
                self.adjusted_weight = True
                self.weight += choose_if(Optimization.SPACE, 1, 2)
        return shuffle_mask_lo | (shuffle_mask_hi << 32)

    def generate_instruction(self):
        mask = self.shuffle_mask()
        shuffle_mask_lo = mask & 0xffffffff
        shuffle_mask_hi = (mask >> 32) & 0xffffffff
        if shuffle_mask_lo != 0xe4 and shuffle_mask_hi != 0xe4:
            if self.adjusted_weight is False:
                self.adjusted_weight = True
                self.weight += choose_if(Optimization.SPACE, 1, 2)
            return "{}_shufflehi_epi16({}_shufflelo_epi16([V1], [SHUFFLE_MASK_LO]), [SHUFFLE_MASK_HI])".format(
                self.simd_type.prefix(), self.simd_type.prefix()).replace(
                    "[SHUFFLE_MASK_LO]", str(hex(shuffle_mask_lo))).replace(
                        "[SHUFFLE_MASK_HI]", str(hex(shuffle_mask_hi)))
        elif shuffle_mask_lo == 0xe4:

            return "{}_shufflehi_epi16([V1], [SHUFFLE_MASK_HI])".format(
                self.simd_type.prefix(),
                self.simd_type.prefix()).replace("[SHUFFLE_MASK_HI]",
                                                 str(hex(shuffle_mask_hi)))

        elif shuffle_mask_hi == 0xe4:
            return "{}_shufflelo_epi16([V1], [SHUFFLE_MASK_LO])".format(
                self.simd_type.prefix(),
                self.simd_type.prefix()).replace("[SHUFFLE_MASK_LO]",
                                                 str(hex(shuffle_mask_lo)))
        else:
            err_assert(False, "missed optimization")


class SIMD_Shuffle_As_Epi8(SIMD_Instruction):
    def __init__(self, T_size, perm, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)
        self.perm = copy.deepcopy(perm)

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and in_same_lanes(
                    16, self.T_size, self.perm)

    def generate_instruction(self):
        scaled_perm = scale_perm(self.T_size, 1, self.perm)
        list_perm = arr_to_csv(scaled_perm)

        instruction = "{}_shuffle_epi8([V1], {}_set_epi8([SHUFFLE_VEC]))".format(
            self.simd_type.prefix(),
            self.simd_type.prefix()).replace("[SHUFFLE_VEC]", list_perm)

        return instruction


class SIMD_Alignr(SIMD_Instruction):
    def __init__(self, T_size, perm, prev_perm, simd_type, constraints,
                 weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)
        self.weight = int(-1000)
        self.prev_perm = copy.deepcopy(prev_perm)
        self.perm = copy.deepcopy(perm)
        self.shift_width = None
        self.adjusted_weight = False
        self.ll = False
        self.vargs = ["[V1]", "[V1]"]

    def match(self, match_info):
        return super().match(match_info) and self.valid_alignr()

    def valid_alignr(self):
        group_size = min(16, self.simd_type.sizeof())
        if in_same_lanes(group_size, self.T_size, self.perm) is False:
            return False

        cur_dst = (((len(self.perm) - 1)) - self.perm[0]) % int(
            group_size / self.T_size)
        for i in range(0, len(self.perm)):
            idst = (((len(self.perm) - 1) -
                     (i)) - self.perm[i]) % int(group_size / self.T_size)
            if cur_dst != idst:
                return False

        self.shift_width = cur_dst
        if len(self.prev_perm) != len(self.perm):
            if self.adjusted_weight is False:
                self.adjusted_weight = True
                self.weight += 1
            return True

        err_assert(len(self.prev_perm) == len(self.perm), "tabs are bad")

        blend_mask = blend_mask_ge_T(self.prev_perm, self.T_size, self.T_size)
        ll_lane_blend_mask = (int(1) << (self.shift_width)) - 1

        ll_blend_mask = int(0)
        for i in range(0, self.simd_type.sizeof(), group_size):
            ll_blend_mask |= (ll_lane_blend_mask << int(i / self.T_size))

        not_ll_blend_mask = (~ll_blend_mask) & (
            (int(1) << int(self.simd_type.sizeof() / self.T_size)) - 1)

        if blend_mask != ll_blend_mask and blend_mask != not_ll_blend_mask:
            if self.adjusted_weight is False:
                self.adjusted_weight = True
                self.weight += 1
            return True

        self.ll = True
        err_assert(False, "Waiting on this")
        if (blend_mask & 0x1) == int(0):
            self.vargs[0] = "[V1]"
            self.vargs[1] = "[V2]"
        else:
            self.vargs[0] = "[V2]"
            self.vargs[1] = "[V1]"

        return True

    def generate_instruction(self):
        epi = "pi"
        if self.simd_type.sizeof() != 8:
            epi = "e" + epi
        return "{}_alignr_{}8([ARG1], [ARG2], [SHIFT_WIDTH])".format(
            self.simd_type.prefix(),
            epi).replace("[ARG1]", self.vargs[0]).replace(
                "[ARG2]",
                self.vargs[1]).replace("[SHIFT_WIDTH]",
                                       str(self.shift_width * self.T_size))


class SIMD_rotate(SIMD_Instruction):
    def __init__(self, T_size, perm, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)
        self.perm = copy.deepcopy(perm)
        self.use_vec = False
        self.distances = None
        self.group_size = None
        self.adjusted_weight = False

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and (self.rotate_mask())

    def rotate_mask(self):
        t1 = self.rotate_mask_inner(4)
        if t1 is False:
            t1 = self.rotate_mask_inner(8)
        return t1

    def rotate_mask_inner(self, group_size):
        if group_size <= self.T_size:
            return False
        if in_same_lanes(group_size, self.T_size, self.perm) is False:
            return False

        group_size_mod = int(group_size / self.T_size)
        distances = []
        for g in range(0, len(self.perm), int(group_size / self.T_size)):
            cur_dst = ((
                (len(self.perm) - 1) - g) - self.perm[g]) % group_size_mod
            for i in range(1, int(group_size / self.T_size)):
                idst = (((len(self.perm) - 1) -
                         (g + i)) - self.perm[g + i]) % group_size_mod
                if cur_dst != idst:
                    return False
            shift = 8 * self.T_size * cur_dst
            if shift < 0:
                shift = 8 * self.T_size + shift
            distances.append(shift)

        first_d = distances[0]
        for d in distances:
            if d != first_d:
                if self.adjusted_weight is False:
                    self.use_vec = True
                    self.adjusted_weight = True
                    self.weight += 1

        self.group_size = group_size
        self.distances = copy.deepcopy(distances)
        return True

    def build_rotate_tail(self):
        if self.use_vec is True:
            epi_postfix = str(8 * self.group_size)
            if self.simd_type.sizeof() != 64 and self.group_size == 8:
                epi_postfix += "x"
            return "{}_set_epi{}([DISTANCES])".format(
                self.simd_type.prefix(),
                epi_postfix).replace("[DISTANCES]",
                                     arr_to_csv(self.distances, False))
        else:
            return str((self.distances[0]))

    def generate_instruction(self):
        postfix = ""
        if self.use_vec:
            postfix += "v"
        postfix += "_epi{}".format(8 * self.group_size)
        return "{}_ror{}([V1], {})".format(self.simd_type.prefix(), postfix,
                                           self.build_rotate_tail())


class SIMD_Shuffle_Rotate(SIMD_Instruction):
    def __init__(self, T_size, perm, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)

        self.perm = copy.deepcopy(perm)
        self.smask = None
        self.r_ins = ""
        self.adjusted_weight = False

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and (self.shuffle_mask() is True)

    def create_mod_perm(self, scaled_perm, perm_mask):
        modified_perm_list = []
        for i in range(0, len(scaled_perm)):
            modified_perm_list.append(int(-1))

        for l in range(0, self.simd_type.sizeof(), 16):
            for i in range(0, 8, 2):
                to_idx = 3 - int(i / 2)
                from_idx = 3 - ((perm_mask >> i) & 0x3)
                for j in range(0, 4):
                    tidx = to_idx * 4 + j + l
                    fidx = from_idx * 4 + j + l
                    modified_perm_list[tidx] = scaled_perm[fidx]

        for i in range(0, len(modified_perm_list)):
            err_assert(
                modified_perm_list[i] != int(-1),
                "index {} not correctly set\n\t{}\n\t{}\n\t{}".format(
                    i, perm_mask, str(self.perm), str(modified_perm_list)))

        return modified_perm_list

    def shuffle_mask(self):
        scaled_perm = scale_perm(self.T_size, 1, self.perm)
        if grouped_by_lanes(4, 1, scaled_perm) is False or grouped_by_lanes(
                16, 1, scaled_perm) is False:
            return False

        possibility = None
        ppmask = None
        all_perms = itertools.permutations([0, 1, 2, 3])
        for perm_lists in list(all_perms):
            pmask = perm_lists[0]
            pmask |= perm_lists[1] << 2
            pmask |= perm_lists[2] << 4
            pmask |= perm_lists[3] << 6
            if pmask == 0xe4:
                continue

            mplist = self.create_mod_perm(scaled_perm, pmask)

            rotate = SIMD_rotate(1, mplist, self.simd_type,
                                 ["AVX512f", "AVX512vl"], 0)
            if rotate.match(
                    Match_Info(Sort_Type(1, Sign.UNSIGNED),
                               self.simd_type)) is False:
                continue

            if rotate.weight != 0:
                if possibility is None:
                    possibility = rotate
                    ppmask = pmask
                continue

            self.smask = pmask
            self.r_ins = rotate.generate_instruction()
            return True

        if possibility is not None:
            if self.adjusted_weight is False:
                self.weight += possibility.weight
                self.adjusted_weight = True
            self.smask = ppmask
            self.r_ins = possibility.generate_instruction()
            return True
        return False

    def generate_instruction(self):
        err_assert(self.smask is not None, "something went wrong")
        err_assert(self.r_ins != "", "something went wrong")

        mask_t = "uint8_t"
        if self.simd_type.sizeof() == 64:
            mask_t = "_MM_PERM_ENUM"
        instruction = "{} [TMP0] = {}_shuffle_epi32([V1], [MASK_T]([SHUFFLE_MASK]));".format(
            self.simd_type.to_string(), self.simd_type.prefix()).replace(
                "[MASK_T]", mask_t).replace("[SHUFFLE_MASK]",
                                            str(hex(self.smask)))
        instruction += "\n"
        return instruction + self.r_ins.replace("[V1]", "[TMP0]")


class SIMD_Permute_Move_Lanes_Shuffle(SIMD_Instruction):
    def __init__(self, T_size, prev_perm, perm, simd_type, constraints,
                 weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)

        self.tmp_vargs = ["", ""]
        self.vargs = ["", ""]
        self.prev_perm = copy.deepcopy(prev_perm)
        self.perm = copy.deepcopy(perm)
        self.to_use = None
        self.perm_mask = None
        self.instruction_base = ""
        self.modified_weight = False
        self.use_LL = False

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and (self.shuffle_mask() != int(-1))

    def perm_shuffle_mask(self, lane_size):
        err_assert(self.simd_type.sizeof() >= 32,
                   "using lane reshuffle for register smaller than __m256i")

        scaled_perm = scale_perm(self.T_size, 1, self.perm)
        N = len(scaled_perm)

        ele_per_lane = int(lane_size)

        # either vpermq of vshuf64x2 (both of which have 4 elements per lane)
        N_lanes = 4

        mask = int(0)
        for i in range(0, N_lanes):
            first_p = scaled_perm[i * ele_per_lane]
            target_lane = int(first_p / ele_per_lane)
            from_lane = (N_lanes - 1) - i
            err_assert((mask & (int(from_lane) << (2 * target_lane))) == 0,
                       "overlapping lane assignments")
            mask |= int(from_lane) << (2 * target_lane)
        return mask

    def create_mod_perm(self, scaled_perm, perm_mask, lane_size):
        modified_perm_list = []
        for i in range(0, len(scaled_perm)):
            modified_perm_list.append(int(-1))

        for o in range(0, self.simd_type.sizeof(), 4 * lane_size):
            for i in range(0, 8, 2):
                to_idx = 3 - int(i / 2)
                from_idx = 3 - ((perm_mask >> i) & 0x3)
                for j in range(0, lane_size):
                    tidx = to_idx * lane_size + j + o
                    fidx = from_idx * lane_size + j + o
                    modified_perm_list[tidx] = scaled_perm[fidx]

        for i in range(0, len(modified_perm_list)):
            err_assert(
                modified_perm_list[i] != int(-1),
                "index {} not correctly set\n\t{}\n\t{}\n\t{}".format(
                    i, perm_mask, str(self.perm), str(modified_perm_list)))
        return modified_perm_list

    def test_possible_instructions(self, modified_perm_list):
        if in_same_lanes(16, 1, modified_perm_list) is False:
            return int(-1), None

        possible_instructions = []

        # ror is prioritized because its p0
        if self.simd_type.sizeof() == 32:
            possible_instructions.append(
                SIMD_rotate(1, modified_perm_list, SIMD_m256(),
                            ["AVX512f", "AVX512vl"], 0))
            possible_instructions.append(
                SIMD_Shuffle_As_Epi32(1, modified_perm_list, SIMD_m256(),
                                      ["AVX2"], 1))
            possible_instructions.append(
                SIMD_Shuffle_As_Epi16(1, modified_perm_list, SIMD_m256(),
                                      ["AVX2"], 1))
            possible_instructions.append(
                SIMD_Shuffle_As_Epi8(1, modified_perm_list,
                                     SIMD_m256(), ["AVX2"],
                                     choose_if(Optimization.UOP, 1, 2)))

        else:
            possible_instructions.append(
                SIMD_rotate(1, modified_perm_list, SIMD_m512(),
                            ["AVX512f", "AVX512vl"], 0))
            possible_instructions.append(
                SIMD_Shuffle_As_Epi32(1, modified_perm_list, SIMD_m512(),
                                      ["AVX512f"], 1))
            possible_instructions.append(
                SIMD_Shuffle_As_Epi16(1, modified_perm_list, SIMD_m512(),
                                      ["AVX512bw"], 1))
            possible_instructions.append(
                SIMD_Shuffle_As_Epi64(1, modified_perm_list, SIMD_m512(),
                                      ["AVX512f"], 3))

        min_idx = int(-1)
        for i in range(0, len(possible_instructions)):
            if possible_instructions[i].match(
                    Match_Info(Sort_Type(1, Sign.UNSIGNED),
                               self.simd_type)) is True:
                if min_idx == int(-1):
                    min_idx = i
                elif possible_instructions[
                        min_idx].weight > possible_instructions[i].weight:
                    min_idx = i
        if min_idx == int(-1):
            return int(-1), None
        else:
            return possible_instructions[
                min_idx].weight, possible_instructions[min_idx]

    def shuffle_mask(self):
        rets = []
        pmasks = []
        mweights = []
        perms = []
        ll_weight = []
        ibase = []
        if self.simd_type.sizeof() == 32:
            r, pmask, mweight, p = self.shuffle_mask_inner(8, True)
            rets.append(r)
            pmasks.append(0x21)
            mweights.append(mweight)
            if p is not None:
                perms.append(copy.deepcopy(p))
            else:
                perms.append([])
            ll_weight.append(0)
            ibase.append(
                "_mm256_permute2x128_si256([ARG1], [ARG2], [SHUFFLE_MASK])")

            r, pmask, mweight, p = self.shuffle_mask_inner(8, False)
            rets.append(r)
            pmasks.append(pmask)
            mweights.append(mweight)
            if p is not None:
                perms.append(copy.deepcopy(p))
            else:
                perms.append([])
            ll_weight.append(1)
            ibase.append("_mm256_permute4x64_epi64([V1], [SHUFFLE_MASK])")

        else:
            r, pmask, mweight, p = self.shuffle_mask_inner(16, True)
            rets.append(r)
            pmasks.append(pmask)
            mweights.append(mweight)
            if p is not None:
                perms.append(copy.deepcopy(p))
            else:
                perms.append([])
            ll_weight.append(0)
            ibase.append(
                "_mm512_shuffle_i64x2([ARG1], [ARG2], [SHUFFLE_MASK])")

            r, pmask, mweight, p = self.shuffle_mask_inner(8, False)
            rets.append(r)
            pmasks.append(pmask)
            mweights.append(mweight)
            if p is not None:
                perms.append(copy.deepcopy(p))
            else:
                perms.append([])
            ll_weight.append(1)
            ibase.append("_mm512_permutex_epi64([V1], [SHUFFLE_MASK])")

            r, pmask, mweight, p = self.shuffle_mask_inner(16, False)
            rets.append(r)
            pmasks.append(pmask)
            mweights.append(mweight)
            if p is not None:
                perms.append(copy.deepcopy(p))
            else:
                perms.append([])
            ll_weight.append(1)
            ibase.append("_mm512_shuffle_i64x2([V1], [V1], [SHUFFLE_MASK])")

        has_one = False
        best_w_idx = int(-1)
        for i in range(0, len(rets)):
            if rets[i] is False:
                continue

            has_one = True
            if best_w_idx == int(-1):
                best_w_idx = i
                continue

            if (mweights[i] + ll_weight[i]) < (mweights[best_w_idx] +
                                               ll_weight[best_w_idx]):
                best_w_idx = i

        if has_one is False:
            return int(-1)

        if self.modified_weight is False:
            self.weight += mweights[best_w_idx]
            self.weight += ll_weight[best_w_idx]
        self.to_use = copy.deepcopy(perms[best_w_idx])
        self.perm_mask = pmasks[best_w_idx]
        self.instruction_base = ibase[best_w_idx]
        return int(0)

    def ll_shuffle_mask(self, pperm, mperm):
        if len(pperm) != len(mperm):
            return False

        blend_mask = blend_mask_ge_T(pperm, 1, 1)

        if blend_mask == int(-1):
            return False

        ele_per_lane = 16

        lane_blend_mask = (int(1) << ele_per_lane) - 1
        for i in range(0, len(mperm), ele_per_lane):
            res = (blend_mask >> i) & lane_blend_mask
            if res != lane_blend_mask and res != 0:
                return False

        expec = 0
        for i in range(0, len(mperm)):
            if i % (int(len(mperm) / 2)) == 0:
                expec = (blend_mask >> (mperm[i])) & 0x1
                if expec == 0:
                    self.tmp_vargs[int(i / int(len(mperm) / 2))] = "[V2]"
                else:
                    self.tmp_vargs[int(i / int(len(mperm) / 2))] = "[V1]"

            cur = (blend_mask >> (mperm[i])) & 0x1
            if expec != cur:
                return False

        if self.tmp_vargs[0] == self.tmp_vargs[1]:
            return False

        return True

    def shuffle_mask_inner(self, group_size, LL):
        err_assert(self.simd_type.sizeof() >= 32,
                   "using lane reshuffle for register < __m256i")
        scaled_perm = scale_perm(self.T_size, 1, self.perm)
        tgroup_size = group_size
        if LL is True:
            tgroup_size = 16
        if grouped_by_lanes(tgroup_size, 1,
                            scaled_perm) is False or grouped_by_lanes(
                                4 * group_size, 1, scaled_perm) is False:
            return False, None, None, None

        scaled_prev_perm = []
        if LL is True and len(self.prev_perm) != len(self.perm):
            return False, None, None, None
        if LL is True:
            scaled_prev_perm = scale_perm(self.T_size, 1, self.prev_perm)

        min_weight = 100
        best_p = None
        best_perm_mask = 0

        all_perms = itertools.permutations([0, 1, 2, 3])
        for perm_lists in list(all_perms):
            pmask = perm_lists[0]
            pmask |= perm_lists[1] << 2
            pmask |= perm_lists[2] << 4
            pmask |= perm_lists[3] << 6
            if pmask == 0xe4:
                continue
            if LL is True and self.simd_type.sizeof() == 32 and pmask != 0x4e:
                continue

            if LL is True and self.ll_shuffle_mask(scaled_prev_perm,
                                                   scaled_perm) is False:
                continue

            rweight, p = self.test_possible_instructions(
                self.create_mod_perm(scaled_perm, pmask, group_size))
            if rweight == int(-1):
                continue
            elif rweight < min_weight:
                if LL is True:
                    self.vargs[0] = self.tmp_vargs[0]
                    self.vargs[1] = self.tmp_vargs[1]
                self.use_LL = LL
                min_weight = rweight
                best_p = p
                best_perm_mask = pmask

        if min_weight != 100:
            return True, best_perm_mask, min_weight, best_p

        return False, None, None, None

    def generate_instruction(self):
        instruction = "{} [TMP0] = {};".format(
            self.simd_type.to_string(), self.instruction_base).replace(
                "[SHUFFLE_MASK]", str(hex(self.perm_mask))).replace(
                    "[ARG1]", self.vargs[0]).replace("[ARG2]", self.vargs[1])
        instruction += "\n"
        instruction += self.to_use.generate_instruction().replace(
            "[V1]", "[TMP0]")
        return instruction


class SIMD_Permutex_Fallback(SIMD_Instruction):
    def __init__(self, T_size, perm, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)

        self.perm = copy.deepcopy(perm)

    def build_lanes_vec(self, cross_lane):
        lane_size = 16
        new_perm = scale_perm(self.T_size, 1, self.perm)

        N = len(new_perm)
        vec = []

        N_lanes = int(len(new_perm) / lane_size)
        for i in range(0, N_lanes):
            lower_bound = N - ((i + 1) * lane_size)
            upper_bound = N - ((i) * lane_size)
            for j in range(0, lane_size):
                p = new_perm[i * lane_size + j]
                # in correct lane
                if p >= lower_bound and p < upper_bound:
                    if cross_lane is True:
                        vec.append(1 << 7)
                    else:
                        vec.append(p % lane_size)
                else:
                    if cross_lane is True:
                        vec.append(p % lane_size)
                    else:
                        vec.append(1 << 7)
        return vec

    def generate_instruction(self):
        err_assert(self.simd_type.sizeof() == 32,
                   "Permutex_Fallback only for __m256i")

        same_lane_vec = self.build_lanes_vec(False)
        other_lane_vec = self.build_lanes_vec(True)
        err_assert(
            len(same_lane_vec) == len(other_lane_vec),
            "These should be the exact same length")

        same_lane_list_vec = arr_to_csv(same_lane_vec)
        other_lane_list_vec = arr_to_csv(other_lane_vec)

        instruction = "__m256i [TMP0] = _mm256_permute4x64_epi64([V1], 0x4e);"
        instruction += "\n"
        instruction += "__m256i [TMP1] = _mm256_shuffle_epi8([V1], _mm256_set_epi8([SAME_LANE_VEC]));".replace(
            "[SAME_LANE_VEC]", same_lane_list_vec)
        instruction += "\n"
        instruction += "__m256i [TMP2] = _mm256_shuffle_epi8([TMP0], _mm256_set_epi8([OTHER_LANE_VEC]));".replace(
            "[OTHER_LANE_VEC]", other_lane_list_vec)
        instruction += "\n"
        instruction += "_mm256_or_si256([TMP1], [TMP2])"
        return instruction


class SIMD_Shuffle_m64(SIMD_Instruction):
    def __init__(self, T_size, perm, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)
        self.perm = copy.deepcopy(perm)

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and (self.shuffle_mask() != int(-1))

    def shuffle_mask(self):
        err_assert(self.simd_type.sizeof() == 8,
                   "using __m64i shuffle for non __m64i register")
        if can_shrink(self.T_size, 2, self.perm) is False:
            return int(-1)

        new_perm = scale_perm(self.T_size, 2, self.perm)

        return shuffle_mask_impl(4, 4, 2, 0, new_perm)

    def generate_instruction(self):
        return "_mm_shuffle_pi16([V1], [SHUFFLE_MASK])".replace(
            "[SHUFFLE_MASK]", str(hex(self.shuffle_mask())))


class SIMD_Permutex_Generate(SIMD_Instruction):
    def __init__(self, iname, T_size, perm, simd_type, constraints, weight):
        super().__init__(iname, Sign.NOT_SIGNED, T_size, simd_type,
                         constraints, weight)
        self.perm = copy.deepcopy(perm)

    def generate_instruction(self):
        return self.iname.replace("[PERM_LIST]", arr_to_csv(self.perm))


class SIMD_Shuffle2_Blend(SIMD_Instruction):
    def __init__(self, T_size, prev_perm, perm, simd_type, constraints,
                 weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)
        self.perm = copy.deepcopy(perm)
        self.prev_perm = copy.deepcopy(prev_perm)

        self.vargs = ["", ""]

    def match(self, match_info):
        return super().match(match_info) and (self.shuffle_mask() != int(-1))

    def shuffle_mask(self):
        if len(self.prev_perm) != len(self.perm):
            return int(-1)

        if can_shrink(self.T_size, 16, self.perm) is False:
            return int(-1)

        blend_mask = int(-1)

        using_T_size = self.T_size
        using_perm = self.perm
        if self.T_size < 4:
            if can_shrink(self.T_size, 4, self.perm) is False:
                return int(-1)
            using_perm = shrink_perm(self.T_size, 4, self.perm)
            using_T_size = 4
            blend_mask = blend_mask_lt_T(self.prev_perm, self.T_size, 4)

        else:
            blend_mask = blend_mask_ge_T(self.prev_perm, self.T_size,
                                         self.T_size)
        if blend_mask == int(-1):
            return int(-1)

        ele_per_lane = int(16 / using_T_size)

        lane_blend_mask = (int(1) << ele_per_lane) - 1
        for i in range(0, len(using_perm), ele_per_lane):
            res = (blend_mask >> i) & lane_blend_mask
            if res != lane_blend_mask and res != 0:
                return int(-1)

        shrunk_perm = shrink_perm(self.T_size, 16, self.perm)
        shuf_mask = int(-1)
        if self.simd_type.sizeof() == 64:
            shuf_mask = shuffle_mask_impl(4, 4, 2, 0, shrunk_perm)
        else:
            shuf_mask = 0x21

        expec = 0
        for i in range(0, len(shrunk_perm)):
            if i % (int(len(shrunk_perm) / 2)) == 0:
                expec = (blend_mask >>
                         (ele_per_lane * shrunk_perm[i])) & lane_blend_mask
                if expec == 0:
                    self.vargs[int(i / int(len(shrunk_perm) / 2))] = "[V2]"
                else:
                    self.vargs[int(i / int(len(shrunk_perm) / 2))] = "[V1]"

            cur = (blend_mask >>
                   (ele_per_lane * shrunk_perm[i])) & lane_blend_mask
            if expec != cur:
                return int(-1)

        if self.vargs[0] == self.vargs[1]:
            return int(-1)

        return shuf_mask

    def generate_instruction(self):
        instruction = ""
        if self.simd_type.sizeof() == 32:
            instruction = "_mm256_permute2x128_si256([ARG1], [ARG2], [SHUFFLE_MASK])"
        else:
            instruction = "_mm512_shuffle_i64x2([ARG1], [ARG2], [SHUFFLE_MASK])"
        return instruction.replace("[ARG1]", self.vargs[0]).replace(
            "[ARG2]", self.vargs[1]).replace("[SHUFFLE_MASK]",
                                             str(hex(self.shuffle_mask())))


class SIMD_Permutex2_Blend(SIMD_Instruction):
    def __init__(self, T_size, prev_perm, perm, simd_type, constraints,
                 weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)
        self.perm = copy.deepcopy(perm)
        self.prev_perm = copy.deepcopy(prev_perm)

        self.using_T_size = self.T_size

    def match(self, match_info):
        return super().match(match_info) and len(self.prev_perm) == len(
            self.perm) and (len(self.create_blend_perm_list()) != 0)

    def create_blend_perm_list(self):
        err_assert(
            len(self.prev_perm) == len(self.perm), "Permutations don't match")

        empty = []

        lperm = self.perm
        blend_mask = int(-1)
        if self.T_size < 4:
            blend_mask = blend_mask_lt_T(self.prev_perm, self.T_size, 4)
            if blend_mask == int(-1):
                return empty
            if can_shrink(self.T_size, 4, self.perm) is False:
                return empty
            lperm = shrink_perm(self.T_size, 4, self.perm)
            self.using_T_size = 4

        else:
            blend_mask = blend_mask_ge_T(self.prev_perm, self.T_size,
                                         self.T_size)

        blend_plist = []
        for i in range(0, len(lperm)):
            if (blend_mask & (int(1) << lperm[i])) != int(0):
                blend_plist.append(lperm[i] | len(lperm))
            else:
                blend_plist.append(lperm[i])
        return blend_plist

    def generate_instruction(self):
        instruction = "{}_permutex2var_epi{}([V1], {}_set_epi{}([PERM_LIST]), [V2])"
        set_epi_postfix = str(8 * self.using_T_size)
        if self.simd_type.sizeof() != 64 and self.T_size == 8:
            set_epi_postfix += "x"
        return instruction.format(self.simd_type.prefix(),
                                  8 * self.using_T_size,
                                  self.simd_type.prefix(),
                                  set_epi_postfix).replace(
                                      "[PERM_LIST]",
                                      arr_to_csv(self.create_blend_perm_list(),
                                                 False))


class SIMD_Permute():
    def __init__(self, prev_perm, perm):
        self.instructions = [
            ###################################################################
            SIMD_Alignr(1, perm, prev_perm, SIMD_m64(), ["SSE3"], 0),
            SIMD_Shuffle_m64(1, perm, SIMD_m64(), ["MMX", "SSSE3"], 1),
            SIMD_Permutex_Generate(
                "_mm_shuffle_pi8([V1], _mm_set_pi8([PERM_LIST]))", 1, perm,
                SIMD_m64(), ["MMX", "SSSE3"], 2),
            SIMD_Alignr(2, perm, prev_perm, SIMD_m64(), ["SSE3"], 0),
            SIMD_Shuffle_m64(2, perm, SIMD_m64(), ["MMX", "SSSE3"], 1),
            SIMD_Permutex_Generate(
                "_mm_shuffle_pi8([V1], _mm_set_pi8([PERM_LIST]))", 2, perm,
                SIMD_m64(), ["MMX", "SSSE3"], 2),
            ###################################################################
            # __m128i epi8 ordering
            SIMD_Shuffle2_As_PS(1, perm, prev_perm, SIMD_m128(), ["AVX"], 0),
            SIMD_Shuffle_As_Epi32(1, perm, SIMD_m128(), ["SSE2"], 1),
            SIMD_Alignr(1, perm, prev_perm, SIMD_m128(), ["SSE3"], 0),
            SIMD_Shuffle_As_Epi16(1, perm, SIMD_m128(), ["SSE2"], 1),
            SIMD_Shuffle_As_Epi8(1, perm, SIMD_m128(), ["SSSE3"],
                                 choose_if(Optimization.UOP, 2, 3)),
            SIMD_rotate(1, perm, SIMD_m128(), ["AVX512f", "AVX512vl"], 1),
            SIMD_Shuffle_Rotate(1, perm, SIMD_m128(), ["AVX512f", "AVX512vl"],
                                choose_if(Optimization.SPACE, 2, 3)),
            # __m128i epi16 ordering
            SIMD_Shuffle2_As_PS(2, perm, prev_perm, SIMD_m128(), ["AVX"], 0),
            SIMD_Shuffle_As_Epi32(2, perm, SIMD_m128(), ["SSE2"], 1),
            SIMD_Alignr(2, perm, prev_perm, SIMD_m128(), ["SSE3"], 0),
            SIMD_Shuffle_As_Epi16(2, perm, SIMD_m128(), ["SSE2"], 1),
            SIMD_Shuffle_As_Epi8(2, perm, SIMD_m128(), ["SSSE3"],
                                 choose_if(Optimization.UOP, 2, 3)),
            SIMD_rotate(2, perm, SIMD_m128(), ["AVX512f", "AVX512vl"], 1),
            SIMD_Shuffle_Rotate(2, perm, SIMD_m128(), ["AVX512f", "AVX512vl"],
                                choose_if(Optimization.SPACE, 2, 3)),
            # __m128i epi32 ordering
            SIMD_Shuffle2_As_PS(4, perm, prev_perm, SIMD_m128(), ["AVX"], 0),
            SIMD_Shuffle_As_Epi32(4, perm, SIMD_m128(), ["SSE2"], 1),
            SIMD_Alignr(4, perm, prev_perm, SIMD_m128(), ["SSE3"], 0),
            SIMD_rotate(4, perm, SIMD_m128(), ["AVX512f", "AVX512vl"], 1),
            # __m128i epi64 ordering
            SIMD_Shuffle2_As_PS(8, perm, prev_perm, SIMD_m256(), ["AVX"], 0),
            SIMD_Shuffle_As_Epi32(8, perm, SIMD_m128(), ["SSE2"], 1),
            SIMD_Alignr(8, perm, prev_perm, SIMD_m128(), ["SSE3"], 0),
            ###################################################################
            # __m256i epi8 ordering
            SIMD_Shuffle2_As_PS(1, perm, prev_perm, SIMD_m128(), ["AVX"], 0),
            SIMD_Shuffle_As_Epi32(1, perm, SIMD_m256(), ["AVX2"], 1),
            SIMD_Alignr(1, perm, prev_perm, SIMD_m256(), ["AVX2"], 0),
            SIMD_Shuffle_As_Epi16(1, perm, SIMD_m256(), ["AVX2"], 1),
            SIMD_Shuffle_As_Epi8(1, perm, SIMD_m256(), ["AVX2"],
                                 choose_if(Optimization.UOP, 2, 3)),
            SIMD_Shuffle_Rotate(1, perm, SIMD_m256(), ["AVX512f", "AVX512vl"],
                                choose_if(Optimization.SPACE, 2, 3)),
            SIMD_rotate(1, perm, SIMD_m256(), ["AVX512f", "AVX512vl"], 1),
            SIMD_Shuffle2_Blend(1, prev_perm, perm, SIMD_m256(), ["AVX2"], 2),
            SIMD_Permutex2_Blend(1, prev_perm, perm, SIMD_m256(),
                                 ["AVX512vl", "AVX512f"],
                                 choose_if(Optimization.UOP, 3, 4)),
            SIMD_Shuffle_As_Epi64(1, perm, SIMD_m256(), ["AVX2"],
                                  choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Permute_Move_Lanes_Shuffle(
                1, prev_perm, perm, SIMD_m256(), ["AVX2", "AVX"],
                choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Permutex_Generate(
                "_mm256_permutexvar_epi8(_mm256_set_epi8([PERM_LIST]), [V1])",
                1, perm, SIMD_m256(), ["AVX512vbmi", "AVX512vl", "AVX"],
                choose_if(Optimization.UOP, 4, 5)),
            # this is a super poorly optimized case. irrelivant of what we select with Move_Lanes
            SIMD_Permutex_Fallback(1, perm, SIMD_m256(), ["AVX2", "AVX"], 50),

            # __m256i epi16 ordering
            SIMD_Shuffle2_As_PS(2, perm, prev_perm, SIMD_m256(), ["AVX"], 0),
            SIMD_Shuffle_As_Epi32(2, perm, SIMD_m256(), ["AVX2"], 1),
            SIMD_Alignr(2, perm, prev_perm, SIMD_m256(), ["AVX2"], 0),
            SIMD_Shuffle_As_Epi16(2, perm, SIMD_m256(), ["AVX2"], 1),
            SIMD_Shuffle_As_Epi8(2, perm, SIMD_m256(), ["AVX2"],
                                 choose_if(Optimization.UOP, 2, 3)),
            SIMD_Shuffle_Rotate(2, perm, SIMD_m256(), ["AVX512f", "AVX512vl"],
                                choose_if(Optimization.SPACE, 2, 3)),
            SIMD_rotate(2, perm, SIMD_m256(), ["AVX512f", "AVX512vl"], 1),
            SIMD_Shuffle2_Blend(2, prev_perm, perm, SIMD_m256(), ["AVX2"], 2),
            SIMD_Permutex2_Blend(2, prev_perm, perm, SIMD_m256(),
                                 ["AVX512vl", "AVX512f"],
                                 choose_if(Optimization.UOP, 3, 4)),
            SIMD_Shuffle_As_Epi64(2, perm, SIMD_m256(), ["AVX2"],
                                  choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Permute_Move_Lanes_Shuffle(
                2, prev_perm, perm, SIMD_m256(), ["AVX2", "AVX"],
                choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Permutex_Generate(
                "_mm256_permutexvar_epi16(_mm256_set_epi16([PERM_LIST]), [V1])",
                2, perm, SIMD_m256(), ["AVX512bw", "AVX512vl", "AVX"],
                choose_if(Optimization.UOP, 4, 5)),

            # this is a super poorly optimized case. irrelivant of what we select with Move_Lanes
            SIMD_Permutex_Fallback(2, perm, SIMD_m256(), ["AVX2", "AVX"], 50),

            # __m256i epi32 ordering
            SIMD_Shuffle2_As_PS(4, perm, prev_perm, SIMD_m256(), ["AVX"], 0),
            SIMD_Shuffle_As_Epi32(4, perm, SIMD_m256(), ["AVX2"], 1),
            SIMD_Alignr(4, perm, prev_perm, SIMD_m256(), ["AVX2"], 0),
            SIMD_Shuffle_As_Epi8(4, perm, SIMD_m256(), ["AVX2"],
                                 choose_if(Optimization.UOP, 2, 3)),
            SIMD_Shuffle_Rotate(4, perm, SIMD_m256(), ["AVX512f", "AVX512vl"],
                                choose_if(Optimization.SPACE, 2, 3)),
            SIMD_rotate(4, perm, SIMD_m256(), ["AVX512f", "AVX512vl"], 1),
            SIMD_Shuffle2_Blend(4, prev_perm, perm, SIMD_m256(), ["AVX2"], 2),
            SIMD_Permutex2_Blend(4, prev_perm, perm, SIMD_m256(),
                                 ["AVX512vl", "AVX512f"],
                                 choose_if(Optimization.UOP, 3, 4)),
            SIMD_Shuffle_As_Epi64(4, perm, SIMD_m256(), ["AVX2"],
                                  choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Permute_Move_Lanes_Shuffle(
                4, prev_perm, perm, SIMD_m256(), ["AVX2", "AVX"],
                choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Permutex_Generate(
                "_mm256_permutevar8x32_epi32([V1], _mm256_set_epi32([PERM_LIST]))",
                4, perm, SIMD_m256(), ["AVX2", "AVX"],
                choose_if(Optimization.UOP, 4, 5)),

            # __m256i epi64 ordering
            SIMD_Shuffle2_As_PS(8, perm, prev_perm, SIMD_m256(), ["AVX"], 0),
            SIMD_Shuffle_As_Epi32(8, perm, SIMD_m256(), ["AVX2"], 1),
            SIMD_Alignr(8, perm, prev_perm, SIMD_m256(), ["AVX2"], 0),
            SIMD_Shuffle2_Blend(8, prev_perm, perm, SIMD_m256(), ["AVX2"], 2),
            SIMD_Permutex2_Blend(8, prev_perm, perm, SIMD_m256(),
                                 ["AVX512vl", "AVX512f"],
                                 choose_if(Optimization.UOP, 3, 4)),
            SIMD_Shuffle_As_Epi64(8, perm, SIMD_m256(), ["AVX2"],
                                  choose_if(Optimization.SPACE, 3, 4)),

            ###################################################################
            # __m512i epi8 ordering
            SIMD_Shuffle2_As_PS(1, perm, prev_perm, SIMD_m512(), ["AVX512f"], 0),
            SIMD_Shuffle_As_Epi32(1, perm, SIMD_m512(), ["AVX512f"], 1),
            SIMD_Alignr(1, perm, prev_perm, SIMD_m512(), ["AVX512bw"], 0),
            SIMD_Shuffle_As_Epi16(1, perm, SIMD_m512(), ["AVX512bw"], 1),
            SIMD_Shuffle_As_Epi8(1, perm, SIMD_m512(), ["AVX512bw"],
                                 choose_if(Optimization.UOP, 2, 3)),
            SIMD_Shuffle_Rotate(1, perm, SIMD_m512(), ["AVX512f", "AVX512vl"],
                                choose_if(Optimization.SPACE, 2, 3)),
            SIMD_rotate(1, perm, SIMD_m512(), ["AVX512f", "AVX512vl"], 1),
            SIMD_Shuffle2_Blend(1, prev_perm, perm, SIMD_m512(),
                                ["AVX512vl", "AVX512f"], 2),
            SIMD_Permutex2_Blend(1, prev_perm, perm, SIMD_m512(),
                                 ["AVX512vl", "AVX512f"],
                                 choose_if(Optimization.UOP, 3, 4)),
            SIMD_Shuffle_As_Epi64(1, perm, SIMD_m512(), ["AVX512f"],
                                  choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Shuffle_As_si128(1, perm, SIMD_m512(), ["AVX512f"],
                                  choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Permute_Move_Lanes_Shuffle(
                1, prev_perm, perm, SIMD_m512(), ["AVX512f"],
                choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Permutex_Generate(
                "_mm512_permutexvar_epi8(_mm512_set_epi8([PERM_LIST]), [V1])",
                1, perm, SIMD_m512(), ["AVX512vbmi", "AVX512f"],
                choose_if(Optimization.UOP, 4, 5)),
            # __m512i epi16 ordering
            SIMD_Shuffle2_As_PS(2, perm, prev_perm, SIMD_m512(), ["AVX512f"], 0),
            SIMD_Shuffle_As_Epi32(2, perm, SIMD_m512(), ["AVX512f"], 1),
            SIMD_Alignr(2, perm, prev_perm, SIMD_m512(), ["AVX512bw"], 0),
            SIMD_Shuffle_As_Epi16(2, perm, SIMD_m512(), ["AVX512bw"], 1),
            SIMD_Shuffle_As_Epi8(2, perm, SIMD_m512(), ["AVX512bw"],
                                 choose_if(Optimization.UOP, 2, 3)),
            SIMD_Shuffle_Rotate(1, perm, SIMD_m512(), ["AVX512f", "AVX512vl"],
                                choose_if(Optimization.SPACE, 2, 3)),
            SIMD_rotate(2, perm, SIMD_m512(), ["AVX512f", "AVX512vl"], 1),
            SIMD_Shuffle2_Blend(2, prev_perm, perm, SIMD_m512(),
                                ["AVX512vl", "AVX512f"], 2),
            SIMD_Permutex2_Blend(2, prev_perm, perm, SIMD_m512(),
                                 ["AVX512vl", "AVX512f"],
                                 choose_if(Optimization.UOP, 3, 4)),
            SIMD_Shuffle_As_Epi64(2, perm, SIMD_m512(), ["AVX512f"],
                                  choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Shuffle_As_si128(2, perm, SIMD_m512(), ["AVX512f"],
                                  choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Permute_Move_Lanes_Shuffle(
                2, prev_perm, perm, SIMD_m512(), ["AVX512f"],
                choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Permutex_Generate(
                "_mm512_permutexvar_epi16(_mm512_set_epi16([PERM_LIST]), [V1])",
                2, perm, SIMD_m512(), ["AVX512bw", "AVX512f"],
                choose_if(Optimization.UOP, 4, 5)),
            # __m512i epi32 ordering
            SIMD_Shuffle2_As_PS(4, perm, prev_perm, SIMD_m512(), ["AVX512f"], 0),
            SIMD_Shuffle_As_Epi32(4, perm, SIMD_m512(), ["AVX512f"], 1),
            SIMD_Alignr(4, perm, prev_perm, SIMD_m512(), ["AVX512bw"], 0),
            SIMD_Shuffle_As_Epi8(4, perm, SIMD_m512(), ["AVX512bw"],
                                 choose_if(Optimization.UOP, 2, 3)),
            SIMD_Shuffle_Rotate(4, perm, SIMD_m512(), ["AVX512f", "AVX512vl"],
                                choose_if(Optimization.SPACE, 2, 3)),
            SIMD_rotate(4, perm, SIMD_m512(), ["AVX512f", "AVX512vl"], 1),
            SIMD_Shuffle2_Blend(4, prev_perm, perm, SIMD_m512(),
                                ["AVX512vl", "AVX512f"], 2),
            SIMD_Permutex2_Blend(4, prev_perm, perm, SIMD_m512(),
                                 ["AVX512vl", "AVX512f"],
                                 choose_if(Optimization.UOP, 3, 4)),
            SIMD_Shuffle_As_Epi64(4, perm, SIMD_m512(), ["AVX512f"],
                                  choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Shuffle_As_si128(4, perm, SIMD_m512(), ["AVX512f"],
                                  choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Permute_Move_Lanes_Shuffle(
                4, prev_perm, perm, SIMD_m512(), ["AVX512f"],
                choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Permutex_Generate(
                "_mm512_permutexvar_epi32(_mm512_set_epi32([PERM_LIST]), [V1])",
                4, perm, SIMD_m512(), ["AVX512f"],
                choose_if(Optimization.UOP, 4, 5)),

            # __m512i epi64 ordering
            SIMD_Shuffle2_As_PS(8, perm, prev_perm, SIMD_m512(), ["AVX512f"], 0),
            SIMD_Shuffle_As_Epi32(8, perm, SIMD_m512(), ["AVX512f"], 1),
            SIMD_Alignr(8, perm, prev_perm, SIMD_m512(), ["AVX512bw"], 0),
            SIMD_Shuffle_As_Epi8(8, perm, SIMD_m512(), ["AVX512bw"],
                                 choose_if(Optimization.UOP, 2, 3)),
            SIMD_Shuffle2_Blend(8, prev_perm, perm, SIMD_m512(),
                                ["AVX512vl", "AVX512f"], 2),
            SIMD_Permutex2_Blend(8, prev_perm, perm, SIMD_m512(),
                                 ["AVX512vl", "AVX512f"],
                                 choose_if(Optimization.UOP, 3, 4)),
            SIMD_Shuffle_As_Epi64(8, perm, SIMD_m512(), ["AVX512f"],
                                  choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Shuffle_As_si128(8, perm, SIMD_m512(), ["AVX512f"],
                                  choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Permute_Move_Lanes_Shuffle(
                8, prev_perm, perm, SIMD_m512(), ["AVX512f"],
                choose_if(Optimization.SPACE, 3, 4)),
            SIMD_Permutex_Generate(
                "_mm512_permutexvar_epi64(_mm512_set_epi64([PERM_LIST]), [V1])",
                8, perm, SIMD_m512(), ["AVX512f"],
                choose_if(Optimization.UOP, 4, 5))
        ]


######################################################################
# Load & Store Shared Helper(s)
def build_mask_bool_vec_epi32(alignment_incr, raw_N, sort_type, simd_type):
    sort_bytes = raw_N * sort_type.sizeof()
    sort_bytes += alignment_incr
    N_sets = int(sort_bytes / 4)
    N_loads = int(simd_type.sizeof() / 4)

    set_arr = []
    for i in range(0, N_loads):
        if i < N_sets:
            set_arr.append(int(1) << (31))
        else:
            set_arr.append(0)

    set_arr_ret = []
    for i in range(0, len(set_arr)):
        set_arr_ret.append(set_arr[len(set_arr) - (i + 1)])

    return set_arr_ret


def build_mask_bool_vec_epi8(raw_N, sort_type, simd_type):
    N_loads = simd_type.sizeof()
    N_sets = raw_N * sort_type.sizeof()

    set_arr = []
    for i in range(0, N_loads):
        if i < N_sets:
            set_arr.append(int(1) << (7))
        else:
            set_arr.append(0)

    set_arr_ret = []
    for i in range(0, len(set_arr)):
        set_arr_ret.append(set_arr[len(set_arr) - (i + 1)])

    return set_arr_ret


######################################################################
# Load


# A few cases. First aligned vs unaligned. This doesn't make a big
# deal anymore but why not. Second is incomplete loads i.e sorting
# int32s with N = 12 requires an __m256i register but an array of 12
# int32s isnt guranteed to have enough memory to fully load a __m256i
# register so some logic to accomidate that is necessary
######################################################################
def get_max_set_vec(sort_type, simd_type):
    epi_postfix = str(sort_type.sizeof_bits())
    if simd_type.sizeof() != 64 and sort_type.sizeof() == 8:
        epi_postfix += "x"
    return "{}_set1_epi{}([MAX])".format(simd_type.prefix(),
                                         epi_postfix).replace(
                                             "[MAX]", sort_type.max_value())


def generate_blend_instruction(raw_N, sort_type, simd_type, tmp1, tmp2):
    sort_bytes = raw_N * sort_type.sizeof()
    if sort_bytes % 4 == 0:
        return "{}_blend_epi32({}, {}, [BLEND_MASK])".format(
            simd_type.prefix(), tmp1,
            tmp2).replace("[BLEND_MASK]",
                          str(hex((int(1) << int(sort_bytes / 4)) - 1)))
    else:
        blend_vec = build_mask_bool_vec_epi8(raw_N, sort_type, simd_type)
        list_blend_vec = arr_to_csv(blend_vec, False)
        return "{}_blendv_epi8({}, {}, {}_set_epi8([BLEND_VEC]))".format(
            simd_type.prefix(), tmp1, tmp2,
            simd_type.prefix()).replace("[BLEND_VEC]", list_blend_vec)


class SIMD_Full_Load_Base(SIMD_Instruction):
    def __init__(self, iname, aligned, simd_type, constraints, weight):
        # T_size doesn't matter, only register size
        super().__init__(iname, Sign.NOT_SIGNED, 0, simd_type, constraints,
                         weight)

        # Booleans
        self.aligned = aligned

    def match(self, match_info):
        return self.has_support() and self.match_simd_type(
            match_info.simd_type) and (
                (match_info.aligned is self.aligned) or
                (self.aligned is False)) and (match_info.full is True)


class SIMD_Full_Load(SIMD_Instruction):
    def __init__(self, iname, fill, aligned, simd_type, constraints, weight):
        # T_size doesn't matter, only register size
        super().__init__(iname, Sign.NOT_SIGNED, 0, simd_type, constraints,
                         weight)

        # Booleans
        self.aligned = aligned
        self.fill = fill
        self.SIMD_full_load_base = SIMD_Full_Load_Base(iname, aligned,
                                                       simd_type, constraints,
                                                       weight)

    def match(self, match_info):
        return (self.fill is
                False) and (self.SIMD_full_load_base.match(match_info) is True)

    def generate_instruction(self):
        return self.SIMD_full_load_base.generate_instruction()


class SIMD_Full_Load_Fill(SIMD_Instruction):
    def __init__(self, iname, sort_type, fill, aligned, raw_N, simd_type,
                 constraints, weight):
        # T_size doesn't matter, only register size
        super().__init__(iname, Sign.NOT_SIGNED, 0, simd_type, constraints,
                         weight)

        self.raw_N = raw_N
        self.sort_type = sort_type
        # Booleans
        self.aligned = aligned
        self.fill = fill
        self.SIMD_full_load_base = SIMD_Full_Load_Base(iname, aligned,
                                                       simd_type, constraints,
                                                       weight)

    def match(self, match_info):
        return (self.fill is
                True) and (self.SIMD_full_load_base.match(match_info) is True)

    def generate_instruction(self):
        err_assert(self.simd_type.sizeof() < 64, "should use mask_load here")
        instruction = "{} [TMP0] = [MAX_SET];".format(
            self.simd_type.to_string()).replace(
                "[MAX_SET]", get_max_set_vec(self.sort_type, self.simd_type))
        instruction += "\n"
        instruction += "{} [TMP1] = [BASE_LOAD];".format(
            self.simd_type.to_string()).replace(
                "[BASE_LOAD]", self.SIMD_full_load_base.generate_instruction())
        instruction += "\n"
        return instruction + generate_blend_instruction(
            self.raw_N, self.sort_type, self.simd_type, "[TMP0]", "[TMP1]")


class SIMD_Partial_Load_m64(SIMD_Instruction):
    def __init__(self, sort_type, raw_N, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, sort_type.sizeof(), simd_type,
                         constraints, weight)
        self.raw_N = raw_N
        self.sort_type = sort_type

    def generate_instruction(self):
        err_assert(self.simd_type.sizeof() == 8,
                   "using __m64 load for non __m64 register")

        instruction = "{} [TMP0] = {}_set1_pi{}([MAX]);".format(
            self.simd_type.to_string(), self.simd_type.prefix(),
            self.sort_type.sizeof_bits()).replace("[MAX]",
                                                  self.sort_type.max_value())
        instruction += "\n"
        instruction += "__builtin_memcpy(&[TMP0], [ARR], [SORT_BYTES]);".replace(
            "[SORT_BYTES]", str(self.raw_N * self.sort_type.sizeof()))
        instruction += "\n"
        instruction += "[TMP0]"
        return instruction


class SIMD_Mask_Load_ASM_Epi8(SIMD_Instruction):
    def __init__(self, sort_type, T_size, fill, raw_N, simd_type, constraints,
                 weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)
        self.raw_N = raw_N
        self.sort_type = sort_type
        self.fill = fill

    def match(self, match_info):
        return self.has_support() and self.match_simd_type(
            match_info.simd_type) and (EXTRA_MEMORY is True)

    def generate_instruction(self):
        err_assert(
            int((self.raw_N * self.sort_type.sizeof()) % 4) != 0,
            "should be using epi32 blend")
        err_assert(INT_ALIGNED is False or self.fill is True,
                   "should be using epi32 blend")

        err_assert(
            self.simd_type.sizeof() <= 32 and self.simd_type.sizeof() >= 8,
            "this only applies to __m128i and __m256i")

        instruction = "{} [TMP0]".format(self.simd_type.to_string())

        if self.fill is True:
            instruction += " = [MAX_SET]".replace(
                "[MAX_SET]", get_max_set_vec(self.sort_type, self.simd_type))

        instruction += ";"
        instruction += "\n"

        blend_vec = build_mask_bool_vec_epi8(self.raw_N, self.sort_type,
                                             self.simd_type)
        list_blend_vec = arr_to_csv(blend_vec, False)

        instruction += "{} [TMP1] = {}_set_epi8([BLEND_VEC]);".format(
            self.simd_type.to_string(),
            self.simd_type.prefix()).replace("[BLEND_VEC]", list_blend_vec)
        instruction += "\n"

        instruction += "asm volatile(\"vpblendvb %[load_mask], (%[arr]), %[fill_v], %[fill_v]\\n\""
        instruction += "\n"
        instruction += ": [ fill_v ] \"+x\" ([TMP0])"
        instruction += "\n"
        instruction += ": [ arr ] \"r\" ([ARR]), [ load_mask ] \"x\" ([TMP1])"
        instruction += "\n"
        instruction += ":);"
        instruction += "\n"
        instruction += "[TMP0]"
        return instruction


class SIMD_Mask_Load_ASM_Epi32(SIMD_Instruction):
    def __init__(self, sort_type, T_size, fill, raw_N, simd_type, constraints,
                 weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)
        self.raw_N = raw_N
        self.sort_type = sort_type
        self.fill = fill

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and (self.can_do_epi32() is True)

    def can_do_epi32(self):
        sort_bytes = self.raw_N * self.sort_type.sizeof()
        return ((sort_bytes % 4) == 0 or
                (self.fill is False
                 and INT_ALIGNED is True)) and EXTRA_MEMORY is True

    def generate_instruction(self):
        err_assert(
            self.simd_type.sizeof() <= 32 and self.simd_type.sizeof() >= 8,
            "this only applies to __m128i and __m256i")
        epi32_n = 0

        if self.sort_type.sizeof() < 4:
            tdiv = int(4 / self.sort_type.sizeof())
            epi32_n = (int(1) << int((self.raw_N + (tdiv - 1)) / tdiv)) - 1
        else:
            epi32_n = (int(1) << int(
                self.raw_N * int(self.sort_type.sizeof() / 4))) - 1

        instruction = "{} [TMP0]".format(self.simd_type.to_string())

        if self.fill is True:
            instruction += " = [MAX_SET]".replace(
                "[MAX_SET]", get_max_set_vec(self.sort_type, self.simd_type))

        instruction += ";"
        instruction += "\n"

        instruction += "asm volatile(\"vpblendd %[load_mask], (%[arr]), %[fill_v], %[fill_v]\\n\""
        instruction += "\n"
        instruction += ": [ fill_v ] \"+x\" ([TMP0])"
        instruction += "\n"
        instruction += ": [ arr ] \"r\" ([ARR]), [ load_mask ] \"i\" ([LOAD_MASK])".replace(
            "[LOAD_MASK]", str(hex(epi32_n)))
        instruction += "\n"
        instruction += ":);"
        instruction += "\n"
        instruction += "[TMP0]"
        return instruction


class SIMD_Mask_Load_Epi32(SIMD_Instruction):
    def __init__(self, sort_type, T_size, fill, aligned, raw_N, simd_type,
                 constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)
        self.raw_N = raw_N
        self.sort_type = sort_type
        self.fill = fill

        # Booleans
        self.aligned = aligned

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and (
                    (match_info.aligned is self.aligned) or
                    (self.aligned is False)) and (self.can_do_epi32() is True)

    def can_do_epi32(self):
        sort_bytes = self.raw_N * self.sort_type.sizeof()
        return (sort_bytes % 4) == 0 or (self.fill is False
                                         and INT_ALIGNED is True)

    def generate_instruction(self):
        epi32_n = 0

        if self.sort_type.sizeof() < 4:
            tdiv = int(4 / self.sort_type.sizeof())
            epi32_n = (int(1) << int((self.raw_N + (tdiv - 1)) / tdiv)) - 1
        else:
            epi32_n = (int(1) << int(
                self.raw_N * int(self.sort_type.sizeof() / 4))) - 1

        instruction = "{} [TMP0]".format(self.simd_type.to_string())
        aligned_postfix = "u"
        if self.aligned is True:
            aligned_postfix = ""

        if self.fill is True:
            instruction += " = [MAX_SET]".replace(
                "[MAX_SET]", get_max_set_vec(self.sort_type, self.simd_type))

        instruction += ";"
        instruction += "\n"

        header.stdint = True
        return instruction + "{}_mask_load{}_epi32([TMP0], [LOAD_MASK], (int32_t * const)[ARR])".format(
            self.simd_type.prefix(), aligned_postfix).replace(
                "[LOAD_MASK]", str(hex(epi32_n)))


class SIMD_Mask_Load(SIMD_Instruction):
    def __init__(self, iname, sort_type, T_size, fill, aligned, raw_N,
                 simd_type, constraints, weight):
        super().__init__(iname, Sign.NOT_SIGNED, T_size, simd_type,
                         constraints, weight)
        self.raw_N = raw_N
        self.sort_type = sort_type
        self.fill = fill

        # Booleans
        self.aligned = aligned

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and ((match_info.aligned is self.aligned)
                                           or (self.aligned is False))

    def generate_instruction(self):
        instruction = "{} [TMP0]".format(self.simd_type.to_string())

        if self.fill is True:
            instruction += " = [MAX_SET]".replace(
                "[MAX_SET]", get_max_set_vec(self.sort_type, self.simd_type))

        instruction += ";"
        instruction += "\n"

        return instruction + self.iname.replace(
            "[FILL_TMP]", "[TMP0]").replace(
                "[LOAD_MASK]", str(hex((int(1) << self.raw_N) - 1)))


class SIMD_Mask_Load_Fallback_As_Epi32(SIMD_Instruction):
    def __init__(self, sort_type, fill, raw_N, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, sort_type.sizeof(), simd_type,
                         constraints, weight)

        self.raw_N = raw_N
        self.sort_type = sort_type
        self.extra_insert = False
        self.fill = fill

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and (self.fill is False)

    def build_mask_bool_vec_wrapper(self):
        sort_bytes = self.raw_N * self.sort_type.sizeof()
        if INT_ALIGNED is False:
            if (sort_bytes % 4) != 0:
                self.extra_insert = True
                self.weight = 100
            return build_mask_bool_vec_epi32(0, self.raw_N, self.sort_type,
                                             self.simd_type)
        else:
            return build_mask_bool_vec_epi32(3, self.raw_N, self.sort_type,
                                             self.simd_type)

    def generate_tmp_set(self, tmp_name):
        ub = self.raw_N
        lb = self.raw_N - (self.raw_N % int(4 / self.sort_type.sizeof()))
        dif = ub - lb
        instruction = "const uint32_t {} = ".format(tmp_name)

        if dif == 1:
            instruction += "(uint32_t)[ARR][{}]".format(lb)
        elif dif == 2:
            header.aliasing_int16 = True
            instruction += "((_aliasing_int16_t_ *)[ARR])[{}]".format(
                int(lb / 2))
        elif dif == 3:
            header.aliasing_int16 = True
            instruction += "(((uint32_t)((_aliasing_int16_t_ *)[ARR])[{}]) & 0xffff) | ((((uint32_t)[ARR][{}]) & 0xff) << 16)".format(
                int(lb / 2), lb + 2)

        return instruction

    def generate_instruction(self):
        header.stdint = True

        load_bool_vec = self.build_mask_bool_vec_wrapper()
        list_load_bool_vec = arr_to_csv(load_bool_vec, True)
        instruction = "{}_maskload_epi32((int32_t * const)[ARR], {}_set_epi32([LOAD_BOOLS]))".format(
            self.simd_type.prefix(),
            self.simd_type.prefix()).replace("[LOAD_BOOLS]",
                                             list_load_bool_vec)
        if self.extra_insert is True:
            err_assert(INT_ALIGNED is False,
                       "using extra_insert unnecissarily")
            instruction = "{} [TMP0] = ".format(
                self.simd_type.to_string()) + instruction + ";"
            instruction += "\n"
            instruction += self.generate_tmp_set("[TMP1]") + ";"
            instruction += "\n"
            instruction += "{}_insert_epi32([TMP0], [TMP1], [IDX])".format(
                self.simd_type.prefix()).replace(
                    "[IDX]",
                    str(int(self.raw_N / int(4 / self.sort_type.sizeof()))))
        return instruction


class SIMD_Mask_Load_Fallback_As_Epi32_Fill(SIMD_Instruction):
    def __init__(self, sort_type, fill, raw_N, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, sort_type.sizeof(), simd_type,
                         constraints, weight)

        self.raw_N = raw_N
        self.sort_type = sort_type
        self.extra_insert = False
        self.fill = fill

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and (self.fill is True)

    def build_fill_vec(self, tmp_name):
        T_size = self.sort_type.sizeof()
        err_assert(self.simd_type.sizeof() <= 32,
                   "building fallback unnecissarily")

        if self.extra_insert is False:
            return get_max_set_vec(self.sort_type, self.simd_type)

        sort_bytes = self.raw_N * T_size
        err_assert(sort_bytes % 4 != 0 and INT_ALIGNED is False,
                   "should be using std max vec")

        scale_to_int = int(4 / T_size)
        N_loads = int(self.simd_type.sizeof() / T_size)

        max_fill = []
        for i in range(0, N_loads, scale_to_int):
            val = int(0)
            for j in range(i, i + scale_to_int):
                val |= (int(self.sort_type.max_value_int()) <<
                        ((j - i) * self.sort_type.sizeof_bits()))
            max_fill.append(str(hex(val)))

        replacement_idx = (int((sort_bytes + 3) / 4))
        replacement_idx = int(self.simd_type.sizeof() / 4) - replacement_idx
        max_fill[replacement_idx] = tmp_name
        list_max_fill = arr_to_csv(max_fill, False)

        return "{}_set_epi32([MAX_FILL])".format(
            self.simd_type.prefix()).replace("[MAX_FILL]", list_max_fill)

    def build_mask_bool_vec_wrapper(self):
        sort_bytes = self.raw_N * self.sort_type.sizeof()
        if INT_ALIGNED is False:
            if (sort_bytes % 4) != 0:
                self.extra_insert = True
                self.weight = 100
            return build_mask_bool_vec_epi32(0, self.raw_N, self.sort_type,
                                             self.simd_type)
        else:
            return build_mask_bool_vec_epi32(3, self.raw_N, self.sort_type,
                                             self.simd_type)

    def generate_tmp_set(self, tmp_name):
        err_assert(self.sort_type.sizeof() < 4,
                   "generating tmp with int aligned load")
        ub = self.raw_N
        lb = self.raw_N - (self.raw_N % int(4 / self.sort_type.sizeof()))
        dif = ub - lb
        err_assert(dif > 0 and dif < 4, "no remainder for tmp generation")

        instruction = "const uint32_t {} = ".format(tmp_name)

        v = int(0)
        for i in range(dif, int(4 / self.sort_type.sizeof())):
            v |= self.sort_type.max_value_int() << (
                i * self.sort_type.sizeof_bits())

        instruction += "{} | ".format(str(hex(v)))

        if dif == 1:
            instruction += "((uint32_t)[ARR][{}] & {})".format(
                lb, str(hex((int(1) << (self.sort_type.sizeof_bits())) - 1)))
        elif dif == 2:
            header.aliasing_int16 = True
            instruction += "(((_aliasing_int16_t_ *)[ARR])[{}] & 0xffff)".format(
                int(lb / 2))
        elif dif == 3:
            header.aliasing_int16 = True
            instruction += "(((uint32_t)((_aliasing_int16_t_ *)[ARR])[{}]) & 0xffff) | ((((uint32_t)[ARR][{}]) & 0xff) << 16)".format(
                int(lb / 2), lb + 2)

        return instruction

    def generate_instruction(self):
        err_assert(self.simd_type.sizeof() <= 32,
                   "using fallback with AVX512 available")
        header.stdint = True

        load_bool_vec = self.build_mask_bool_vec_wrapper()
        list_load_bool_vec = arr_to_csv(load_bool_vec, True)

        instruction = "{} [TMP0] = {}_maskload_epi32((int32_t * const)[ARR], {}_set_epi32([LOAD_BOOLS]));".format(
            self.simd_type.to_string(), self.simd_type.prefix(),
            self.simd_type.prefix()).replace("[LOAD_BOOLS]",
                                             list_load_bool_vec)

        instruction += "\n"

        raw_N = self.raw_N
        fill_vec = self.build_fill_vec("[TMP2]")

        if self.extra_insert is True:
            instruction += self.generate_tmp_set("[TMP2]") + ";"
            instruction += "\n"

            rounded_sort_bytes = 4 * int(
                (self.raw_N * self.sort_type.sizeof()) / 4)
            rounded_N = int(rounded_sort_bytes / self.sort_type.sizeof())
            raw_N = rounded_N

        instruction += "{} [TMP1] = [FILL_VEC];".format(
            self.simd_type.to_string()).replace("[FILL_VEC]", fill_vec)
        instruction += "\n"

        instruction += generate_blend_instruction(raw_N, self.sort_type,
                                                  self.simd_type, "[TMP1]",
                                                  "[TMP0]")

        return instruction


class SIMD_Load():
    def __init__(self, raw_N, sort_type, scaled_sort_N):
        self.instructions = [
            ###################################################################
            SIMD_Full_Load("(*((_aliasing_m64_ *)[ARR]))", scaled_sort_N,
                           False, SIMD_m64(), ["MMX"], 0),
            SIMD_Partial_Load_m64(sort_type, raw_N, SIMD_m64(), ["MMX"], 1),
            ###################################################################
            # These are universal best instructions if applicable
            SIMD_Full_Load("_mm_load_si128((__m128i *)[ARR])", scaled_sort_N,
                           True, SIMD_m128(), ["SSE2"], 0),
            SIMD_Full_Load("_mm_loadu_si128((__m128i *)[ARR])", scaled_sort_N,
                           False, SIMD_m128(), ["SSE2"], 1),

            # For fill mask_load with AVX512 is prefered so putting between mask_load and universal fallback
            SIMD_Full_Load_Fill("_mm_load_si128((__m128i *)[ARR])",
                                sort_type, scaled_sort_N, True, raw_N,
                                SIMD_m128(), ["SSE2"], 10),
            SIMD_Full_Load_Fill("_mm_loadu_si128((__m128i *)[ARR])",
                                sort_type, scaled_sort_N, False, raw_N,
                                SIMD_m128(), ["SSE2"], 11),

            # This is universal fallback and always lowest priority
            SIMD_Mask_Load_Fallback_As_Epi32(sort_type, scaled_sort_N, raw_N,
                                             SIMD_m128(), ["AVX2", "SSE2"],
                                             15),
            SIMD_Mask_Load_Fallback_As_Epi32_Fill(sort_type, scaled_sort_N,
                                                  raw_N, SIMD_m128(),
                                                  ["AVX2", "SSE2"], 25),

            # epi8 __m128i ordering
            SIMD_Mask_Load_ASM_Epi32(sort_type, 1, scaled_sort_N, raw_N,
                                     SIMD_m128(), ["AVX2", "SSE2"], 2),
            SIMD_Mask_Load_Epi32(sort_type, 1, scaled_sort_N, True, raw_N,
                                 SIMD_m128(), ["AVX512vl", "AVX512f", "SSE2"],
                                 3),
            SIMD_Mask_Load_Epi32(sort_type, 1, scaled_sort_N, False, raw_N,
                                 SIMD_m128(), ["AVX512vl", "AVX512f", "SSE2"],
                                 4),
            SIMD_Mask_Load(
                "_mm_mask_loadu_epi8([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 1, scaled_sort_N, False, raw_N, SIMD_m128(),
                ["AVX512vl", "AVX512bw", "SSE2"], 5),
            SIMD_Mask_Load_ASM_Epi8(sort_type, 1, scaled_sort_N, raw_N,
                                    SIMD_m128(), ["SSE4.1", "SSE2"], 6),

            # epi16 __m128i ordering
            SIMD_Mask_Load_ASM_Epi32(sort_type, 2, scaled_sort_N, raw_N,
                                     SIMD_m128(), ["AVX2", "SSE2"], 2),
            SIMD_Mask_Load_Epi32(sort_type, 2, scaled_sort_N, True, raw_N,
                                 SIMD_m128(), ["AVX512vl", "AVX512f", "SSE2"],
                                 3),
            SIMD_Mask_Load_Epi32(sort_type, 2, scaled_sort_N, False, raw_N,
                                 SIMD_m128(), ["AVX512vl", "AVX512f", "SSE2"],
                                 4),
            SIMD_Mask_Load(
                "_mm_mask_loadu_epi16([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 2, scaled_sort_N, False, raw_N, SIMD_m128(),
                ["AVX512vl", "AVX512bw", "SSE2"], 5),
            SIMD_Mask_Load_ASM_Epi8(sort_type, 2, scaled_sort_N, raw_N,
                                    SIMD_m128(), ["SSE4.1", "SSE2"], 6),

            # epi32 __m128i ordering
            SIMD_Mask_Load_ASM_Epi32(sort_type, 4, scaled_sort_N, raw_N,
                                     SIMD_m128(), ["AVX2", "SSE2"], 2),
            SIMD_Mask_Load_Epi32(sort_type, 4, scaled_sort_N, True, raw_N,
                                 SIMD_m128(), ["AVX512vl", "AVX512f", "SSE2"],
                                 3),
            SIMD_Mask_Load_Epi32(sort_type, 4, scaled_sort_N, False, raw_N,
                                 SIMD_m128(), ["AVX512vl", "AVX512f", "SSE2"],
                                 4),
            SIMD_Mask_Load(
                "_mm_mask_load_epi32([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 4, scaled_sort_N, True, raw_N, SIMD_m128(),
                ["AVX512vl", "AVX512f", "SSE2"], 4),
            SIMD_Mask_Load(
                "_mm_mask_loadu_epi32([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 4, scaled_sort_N, False, raw_N, SIMD_m128(),
                ["AVX512vl", "AVX512f", "SSE2"], 5),

            # epi64 __m128i ordering
            SIMD_Mask_Load_ASM_Epi32(sort_type, 8, scaled_sort_N, raw_N,
                                     SIMD_m128(), ["AVX2", "SSE2"], 2),
            SIMD_Mask_Load_Epi32(sort_type, 8, scaled_sort_N, True, raw_N,
                                 SIMD_m128(), ["AVX512vl", "AVX512f", "SSE2"],
                                 3),
            SIMD_Mask_Load_Epi32(sort_type, 8, scaled_sort_N, False, raw_N,
                                 SIMD_m128(), ["AVX512vl", "AVX512f", "SSE2"],
                                 4),
            SIMD_Mask_Load(
                "_mm_mask_load_epi64([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 8, scaled_sort_N, True, raw_N, SIMD_m128(),
                ["AVX512vl", "AVX512f", "SSE2"], 5),
            SIMD_Mask_Load(
                "_mm_mask_loadu_epi64([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 8, scaled_sort_N, False, raw_N, SIMD_m128(),
                ["AVX512vl", "AVX512f", "SSE2"], 6),

            ###################################################################
            # These are universal best instructions if applicable
            SIMD_Full_Load("_mm256_load_si256((__m256i *)[ARR])",
                           scaled_sort_N, True, SIMD_m256(), ["AVX"], 0),
            SIMD_Full_Load("_mm256_loadu_si256((__m256i *)[ARR])",
                           scaled_sort_N, False, SIMD_m256(), ["AVX"], 1),

            # For fill mask_load with AVX512 is prefered so putting between mask_load and universal fallback
            SIMD_Full_Load_Fill("_mm256_load_si256((__m256i *)[ARR])",
                                sort_type, scaled_sort_N, True, raw_N,
                                SIMD_m256(), ["AVX", "AVX2"], 10),
            SIMD_Full_Load_Fill("_mm256_loadu_si256((__m256i *)[ARR])",
                                sort_type, scaled_sort_N, False, raw_N,
                                SIMD_m256(), ["AVX2", "SSE2"], 11),

            # This is universal fallback and always lowest priority
            SIMD_Mask_Load_Fallback_As_Epi32(sort_type, scaled_sort_N, raw_N,
                                             SIMD_m256(), ["AVX2"], 15),
            SIMD_Mask_Load_Fallback_As_Epi32_Fill(sort_type, scaled_sort_N,
                                                  raw_N, SIMD_m256(),
                                                  ["AVX2", "SSE2"], 25),

            # epi8 __m256i ordering
            SIMD_Mask_Load_ASM_Epi32(sort_type, 1, scaled_sort_N, raw_N,
                                     SIMD_m256(), ["AVX2", "SSE2"], 2),
            SIMD_Mask_Load_Epi32(sort_type, 1, scaled_sort_N, True, raw_N,
                                 SIMD_m256(), ["AVX512vl", "AVX512f", "SSE2"],
                                 3),
            SIMD_Mask_Load_Epi32(sort_type, 1, scaled_sort_N, False, raw_N,
                                 SIMD_m256(), ["AVX512vl", "AVX512f", "SSE2"],
                                 4),
            SIMD_Mask_Load(
                "_mm256_mask_loadu_epi8([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 1, scaled_sort_N, False, raw_N, SIMD_m256(),
                ["AVX512vl", "AVX512bw", "SSE2"], 5),
            SIMD_Mask_Load_ASM_Epi8(sort_type, 1, scaled_sort_N, raw_N,
                                    SIMD_m256(), ["AVX2", "SSE2"], 6),

            # epi16 __m256i ordering
            SIMD_Mask_Load_ASM_Epi32(sort_type, 2, scaled_sort_N, raw_N,
                                     SIMD_m256(), ["AVX2", "SSE2"], 2),
            SIMD_Mask_Load_Epi32(sort_type, 2, scaled_sort_N, True, raw_N,
                                 SIMD_m256(), ["AVX512vl", "AVX512f", "SSE2"],
                                 3),
            SIMD_Mask_Load_Epi32(sort_type, 2, scaled_sort_N, False, raw_N,
                                 SIMD_m256(), ["AVX512vl", "AVX512f", "SSE2"],
                                 4),
            SIMD_Mask_Load(
                "_mm256_mask_loadu_epi16([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 2, scaled_sort_N, False, raw_N, SIMD_m256(),
                ["AVX512vl", "AVX512bw", "SSE2"], 5),
            SIMD_Mask_Load_ASM_Epi8(sort_type, 2, scaled_sort_N, raw_N,
                                    SIMD_m256(), ["AVX2", "SSE2"], 6),

            # epi32 __m256i ordering
            SIMD_Mask_Load_ASM_Epi32(sort_type, 4, scaled_sort_N, raw_N,
                                     SIMD_m256(), ["AVX2", "SSE2"], 2),
            SIMD_Mask_Load_Epi32(sort_type, 4, scaled_sort_N, True, raw_N,
                                 SIMD_m256(), ["AVX512vl", "AVX512f", "SSE2"],
                                 3),
            SIMD_Mask_Load_Epi32(sort_type, 4, scaled_sort_N, False, raw_N,
                                 SIMD_m256(), ["AVX512vl", "AVX512f", "SSE2"],
                                 4),
            SIMD_Mask_Load(
                "_mm256_mask_load_epi32([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 4, scaled_sort_N, True, raw_N, SIMD_m256(),
                ["AVX512vl", "AVX512f", "SSE2"], 4),
            SIMD_Mask_Load(
                "_mm256_mask_loadu_epi32([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 4, scaled_sort_N, False, raw_N, SIMD_m256(),
                ["AVX512vl", "AVX512f", "SSE2"], 5),

            # epi64 __m256i ordering
            SIMD_Mask_Load_ASM_Epi32(sort_type, 8, scaled_sort_N, raw_N,
                                     SIMD_m256(), ["AVX2", "SSE2"], 2),
            SIMD_Mask_Load_Epi32(sort_type, 8, scaled_sort_N, True, raw_N,
                                 SIMD_m256(), ["AVX512vl", "AVX512f", "SSE2"],
                                 3),
            SIMD_Mask_Load_Epi32(sort_type, 8, scaled_sort_N, False, raw_N,
                                 SIMD_m256(), ["AVX512vl", "AVX512f", "SSE2"],
                                 4),
            SIMD_Mask_Load(
                "_mm256_mask_load_epi64([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 8, scaled_sort_N, True, raw_N, SIMD_m256(),
                ["AVX512vl", "AVX512f", "SSE2"], 5),
            SIMD_Mask_Load(
                "_mm256_mask_loadu_epi64([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 8, scaled_sort_N, False, raw_N, SIMD_m256(),
                ["AVX512vl", "AVX512f", "SSE2"], 6),

            ###################################################################
            # These are universal best instructions if applicable
            SIMD_Full_Load("_mm512_load_si512((__m512i *)[ARR])",
                           scaled_sort_N, True, SIMD_m512(), ["AVX512f"], 0),
            SIMD_Full_Load("_mm512_loadu_si512((__m512i *)[ARR])",
                           scaled_sort_N, False, SIMD_m512(), ["AVX512f"], 1),
            SIMD_Mask_Load(
                "_mm512_mask_loadu_epi8([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 1, scaled_sort_N, False, raw_N, SIMD_m512(),
                ["AVX512f", "AVX512bw", "AVX"], 2),
            SIMD_Mask_Load(
                "_mm512_mask_loadu_epi16([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 2, scaled_sort_N, False, raw_N, SIMD_m512(),
                ["AVX512f", "AVX512bw", "AVX"], 2),
            SIMD_Mask_Load(
                "_mm512_mask_load_epi32([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 4, scaled_sort_N, True, raw_N, SIMD_m512(),
                ["AVX512f"], 2),
            SIMD_Mask_Load(
                "_mm512_mask_loadu_epi32([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 4, scaled_sort_N, False, raw_N, SIMD_m512(),
                ["AVX512f"], 3),
            SIMD_Mask_Load(
                "_mm512_mask_load_epi64([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 8, scaled_sort_N, True, raw_N, SIMD_m512(),
                ["AVX512f"], 2),
            SIMD_Mask_Load(
                "_mm512_mask_loadu_epi64([FILL_TMP], [LOAD_MASK], [ARR])",
                sort_type, 8, scaled_sort_N, False, raw_N, SIMD_m512(),
                ["AVX512f"], 3),
            ###################################################################
        ]


######################################################################
# Store

# Store has similiar requirements and logic to load in that array size
# and vec don't necessarily align.
######################################################################


class SIMD_Full_Store(SIMD_Instruction):
    def __init__(self, iname, fill, aligned, simd_type, constraints, weight):
        # T_size doesn't matter, only register size
        super().__init__(iname, Sign.NOT_SIGNED, 0, simd_type, constraints,
                         weight)

        # Booleans
        self.aligned = aligned
        self.fill = fill

    def match(self, match_info):
        return self.has_support() and self.match_simd_type(
            match_info.simd_type) and (
                (match_info.aligned is self.aligned) or
                (self.aligned is False)) and (match_info.full is
                                              True) and (self.fill is False)


class SIMD_Mask_Store(SIMD_Instruction):
    def __init__(self, iname, T_size, aligned, raw_N, simd_type, constraints,
                 weight):
        super().__init__(iname, Sign.NOT_SIGNED, T_size, simd_type,
                         constraints, weight)
        self.raw_N = raw_N
        self.aligned = aligned

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and ((match_info.aligned is self.aligned)
                                           or (self.aligned is False))

    def generate_instruction(self):
        return self.iname.replace("[STORE_MASK]",
                                  str(hex((int(1) << self.raw_N) - 1)))


class SIMD_Partial_Store_m64(SIMD_Instruction):
    def __init__(self, T_size, raw_N, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)

        self.raw_N = raw_N

    def generate_instruction(self):
        err_assert(self.raw_N * self.T_size <= 8,
                   "Invalid Partial_Store_m64 sort_bytes")
        err_assert(self.simd_type.sizeof() == 8,
                   "Using __m64 partial store for non __m64 register")

        instruction = "__builtin_memcpy([ARR], &[V], [SIZE])".replace(
            "[SIZE]", str(self.raw_N * self.T_size))
        return instruction


class SIMD_Mask_Store_Fallback_As_Epi32(SIMD_Instruction):
    def __init__(self, fill, sort_type, raw_N, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, sort_type.sizeof(), simd_type,
                         constraints, weight)
        self.raw_N = raw_N
        self.sort_type = sort_type
        self.extra_insert = False
        self.fill = fill

    def build_store_bool_vec_wrapper(self):
        err_assert(self.simd_type.sizeof() > 8,
                   self.__class__.__name__ + " not applicable for __m64")

        sort_bytes = self.raw_N * self.sort_type.sizeof()
        if INT_ALIGNED is False or self.fill is True:
            self.weight = 75
            return build_mask_bool_vec_epi32(0, self.raw_N, self.sort_type,
                                             self.simd_type)
        else:
            return build_mask_bool_vec_epi32(3, self.raw_N, self.sort_type,
                                             self.simd_type)

    def generate_instruction(self):
        header.stdint = True

        sort_bytes = self.raw_N * self.sort_type.sizeof()
        if INT_ALIGNED is False or self.fill is True:
            if (sort_bytes % 4) != 0:
                self.extra_insert = True

        if (sort_bytes % 4) != 0 and self.extra_insert is True:
            return self.generate_instruction_misaligned()
        else:
            return self.generate_instruction_base()

    def generate_instruction_base(self):
        store_bool_vec = self.build_store_bool_vec_wrapper()
        list_store_bool_vec = arr_to_csv(store_bool_vec, True)

        instruction = "{}_maskstore_epi32((int32_t * const)[ARR], {}_set_epi32([STORE_BOOLS]), [V])".format(
            self.simd_type.prefix(),
            self.simd_type.prefix()).replace("[STORE_BOOLS]",
                                             list_store_bool_vec)
        return instruction

    def generate_instruction_misaligned(self):
        T_size = self.T_size
        err_assert(T_size < 4, "building lt with T_size >= 4")
        instruction = self.generate_instruction_base()

        round_base = int(4 / T_size)
        if self.raw_N % round_base == 0:
            return instruction

        instruction += ";"
        instruction += "\n"

        remainder = self.raw_N % round_base
        err_assert(round_base != 0, "fucked up my logic")

        truncated_N = self.raw_N - remainder
        err_assert(truncated_N % round_base == 0, "fucked up my math")

        instruction += "const uint32_t [TMP0] = {}_extract_epi32([V], [TAIL_IDX]);".format(
            self.simd_type.prefix()).replace("[TAIL_IDX]",
                                             str(int(self.raw_N / round_base)))
        instruction += "\n"

        shift = 8 * T_size
        mask = (int(1) << shift) - 1
        err_assert(remainder > 0 and remainder < 4,
                   "this should be impossible")

        instruction += "__builtin_memcpy([ARR] + [PLACE_IDX], &[TMP0], [NBYTES]);".replace(
            "[PLACE_IDX]",
            str(truncated_N)).replace("[NBYTES]",
                                      str(remainder * self.sort_type.sizeof()))
        return instruction


class SIMD_Store():
    def __init__(self, raw_N, sort_type, scaled_sort_N):
        self.instructions = [
            ###################################################################
            SIMD_Full_Store("(*((_aliasing_m64_ *)[ARR])) = [V]",
                            scaled_sort_N, False, SIMD_m64(), ["MMX"], 0),
            SIMD_Partial_Store_m64(1, raw_N, SIMD_m64(), ["MMX"], 1),
            SIMD_Partial_Store_m64(2, raw_N, SIMD_m64(), ["MMX"], 1),
            ###################################################################
            # Universal __m128i stores
            SIMD_Full_Store("_mm_store_si128((__m128i *)[ARR], [V])",
                            scaled_sort_N, True, SIMD_m128(), ["SSE2"], 0),
            SIMD_Full_Store("_mm_storeu_si128((__m128i *)[ARR], [V])",
                            scaled_sort_N, False, SIMD_m128(), ["SSE2"], 1),

            # Universal fallback
            SIMD_Mask_Store_Fallback_As_Epi32(scaled_sort_N, sort_type, raw_N,
                                              SIMD_m128(), ["AVX2", "SSE4.1"],
                                              25),
            SIMD_Mask_Store(
                "_mm_mask_storeu_epi8((void *)[ARR], [STORE_MASK], [V])", 1,
                False, raw_N, SIMD_m128(), ["AVX512bw", "AVX512vl"], 2),
            SIMD_Mask_Store(
                "_mm_mask_storeu_epi16((void *)[ARR], [STORE_MASK], [V])", 2,
                False, raw_N, SIMD_m128(), ["AVX512bw", "AVX512vl"], 2),
            SIMD_Mask_Store(
                "_mm_mask_store_epi32((void *)[ARR], [STORE_MASK], [V])", 4,
                True, raw_N, SIMD_m128(), ["AVX512f", "AVX512vl"], 2),
            SIMD_Mask_Store(
                "_mm_mask_storeu_epi32((void *)[ARR], [STORE_MASK], [V])", 4,
                False, raw_N, SIMD_m128(), ["AVX512f", "AVX512vl"], 3),
            SIMD_Mask_Store(
                "_mm_mask_store_epi64((void *)[ARR], [STORE_MASK], [V])", 8,
                True, raw_N, SIMD_m128(), ["AVX512f", "AVX512vl"], 2),
            SIMD_Mask_Store(
                "_mm_mask_storeu_epi64((void *)[ARR], [STORE_MASK], [V])", 8,
                False, raw_N, SIMD_m128(), ["AVX512f", "AVX512vl"], 3),
            ###################################################################
            # Universal __m256i stores
            SIMD_Full_Store("_mm256_store_si256((__m256i *)[ARR], [V])",
                            scaled_sort_N, True, SIMD_m256(), ["AVX"], 0),
            SIMD_Full_Store("_mm256_storeu_si256((__m256i *)[ARR], [V])",
                            scaled_sort_N, False, SIMD_m256(), ["AVX"], 1),

            # Universal fallback
            SIMD_Mask_Store_Fallback_As_Epi32(scaled_sort_N, sort_type, raw_N,
                                              SIMD_m256(), ["AVX2", "AVX"],
                                              25),
            SIMD_Mask_Store(
                "_mm256_mask_storeu_epi8((void *)[ARR], [STORE_MASK], [V])", 1,
                False, raw_N, SIMD_m256(), ["AVX512bw", "AVX512vl"], 2),
            SIMD_Mask_Store(
                "_mm256_mask_storeu_epi16((void *)[ARR], [STORE_MASK], [V])",
                2, False, raw_N, SIMD_m256(), ["AVX512bw", "AVX512vl"], 2),
            SIMD_Mask_Store(
                "_mm256_mask_store_epi32((void *)[ARR], [STORE_MASK], [V])", 4,
                True, raw_N, SIMD_m256(), ["AVX512f", "AVX512vl"], 2),
            SIMD_Mask_Store(
                "_mm256_mask_storeu_epi32((void *)[ARR], [STORE_MASK], [V])",
                4, False, raw_N, SIMD_m256(), ["AVX512f", "AVX512vl"], 3),
            SIMD_Mask_Store(
                "_mm256_mask_store_epi64((void *)[ARR], [STORE_MASK], [V])", 8,
                True, raw_N, SIMD_m256(), ["AVX512f", "AVX512vl"], 2),
            SIMD_Mask_Store(
                "_mm256_mask_storeu_epi64((void *)[ARR], [STORE_MASK], [V])",
                8, False, raw_N, SIMD_m256(), ["AVX512f", "AVX512vl"], 3),
            ###################################################################
            # Universal __m512i stores
            SIMD_Full_Store("_mm512_store_si512((__m512i *)[ARR], [V])",
                            scaled_sort_N, True, SIMD_m512(), ["AVX512f"], 0),
            SIMD_Full_Store("_mm512_storeu_si512((__m512i *)[ARR], [V])",
                            scaled_sort_N, False, SIMD_m512(), ["AVX512f"], 1),

            # Universal fallback
            SIMD_Mask_Store(
                "_mm512_mask_storeu_epi8((void *)[ARR], [STORE_MASK], [V])", 1,
                False, raw_N, SIMD_m512(), ["AVX512bw", "AVX512vl"], 2),
            SIMD_Mask_Store(
                "_mm512_mask_storeu_epi16((void *)[ARR], [STORE_MASK], [V])",
                2, False, raw_N, SIMD_m512(), ["AVX512bw", "AVX512vl"], 2),
            SIMD_Mask_Store(
                "_mm512_mask_store_epi32((void *)[ARR], [STORE_MASK], [V])", 4,
                True, raw_N, SIMD_m512(), ["AVX512f", "AVX512vl"], 2),
            SIMD_Mask_Store(
                "_mm512_mask_storeu_epi32((void *)[ARR], [STORE_MASK], [V])",
                4, False, raw_N, SIMD_m512(), ["AVX512f", "AVX512vl"], 3),
            SIMD_Mask_Store(
                "_mm512_mask_store_epi64((void *)[ARR], [STORE_MASK], [V])", 8,
                True, raw_N, SIMD_m512(), ["AVX512f", "AVX512vl"], 2),
            SIMD_Mask_Store(
                "_mm512_mask_storeu_epi64((void *)[ARR], [STORE_MASK], [V])",
                8, False, raw_N, SIMD_m512(), ["AVX512f", "AVX512vl"], 3),
            ###################################################################
        ]


######################################################################
def instruction_filter(instructions,
                       sort_type,
                       simd_type,
                       aligned=None,
                       full_load=None):
    ops = []
    for operation in instructions:
        if operation.match(Match_Info(sort_type, simd_type, aligned,
                                      full_load)) is True:
            ops.append(operation)
    err_assert(
        len(ops) > 0,
        "no operations left after filter {} / {}".format(0, len(instructions)))
    return ops


def best_instruction(instructions):
    min_weight = instructions[0].weight
    min_idx = 0
    for i in range(1, len(instructions)):
        if instructions[i].weight < min_weight:
            min_weight = instructions[i].weight
            min_idx = i

    return instructions[min_idx]


######################################################################
# Compare Exchange


def make_returnable(raw_str, simd_type_str):
    ops = raw_str.split("\n")
    err_assert(len(ops) > 0, "empty string")
    nops = len(ops)
    ops[nops - 1] = "[VTYPE] [V] = " + ops[nops - 1] + ";"

    out = ""
    for op in ops:
        out += op.replace("[VTYPE]", simd_type_str) + "\n"
    return out


def order_str_tmps(raw_str, base):
    ntmps = 0
    for i in range(0, 100):
        new_tmps = raw_str.count("[TMP{}]".format(i))
        if new_tmps == 0:
            break
        ntmps += 1

    for i in range(0, ntmps):
        raw_str = raw_str.replace("[TMP{}]".format(i),
                                  "[OTMP{}]".format(i + base))
    return ntmps, raw_str


class Output_Formatter():
    def __init__(self, raw_output):
        self.raw_output = raw_output
        self.tablen = 4

    def get_fmt_output(self):
        if CLANG_FORMAT_EXE == "" or os.system(
                "which {} > /dev/null".format(CLANG_FORMAT_EXE)) != 0:
            return self.fallback_fmt()

        cmd = "{} --style=file --fallback-style=google".format(
            CLANG_FORMAT_EXE)
        try:
            sproc = subprocess.Popen(cmd,
                                     shell=True,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)

            stdout_data, stderr_data = sproc.communicate(
                input=self.raw_output.encode(), timeout=2)
            if sproc.returncode != 0:
                return self.fallback_fmt()
            if len(stderr_data) != 0:
                return self.fallback_fmt()
            if len(stdout_data) == 0:
                return self.fallback_fmt()

            stdout_data = stdout_data.decode("utf-8", "ignore")
            stderr_data = stderr_data.decode("utf-8", "ignore")
            return stdout_data
        except OSError:
            return self.fallback_fmt()
        except subprocess.TimeoutExpired:
            return self.fallback_fmt()

    def find_end_brace(self, line):
        pos = 0
        for i in range(0, min(len(line), 48)):
            if line[i] == '(':
                pos = i + 1
        return pos

    def padding(self, t):
        out = ""
        for i in range(0, t + 1):
            out += " "
        return out

    def tab_padding(self, ntabs):
        return self.padding(self.tablen * ntabs)

    def fallback_fmt(self):
        fmt_output = ""
        output = self.raw_output.split("\n")

        started = False
        tabs = 0

        extra_spacing = 0
        for lines in output:
            lines += "\n"
            if "__attribute__((" in lines:
                lines = lines.replace(")) ", ")) \n")

            if "{" in lines:
                started = True
            if "}" in lines:
                started = False
                tabs -= 1

            if started is False:
                fmt_output += lines
                continue

            if extra_spacing != 0:
                fmt_output += self.padding(extra_spacing)

            words = lines.split(" ")
            base_len = self.find_end_brace(lines)
            cur_len = 0
            fmt_line = self.tab_padding(tabs)
            in_quotes = False
            for w in words:
                if "\"" in w:
                    if in_quotes is True:
                        in_quotes = False
                    else:
                        in_quotes = True
                if cur_len + len(
                        w
                ) + 1 > 72 and cur_len >= base_len and in_quotes is False and "\"" not in w:
                    fmt_line += "\n" + self.tab_padding(tabs) + self.padding(
                        base_len)
                    cur_len = base_len

                fmt_line += w + " "
                if "__attribute__" in w:
                    cur_len = 0

                cur_len += len(w) + 1
            fmt_output += fmt_line
            if "{" in lines:
                tabs += 1
            if "asm" in lines:
                extra_spacing += len("asm volatile")
            if ":);\n" in lines:
                extra_spacing = 0

        return fmt_output


class Output_Generator():
    def __init__(self, header_info, CAS_info, algorithm_name, depth, N, raw_N,
                 sort_type):

        self.header_info = header_info
        self.CAS_info = CAS_info
        self.algorithm_name = algorithm_name
        self.vec_algorithm_name = algorithm_name
        self.base_algorithm_name = algorithm_name
        if N != raw_N:
            self.base_algorithm_name = "best"
        self.N = N
        self.raw_N = raw_N
        self.depth = depth
        self.sort_type = sort_type

        self.simd_type = get_simd_type(N * sort_type.sizeof())
        self.CAS_info_str = self.CAS_info.get()
        self.loadnstore_ops = self.CAS_info.load.count(
            self.simd_type.prefix()) + self.CAS_info.store.count(
                self.simd_type.prefix())
        self.logic_ops = self.CAS_info_str.count(
            self.simd_type.prefix()) - self.loadnstore_ops

        global TEMPLATED
        self.templated = TEMPLATED

        global SORT_FUNC_NAME
        self.sort_to_str = SORT_FUNC_NAME
        self.sort_to_str_v = self.sort_to_str + "_vec"
        if self.sort_to_str == "":
            if TEMPLATED is False:
                sign = "s"
                if sort_type.sign == Sign.UNSIGNED:
                    sign = "u"
                self.sort_to_str = "{}_{}_{}{}".format(
                    self.base_algorithm_name, raw_N, sort_type.sizeof(), sign)
                self.sort_to_str_v = "{}_{}_{}{}_vec".format(
                    self.vec_algorithm_name, N, sort_type.sizeof(), sign)
            else:
                self.sort_to_str = "vsort"
                self.sort_to_str_v = ""

        full_load_and_store = EXTRA_MEMORY
        if full_load_and_store is False:
            full_load_and_store = N * sort_type.sizeof(
            ) == self.simd_type.sizeof()

        simd_restrictions = SIMD_RESTRICTIONS
        if simd_restrictions != "":
            simd_restrictions += "*"
        else:
            simd_restrictions = "None"

        opt_pref = "space"
        if INSTRUCTION_OPT == Optimization.UOP:
            opt_pref = "uop"

        self.impl_info = [
            "Sorting Network Information:",
            "\tSort Size                        : {}".format(
                self.raw_N), "\tUnderlying Sort Type             : {}".format(
                    self.sort_type.to_string()),
            "\tNetwork Generation Algorithm     : {}".format(algorithm_name),
            "\tNetwork Depth                    : {}".format(depth),
            "\tSIMD Instructions                : {} / {}".format(
                self.loadnstore_ops, self.logic_ops),
            "\tOptimization Preference          : {}".format(opt_pref),
            "\tSIMD Type                        : {}".format(
                self.simd_type.to_string()),
            "\tSIMD Instruction Set(s) Used     : {}".format(
                arr_to_padd_str(self.CAS_info.get_instruction_sets(),
                                "\tSIMD Instruction Set(s) Used     : ")),
            "\tSIMD Instruction Set(s) Excluded : {}".format(
                simd_restrictions),
            "\tAligned Load & Store             : {}".format(
                str(ALIGNED_ACCESS)),
            "\tInteger Aligned Load & Store     : {}".format(
                str(ALIGNED_ACCESS)),
            "\tFull Load & Store                : {}".format(
                str(full_load_and_store))
        ]
        if self.raw_N != self.N:
            self.impl_info.insert(
                2, "\tScaled Sort Size                 : {}".format(self.N))

        self.perf_notes = [
            "Performance Notes:",
            "1) If you are sorting an array where there IS valid memory up to the nearest sizeof a SIMD register, you will get an improvement enable \"EXTRA_MEMORY\" (this turns on \"Full Load & Store\". Note that enabling \"Full Load & Store\" will not modify any of the memory not being sorted and will not affect the sort in any way. i.e sort(3) [4, 3, 2, 1] with full load will still return [2, 3, 4, 1]. Note even if you don't have enough memory for a full SIMD register, enabling \"INT_ALIGNED\" will also improve load efficiency and only requires that there is valid memory up the next factor of sizeof(int).",
            "2) If your sort size is not a power of 2 you are likely running into less efficient instructions. This is especially noticable when sorting 8 bit and 16 bit values. If rounding you sort size up to the next power of 2 will not cost any additional depth it almost definetly worth doing so. The \"Best\" Network Algorithm automatically does this in many cases.",
            "3) There are two optimization settings, \"Optimization.SPACE\" and \"Optimization.UOP\". The former will essentially break ties by picking the instruction that uses less memory (i.e doesn't have to store a register's initializing in memory. The latter will break ties but simply selecting whatever instructions use the least UOPs. Which is best is probably application dependent. Note that while \"Optimization.SPACE\" will save .rodata memory it will often cost more in .text memory. Generally it is advise to optimize for space if you are calling sparingly and uop if you are calling sort in a loop."
        ]

        for i in range(1, len(self.perf_notes)):
            words = self.perf_notes[i].split()
            self.perf_notes[i] = ""

            line_len = 0
            for w in words:
                line_len += (len(w) + 1)
                self.perf_notes[i] += w + " "
                if line_len > 56:
                    line_len = 0
                    self.perf_notes[i] += "\n"

            s = self.perf_notes[i].split("\n")
            self.perf_notes[i] = s[0] + "\n"
            for j in range(1, len(s)):
                if len(s[j].strip()) == 0:
                    continue
                self.perf_notes[i] += "   " + s[j].strip()
                self.perf_notes[i] += "\n"

    def get_info(self):
        blend_weight, perm_weight, load_weight = self.CAS_info.get_weights()
        return self.algorithm_name, self.depth, self.logic_ops, blend_weight, perm_weight, load_weight

    def get(self):
        return self.get_header() + self.get_content() + self.get_tail()

    def get_header(self):
        head = ""
        head += "\n"
        head += "/*"
        head += "\n\n"
        head += arr_to_str(self.impl_info)
        head += "\n\n"
        head += arr_to_str(self.perf_notes)
        head += "\n"
        head += " */"
        head += "\n\n"
        head += self.header_info.get_headers(self.sort_type)
        head += "\n\n"
        if self.templated is True:
            head += "template<typename T, uint32_t n>"
            head += "\n"
            head += "struct {};".format("vsort" if SORT_FUNC_NAME ==
                                        "" else SORT_FUNC_NAME)
            head += "\n\n"

            head += "template<>"
            head += "\n"
            head += "struct {}<{}, {}>".format(self.sort_to_str,
                                               self.sort_type.to_string(),
                                               self.raw_N)
            head += " {"
            head += "\n"
        return head

    def get_content(self):
        if self.templated is True:
            return self.CAS_info.get().replace("[FUNCNAME]", "sort").replace(
                "[FUNCNAME_VEC]", "sort_vec")
        else:
            return self.CAS_info.get().replace("[FUNCNAME]",
                                               self.sort_to_str).replace(
                                                   "[FUNCNAME_VEC]",
                                                   self.sort_to_str_v)

    def get_tail(self):
        tail = ""
        if self.templated is True:
            tail += "};"
            tail += "\n"
        tail += "\n\n"
        return tail


class CAS_Output_Generator():
    def __init__(self, N, sort_type):
        self.simd_type = get_simd_type(N * sort_type.sizeof())
        self.sort_type = sort_type
        self.N = N

        self.load = ""
        self.CAS = []
        self.store = ""

        self.constraints = []

        self.tmp_count = 0

        self.arr_name = "arr"
        self.v_name = "v"
        self.last_v_name = ""
        self.tmp_name = "_tmp"

        self.already_prepared_content = False

        self.cas_blend_weight = 0
        self.cas_perm_weight = 0
        self.cas_load_weight = 0

    def append_cas(self, cas):
        self.CAS.append(cas)
        self.add_constraints(cas.constraints)

    def add_constraints(self, constraints):
        self.constraints += constraints

    def add_load(self, load):
        self.load = load

    def add_store(self, store):
        self.store = store

    def get(self):
        self.prepare_content()
        return self.get_inner() + "\n\n\n" + self.get_wrapper()

    def get_wrapper(self):
        return self.get_wrapper_head() + "\n" + self.get_wrapper_content(
        ) + "\n" + self.get_wrapper_tail()

    def get_inner(self):
        return self.get_inner_head() + "\n" + self.get_inner_content(
        ) + "\n" + self.get_inner_tail()

    def get_instruction_sets(self):
        return list(dict.fromkeys(self.constraints))

    def get_weights(self):
        return self.cas_blend_weight, self.cas_perm_weight, self.cas_load_weight

    def set_weights(self, blend_weight, perm_weight, load_weight):
        self.cas_blend_weight = blend_weight
        self.cas_perm_weight = perm_weight
        self.cas_load_weight = load_weight

    def prepare_content(self):
        if self.already_prepared_content is True:
            return

        self.already_prepared_content = True

        self.load = make_returnable(self.load, self.simd_type.to_string())
        self.tmp_count, self.load = order_str_tmps(self.load, 0)
        for i in range(0, self.tmp_count):
            self.load = self.load.replace("[OTMP{}]".format(i),
                                          "{}{}".format(self.tmp_name, i))
        self.load = self.load.replace("[ARR]", self.arr_name)
        self.load = self.load.replace("[V]", self.v_name)

        for i in range(0, len(self.CAS)):
            last_v = self.v_name
            if i != 0:
                last_v = "{}{}".format(self.v_name, i - 1)
            v_names = [
                "perm{}".format(i), "min{}".format(i), "max{}".format(i),
                "{}{}".format(self.v_name,
                              i), "min{}".format(i - 1), "max{}".format(i - 1)
            ]
            self.last_v_name = v_names[3]

            self.tmp_count = self.CAS[i].make_operation(
                last_v, v_names, self.tmp_count, self.tmp_name,
                self.simd_type.to_string())

        tmp_max, self.store = order_str_tmps(self.store, self.tmp_count)
        tmp_max += self.tmp_count
        for i in range(self.tmp_count, tmp_max):
            self.store = self.store.replace("[OTMP{}]".format(i),
                                            "{}{}".format(self.tmp_name, i))

        self.store = self.store.replace("[ARR]", self.arr_name)
        self.store = self.store.replace("[V]", self.v_name)
        self.store = self.store.replace("[VTYPE]", self.simd_type.to_string())
        self.store += ";\n"

    def get_wrapper_head(self):
        head = ""
        global TEMPLATED
        if TEMPLATED is False:
            head += "#ifndef _SIMD_SORT_ARR_[FUNCNAME]_H_"
            head += "\n"
            head += "#define _SIMD_SORT_ARR_[FUNCNAME]_H_"
            head += "\n\n"

        head += "/* Wrapper For SIMD Sort */"
        head += "\n"
        if TEMPLATED is True:
            head += "static "
        head += "void inline __attribute__((always_inline)) [FUNCNAME]([VTYPE] * const [ARR]) {".replace(
            "[ARR]", self.arr_name).replace("[VTYPE]",
                                            self.sort_type.to_string())
        head += "\n"
        return head

    def get_wrapper_content(self):
        content = self.load
        content += "\n"
        content += "[V] = [FUNCNAME_VEC]([V]);".replace("[V]", self.v_name)
        content += "\n"
        content += "\n"
        content += self.store
        return content

    def get_wrapper_tail(self):
        tail = "}\n"
        global TEMPLATED
        if TEMPLATED is False:
            tail += "#endif"
        tail += "\n"
        return tail

    def get_inner_head(self):
        head = ""
        global TEMPLATED
        if TEMPLATED is False:
            head += "#ifndef _SIMD_SORT_VEC_[FUNCNAME_VEC]_H_"
            head += "\n"
            head += "#define _SIMD_SORT_VEC_[FUNCNAME_VEC]_H_"
            head += "\n\n"

        head += "/* SIMD Sort */"
        head += "\n"
        if TEMPLATED is True:
            head += "static "
        head += "[VTYPE] __attribute__((const)) [FUNCNAME_VEC]([VTYPE] [V]) {".replace(
            "[VTYPE]", self.simd_type.to_string()).replace("[V]", self.v_name)
        head += "\n"
        return head

    def get_inner_content(self):
        ordered_content = []
        for i in range(0, len(self.CAS)):
            reorder, comments, perm, _min, _max, blend = self.CAS[
                i].get_operation()
            if reorder is True:
                err_assert(len(ordered_content) != 0, "about to have OOB")
                tmp = ordered_content[len(ordered_content) - 1]
                ordered_content[len(ordered_content) - 1] = comments
                perm_pieces = perm.split("\n")
                ordered_content.append(perm_pieces[0] + "\n")
                ordered_content.append(tmp)
                for i in range(1, len(perm_pieces)):
                    nl = "\n"
                    if i == len(perm_pieces) - 1:
                        nl = ""
                    ordered_content.append(perm_pieces[i] + nl)
                ordered_content.append("\n")
            else:
                ordered_content.append(comments)
                ordered_content.append(perm)
            ordered_content.append(_min)
            ordered_content.append(_max)
            ordered_content.append(blend)

        content = ""
        for oc in ordered_content:
            if "/* Pairs" in oc and content != "":
                content += "\n"
            content += oc

        return content

    def get_inner_tail(self):
        tail = "return [V];\n".replace("[V]", self.last_v_name)
        tail += "}\n"
        global TEMPLATED
        if TEMPLATED is False:
            tail += "#endif"
        tail += "\n"
        return tail


class Compare_Exchange():
    def __init__(self, raw_plist, raw_perm, raw_min, raw_max, raw_blend,
                 constraints):
        self.raw_plist = raw_plist
        self.raw_perm = raw_perm
        self.raw_min = raw_min
        self.raw_max = raw_max
        self.raw_blend = raw_blend

        self.constraints = constraints

        self.tmps_start = 0
        self.tmps_end = 0

        self.reorder = False

    def get_operation(self):
        return self.reorder, self.make_comment(
        ), self.raw_perm, self.raw_min, self.raw_max, self.raw_blend

    def make_comment(self):
        pair_list_str = ""
        perm_list_str = ""
        skips = []
        for i in range(0, len(self.raw_plist)):

            i_idx = (len(self.raw_plist) - 1) - i
            max_idx = max(self.raw_plist[i], i_idx)
            min_idx = min(self.raw_plist[i], i_idx)
            key = (int(max_idx) << 10) | min_idx

            perm_str = str(self.raw_plist[i])
            pair_str = "[{},{}]".format(perm_str, str(i_idx))

            if len(perm_str) == 1:
                perm_str = " " + perm_str

            if i != 0:
                if key not in skips:
                    pair_list_str += ", "
                perm_list_str += ", "
            if key not in skips:
                pair_list_str += pair_str
            perm_list_str += perm_str
            skips.append(key)

        return "/* Pairs: ({}) */\n/* Perm:  ({}) */\n".format(
            pair_list_str, perm_list_str)

    def make_operation(self, last_v, v_names, tmp_base, tmp_name,
                       simd_type_str):
        tmp_base = self.order_temporaries(tmp_base)
        self.make_strings(last_v, v_names, tmp_name, simd_type_str)
        return tmp_base

    def make_strings(self, last_v, v_names, tmp_name, simd_type_str):
        self.set_complete(simd_type_str)
        self.set_v(v_names)
        self.set_tmp(tmp_name)
        self.set_input_arguments(last_v, v_names)

    def set_complete(self, simd_type_str):
        self.raw_perm = make_returnable(self.raw_perm, simd_type_str)
        self.raw_min = make_returnable(self.raw_min, simd_type_str)
        self.raw_max = make_returnable(self.raw_max, simd_type_str)
        self.raw_blend = make_returnable(self.raw_blend, simd_type_str)

    def set_v(self, v_names):
        self.raw_perm = self.raw_perm.replace("[V]", v_names[0])
        self.raw_min = self.raw_min.replace("[V]", v_names[1])
        self.raw_max = self.raw_max.replace("[V]", v_names[2])
        self.raw_blend = self.raw_blend.replace("[V]", v_names[3])

    def set_tmp(self, tmp_name):
        for i in range(self.tmps_start, self.tmps_end):
            self.raw_perm = self.raw_perm.replace("[OTMP{}]".format(i),
                                                  "{}{}".format(tmp_name, i))
            self.raw_min = self.raw_min.replace("[OTMP{}]".format(i),
                                                "{}{}".format(tmp_name, i))
            self.raw_max = self.raw_max.replace("[OTMP{}]".format(i),
                                                "{}{}".format(tmp_name, i))
            self.raw_blend = self.raw_blend.replace("[OTMP{}]".format(i),
                                                    "{}{}".format(tmp_name, i))

    def set_input_arguments(self, last_v, v_names):
        if "[V2]" in self.raw_perm:
            self.reorder = True
            self.raw_perm = self.raw_perm.replace("[V1]", v_names[5]).replace(
                "[V2]", v_names[4])
        else:
            self.raw_perm = self.raw_perm.replace("[V1]", last_v)

        self.raw_min = self.raw_min.replace("[V1]", last_v).replace(
            "[V2]", v_names[0])
        self.raw_max = self.raw_max.replace("[V1]", last_v).replace(
            "[V2]", v_names[0])
        self.raw_blend = self.raw_blend.replace("[V1]", v_names[2]).replace(
            "[V2]", v_names[1])

    def order_temporaries(self, base):
        self.tmps_start = base
        ntmps, self.raw_perm = order_str_tmps(self.raw_perm, base)
        base += ntmps
        ntmps, self.raw_min = order_str_tmps(self.raw_min, base)
        base += ntmps
        ntmps, self.raw_max = order_str_tmps(self.raw_max, base)
        base += ntmps
        ntmps, self.raw_blend = order_str_tmps(self.raw_blend, base)
        base += ntmps
        self.tmps_end = base
        return base


class Compare_Exchange_Generator():
    def __init__(self, pairs, N, did_scale_N, sort_type):

        self.total_blend_weight = 0
        self.total_perm_weight = 0

        self.pairs = copy.deepcopy(pairs)
        self.sort_N = sort_n(N, sort_type.sizeof())
        self.sort_type = sort_type
        self.simd_type = get_simd_type(self.sort_N * self.sort_type.sizeof())
        err_assert(self.simd_type.sizeof() >= self.sort_N * sort_type.sizeof(),
                   "invalid SIMD type selection")

        header.immintrin = True
        if self.simd_type.sizeof() == 8:
            header.xmmintrin = True

        self.SIMD_min = instruction_filter(SIMD_Min().instructions,
                                           self.sort_type, self.simd_type)
        self.SIMD_max = instruction_filter(SIMD_Max().instructions,
                                           self.sort_type, self.simd_type)

        self.cas_output_generator = CAS_Output_Generator(N, sort_type)

        do_full = EXTRA_MEMORY
        if self.simd_type.sizeof() == N * sort_type.sizeof():
            do_full = True

        if do_full is True and self.simd_type.sizeof() == 8:
            header.aliasing_m64 = True

        self.SIMD_load = instruction_filter(
            SIMD_Load(N, sort_type, did_scale_N).instructions, self.sort_type,
            self.simd_type, ALIGNED_ACCESS, do_full)

        self.SIMD_store = instruction_filter(
            SIMD_Store(N, sort_type, did_scale_N).instructions, self.sort_type,
            self.simd_type, ALIGNED_ACCESS, do_full)

    def Generate_Instructions(self):
        best_load = best_instruction(self.SIMD_load)
        self.cas_output_generator.add_constraints(best_load.constraints)
        self.cas_output_generator.add_load(best_load.generate_instruction())

        for i in range(0, int(len(self.pairs) / self.sort_N)):
            self.cas_output_generator.append_cas(self.Make_Compare_Exchange(i))

        best_store = best_instruction(self.SIMD_store)
        self.cas_output_generator.add_constraints(best_store.constraints)
        self.cas_output_generator.add_store(best_store.generate_instruction())

        self.cas_output_generator.set_weights(self.total_blend_weight,
                                              self.total_perm_weight,
                                              best_load.weight)

        return self.cas_output_generator

    def Make_Compare_Exchange(self, cas_idx):
        cas_prev_perm = []
        cas_perm = []
        for i in range(cas_idx * self.sort_N, (cas_idx + 1) * self.sort_N):
            cas_perm.append(self.pairs[i])
        if cas_idx != 0:
            for i in range((cas_idx - 1) * self.sort_N,
                           (cas_idx) * self.sort_N):
                cas_prev_perm.append(self.pairs[i])

        SIMD_blend = instruction_filter(
            SIMD_Blend(cas_perm).instructions, self.sort_type, self.simd_type)
        SIMD_permute = instruction_filter(
            SIMD_Permute(cas_prev_perm, cas_perm).instructions, self.sort_type,
            self.simd_type)

        best_permutate = best_instruction(SIMD_permute)
        best_min = best_instruction(self.SIMD_min)
        best_max = best_instruction(self.SIMD_max)
        best_blend = best_instruction(SIMD_blend)

        self.total_blend_weight += best_blend.weight
        self.total_perm_weight += best_permutate.weight

        compare_exchange = Compare_Exchange(
            cas_perm, best_permutate.generate_instruction(),
            best_min.generate_instruction(), best_max.generate_instruction(),
            best_blend.generate_instruction(),
            best_permutate.constraints + best_min.constraints +
            best_max.constraints + best_blend.constraints)

        return compare_exchange


######################################################################
# Network Generation
######################################################################


def next_p2(v):
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v += 1
    return v


def sort_n(N, T_size):
    new_N = next_p2(N)
    if new_N * T_size < 8:
        return int(8 / T_size)
    return new_N


class Transform():
    def __init__(self, N, scaled, sort_type, pairs):
        self.N = N
        self.scaled = scaled
        self.sort_type = sort_type

        self.pairs = copy.deepcopy(pairs)

        self.depth = -1

    def unidirectional(self):
        for i in range(0, len(self.pairs), 2):
            x_i = self.pairs[i]
            y_i = self.pairs[i + 1]

            if x_i > y_i:
                for j in range(i + 2, len(self.pairs), 2):
                    x_j = self.pairs[j]
                    y_j = self.pairs[j + 1]

                    if x_i == x_j:
                        self.pairs[j] = y_i
                    if x_i == y_j:
                        self.pairs[j + 1] = y_i
                    if y_i == x_j:
                        self.pairs[j] = x_i
                    if y_i == y_j:
                        self.pairs[j + 1] = x_i

                self.pairs[i] = y_i
                self.pairs[i + 1] = x_i

    def group(self):
        grouped_pairs = []
        for i in range(0, len(self.pairs), 2):
            x_i = self.pairs[i]
            y_i = self.pairs[i + 1]

            save_idx = -1
            for idx in range(len(grouped_pairs) - 1, -1, -1):
                for j in range(0, len(grouped_pairs[idx]), 2):
                    x_j = grouped_pairs[idx][j]
                    y_j = grouped_pairs[idx][j + 1]
                    if x_i == x_j or x_i == y_j or y_i == x_j or y_i == y_j:
                        save_idx = idx
                        break
                if save_idx != (-1):
                    break
            save_idx += 1
            if save_idx == len(grouped_pairs):
                grouped_pairs.append([])

            err_assert(save_idx < len(grouped_pairs), "invalid idx")

            grouped_pairs[save_idx].append(x_i)
            grouped_pairs[save_idx].append(y_i)

        pairs_len = len(self.pairs)
        pidx = 0
        for i in range(0, len(grouped_pairs)):
            for j in range(0, len(grouped_pairs[i])):
                self.pairs[pidx] = grouped_pairs[i][j]
                pidx += 1
        self.depth = len(grouped_pairs)
        err_assert(pidx == pairs_len,
                   "not all pairs processed {} != {}".format(pidx, pairs_len))

    def permutation(self):
        err_assert(self.depth != -1, "permutating before grouping")

        sort_N = sort_n(self.N, self.sort_type.sizeof())

        perm_arr = []
        for i in range(0, self.depth * sort_N):
            perm_arr.append((sort_N - 1) - (i % sort_N))

        current_group = int(0)
        idx = 0

        for i in range(0, self.depth):
            while idx < len(self.pairs):
                if (current_group &
                    (int(1) << int(self.pairs[idx]))) != int(0) or (
                        current_group &
                        (int(1) << int(self.pairs[idx + 1]))) != int(0):
                    current_group = int(0)
                    break
                current_group |= (int(1) << int(self.pairs[idx]))
                current_group |= (int(1) << int(self.pairs[idx + 1]))
                perm_arr[i * sort_N +
                         ((sort_N - 1) - self.pairs[idx])] = self.pairs[idx +
                                                                        1]
                perm_arr[i * sort_N + (
                    (sort_N - 1) - self.pairs[idx + 1])] = self.pairs[idx]

                idx += 2

        global INT_ALIGNED
        global EXTRA_MEMORY

        if ((INT_ALIGNED is False or
             ((self.N * self.sort_type.sizeof()) % 4 == 0)) and
            (self.N == sort_N
             or EXTRA_MEMORY is False)) or self.scaled is True:
            cur = self.N
            max_mod = 1
            while cur % max_mod == 0:
                max_mod = 2 * max_mod
            max_mod = int(max_mod / 2)
            if max_mod != 1:
                for i in range(self.depth):
                    for j in range(0, sort_N - self.N, max_mod):
                        tmp = []
                        for k in range(0, max_mod):
                            tmp.append(perm_arr[(i * sort_N) + j + (max_mod -
                                                                    (k + 1))])
                        for k in range(0, max_mod):
                            perm_arr[(i * sort_N) + j + k] = tmp[k]

        self.pairs = copy.deepcopy(perm_arr)
        err_assert(len(self.pairs) == len(perm_arr), "bad usage of deep copy")


class Bitonic():
    def __init__(self, N):
        self.name = "bitonic"
        self.N = N
        self.pairs = []
        self.depth = None

    def bitonic_merge(self, lo, n, direction):
        if n > 1:
            m = next_p2(n) >> 1
            for i in range(lo, (lo + n) - m):
                if direction is True:
                    self.pairs.append(i)
                    self.pairs.append(i + m)
                else:
                    self.pairs.append(i + m)
                    self.pairs.append(i)

            self.bitonic_merge(lo, m, direction)
            self.bitonic_merge(lo + m, n - m, direction)

    def bitonic_sort(self, lo, n, direction):
        if n > 1:
            m = n >> 1
            self.bitonic_sort(lo, m, not direction)
            self.bitonic_sort(lo + m, n - m, direction)
            self.bitonic_merge(lo, n, direction)

    def valid(self):
        return True

    def create_pairs(self):
        self.bitonic_sort(0, self.N, True)

        transformer = Transform(self.N, False, Sort_Type(1, True), self.pairs)
        transformer.unidirectional()

        self.pairs = copy.deepcopy(transformer.pairs)
        return self.pairs


class Batcher():
    def __init__(self, N):
        self.name = "batcher"
        self.N = N
        self.pairs = []
        self.depth = None

    def valid(self):
        return True

    def create_pairs(self):
        m = int(next_p2(self.N) / 2)
        while (m > 0):
            q = int(next_p2(self.N) / 2)
            r = 0
            d = m
            while (d > 0):
                for i in range(0, self.N - d):
                    if (i & m) == r:
                        self.pairs.append(i)
                        self.pairs.append(i + d)

                d = q - m
                q = int(q / 2)
                r = m

            m = int(m / 2)

        return self.pairs


class Oddeven():
    def __init__(self, N):
        self.name = "oddeven"
        self.N = N
        self.pairs = []
        self.depth = None

    def add_pair(self, i, j):
        if i < self.N and j < self.N:
            self.pairs.append(i)
            self.pairs.append(j)

    def merge(self, lo, n, r):
        m = 2 * r
        if m < n:
            self.merge(lo, n, m)
            self.merge(lo + r, n, m)
            for i in range(lo + r, (lo + n) - r, m):
                self.add_pair(i, i + r)
        else:
            self.add_pair(lo, lo + r)

    def sort(self, lo, n):
        if n > 1:
            m = int(n / 2)
            self.sort(lo, m)
            self.sort(lo + m, m)
            self.merge(lo, n, 1)

    def valid(self):
        return True

    def create_pairs(self):
        self.sort(0, next_p2(self.N))
        return self.pairs


class Bosenelson():
    def __init__(self, N):
        self.name = "bosenelson"
        self.N = N
        self.pairs = []
        self.depth = None

    def bn_merge(self, i, length_i, j, length_j):
        if length_i == 1 and length_j == 1:
            self.pairs.append(i)
            self.pairs.append(j)
        elif length_i == 1 and length_j == 2:
            self.pairs.append(i)
            self.pairs.append(j + 1)

            self.pairs.append(i)
            self.pairs.append(j)
        elif length_i == 2 and length_j == 1:
            self.pairs.append(i)
            self.pairs.append(j)

            self.pairs.append(i + 1)
            self.pairs.append(j)
        else:
            i_mid = int(length_i / 2)
            j_mid = int(0)
            if length_i % 2 == 1:
                j_mid = int(length_j / 2)
            else:
                j_mid = int((length_j + 1) / 2)

            self.bn_merge(i, i_mid, j, j_mid)
            self.bn_merge(i + i_mid, length_i - i_mid, j + j_mid,
                          length_j - j_mid)
            self.bn_merge(i + i_mid, length_i - i_mid, j, j_mid)

    def bn_split(self, i, length):
        if length >= 2:
            mid = int(length / 2)
            self.bn_split(i, mid)
            self.bn_split(i + mid, length - mid)
            self.bn_merge(i, mid, i + mid, length - mid)

    def valid(self):
        return True

    def create_pairs(self):
        self.bn_split(0, self.N)
        return self.pairs


class Minimum():
    def __init__(self, N):
        self.name = "minimum"
        self.N = N
        self.pairs = copy.deepcopy(min_pairs()[min(N, 32)])
        self.depth = None

    def valid(self):
        return self.N <= 32

    def create_pairs(self):
        err_assert(self.N < len(min_pairs()),
                   "No minimum network for N = {}".format(self.N))
        return self.pairs


class Weights():
    def __init__(self, perm_weight, blend_weight, load_weight,
                 instruction_weight, depth_weight, algorithm, npairs):
        self.perm_weight = perm_weight
        self.blend_weight = blend_weight
        self.load_weight = load_weight
        self.instruction_weight = instruction_weight
        self.depth_weight = depth_weight
        self.algorithm = algorithm
        self.npairs = copy.deepcopy(npairs)

    def val(self):
        return 2 * self.perm_weight + self.blend_weight + self.load_weight + 10 * self.depth_weight


class Best():
    def __init__(self, N):
        self.name = "best"
        self.N = N
        self.pairs = []
        self.depth = None

        self.options = []

    def create_options(self):
        max_N = next_p2(self.N) + 1
        # Don't think odd network size is ever optimal
        # attempts = [self.N]
        # min_N = (self.N - self.N % 2) + 2
        test_N = [self.N]
        for i in range(self.N + (self.N % 2), max_N, 2):
            test_N.append(i)
        for n in test_N:
            possible_bests = [
                Bitonic(n),
                Batcher(n),
                Oddeven(n),
                Minimum(n),
            ]
            for p in possible_bests:
                if p.valid() is True:
                    err_assert(p.N == n, "something is seriously wrong")
                    b = Builder(self.N, USER_TYPE, p.name, n)

                    name, depth, instructions, blend_weight, perm_weight, load_weight = b.Stats(
                    )
                    self.options.append(
                        Weights(perm_weight, blend_weight, load_weight,
                                instructions, depth, p, b.network_pairs))

    def create_orders(self):
        all_weights = copy.deepcopy(self.options)
        all_weights = sorted(all_weights, key=lambda w: w.val())

        self.pairs = copy.deepcopy(all_weights[0].npairs)
        self.name = all_weights[0].algorithm.name
        self.depth = all_weights[0].depth_weight
        self.N = all_weights[0].algorithm.N

    def create_pairs(self):
        self.create_options()
        self.create_orders()
        return self.pairs, self.depth


class Algorithms():
    def __init__(self, N):
        self.algorithms = [
            "bitonic", "batcher", "oddeven", "bosenelson", "minimum", "best"
        ]
        self.implementations = [
            Bitonic(N),
            Batcher(N),
            Oddeven(N),
            Bosenelson(N),
            Minimum(N),
            Best(N)
        ]
        err_assert(
            len(self.implementations) == len(self.algorithms),
            "algorithms and implementations arrays do not match")

    def get_algorithm(self, algorithm):

        for i in range(0, len(self.algorithms)):
            if algorithm.lower() == self.algorithms[i]:
                return self.implementations[i]
        err_assert(False, "No matching algorithm for " + algorithm)

    def valid_algorithm(self, algorithm):
        return algorithm.lower() in self.algorithms


class Network():
    def __init__(self, N, scaled, sort_type, algorithm_name):
        err_assert(N * sort_type.sizeof() <= 64, "N to large for network size")

        err_assert(
            Algorithms(N).valid_algorithm(algorithm_name),
            algorithm_name + " is unknown")

        self.N = N
        self.scaled = scaled
        self.sort_type = sort_type
        self.algorithm = Algorithms(N).get_algorithm(algorithm_name)
        self.depth = int(-1)
        self.network_N = N

    def create_pairs(self):
        ret = self.algorithm.create_pairs()
        self.network_N = self.algorithm.N
        return ret

    def get_network(self):
        transformer = Transform(self.N, self.scaled, self.sort_type,
                                self.create_pairs())
        transformer.group()
        transformer.permutation()
        self.depth = transformer.depth
        self.algorithm.depth = self.depth
        return transformer.pairs


class Builder():
    def __init__(self, N, sort_type, algorithm_name, network_N=None):
        algorithm_name = algorithm_name.lower()
        header.reset()
        self.N = N
        self.sort_type = sort_type
        self.algorithm_name = algorithm_name

        if network_N is None:
            network_N = N
        self.network_pairs = None
        self.network = Network(network_N, network_N > N, sort_type,
                               algorithm_name)
        if algorithm_name.lower() == "best":
            self.network_pairs, self.network.depth = self.network.create_pairs(
            )

        else:
            self.network_pairs = self.network.get_network()
        network_N = self.network.network_N
        self.network_N = network_N
        self.network_name = self.network.algorithm.name

        self.did_scale_N = network_N != self.N

        header.reset()
        self.cas_generator = Compare_Exchange_Generator(
            self.network_pairs, N, self.did_scale_N, self.network.sort_type)

        self.cas_info = self.cas_generator.Generate_Instructions()

    def Stats(self):
        full_output = Output_Generator(header, self.cas_info,
                                       self.algorithm_name, self.network.depth,
                                       self.network_N, self.N, self.sort_type)
        return full_output.get_info()

    def Build(self):

        full_output = Output_Generator(header, self.cas_info,
                                       self.network_name, self.network.depth,
                                       self.network_N, self.N, self.sort_type)
        out = full_output.get()

        if DO_FORMAT is True:
            output_fmt = Output_Formatter(out)
            return output_fmt.get_fmt_output()
        else:
            return out


######################################################################
# Main()
def main():
    global INSTRUCTION_OPT
    global SIMD_RESTRICTIONS
    global ALIGNED_ACCESS
    global INT_ALIGNED
    global EXTRA_MEMORY
    global DO_FORMAT
    global CLANG_FORMAT_EXE
    global USER_TYPE
    global SORT_FUNC_NAME
    global TEMPLATED
    global OUTFILE
    global FMODE

    args = parser.parse_args()

    FMODE = args.fmode
    OUTFILE = args.outfile

    SORT_FUNC_NAME = args.name
    TEMPLATED = args.template

    user_opt = args.optimization
    err_assert(user_opt == "space" or user_opt == "uop",
               "Invalid \"optimization\" flag")
    user_aligned = args.aligned
    user_extra_mem = args.extra_memory
    user_Constraint = args.constraint
    user_Int_Aligned = args.int_aligned
    user_Clang_Format_EXE = args.clang_format
    user_Format = args.no_format

    if user_opt == "space":
        INSTRUCTION_OPT = Optimization.SPACE
    elif user_opt == "uop":
        INSTRUCTION_OPT = Optimization.UOP
    else:
        err_assert(False, "have no idea wtf happened")

    CLANG_FORMAT_EXE = user_Clang_Format_EXE
    DO_FORMAT = user_Format

    SIMD_RESTRICTIONS = user_Constraint
    ALIGNED_ACCESS = user_aligned

    # It is impossible to fault with an aligned address
    if ALIGNED_ACCESS is True:
        EXTRA_MEMORY = True
    else:
        EXTRA_MEMORY = user_extra_mem

    INT_ALIGNED = user_Int_Aligned
    if EXTRA_MEMORY is True:
        INT_ALIGNED = True

    user_N = args.N
    try:
        err_assert(user_N != "", "No \"N\" flag")
        user_N = int(user_N)
    except ValueError:
        err_assert(False, "\"N\" flag not valid int type")

    user_T = args.type
    user_Size = args.size
    user_Signed = args.signed
    user_Unsigned = args.unsigned

    set_T = False
    if user_T != "":
        if Sort_Type(1, Sign.SIGNED).is_valid(user_T) is True:
            if user_Size != "" or user_Signed is not False or user_Unsigned is not False:
                print(
                    "Overriding \"signed\", \"unsigned\", and \"size\" flags with \"type\" flag"
                )
            set_T = True
            user_T = Sort_Type(1, Sign.SIGNED).string_to_T(user_T)

    if set_T is False:
        err_assert(user_Size != "",
                   "\"size\" flag is required if \"type\" is not specified")
        try:
            user_Size = int(user_Size)
        except ValueError:
            err_assert(False, "\"size\" flag is not valid integer type")

        if user_Signed is True:
            user_T = Sort_Type(user_Size, Sign.SIGNED)
        elif user_Unsigned is True:
            user_T = Sort_Type(user_Size, Sign.UNSIGNED)
        else:
            err_assert(False,
                       "neither \"signed\" nor \"unsigned\" flag specified")
    USER_TYPE = user_T

    user_Algorithm = args.algorithm
    err_assert(
        Algorithms(0).valid_algorithm(user_Algorithm) is True,
        "\"algorithm\" flag doesn't match")

    network_builder = Builder(user_N, user_T, user_Algorithm)

    if OUTFILE == "":
        print(network_builder.Build())
    else:
        mode = FMODE
        if mode == "":
            mode = "w+"
        if mode != "a" and mode != "w" and mode != "w+" and mode != "a+":
            err_assert(
                False,
                "\"fmode\": {} does not correspond to a valid file write mode flag"
            )

        try:
            f = open(OUTFILE, mode)
            f.write(network_builder.Build())
            f.flush()
            f.close()
        except IOError:
            err_assert(False,
                       "Error writing to \"outfile\": {}".format(OUTFILE))


######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
# Autogenerated Code
# Minimum Depth Known Sorting Networks For N = [4, 32]
# Taken from:
# http://users.telenet.be/bertdobbelaere/SorterHunter/sorting_networks.html
def min_pairs():
    minimum_pairs = [
        [],  # N = 0
        [],  # N = 1
        [],  # N = 2
        [],  # N = 3
        # Sorting Network For N = 4, with Depth = 3
        [0, 2, 1, 3, 0, 1, 2, 3, 1, 2],
        # Sorting Network For N = 5, with Depth = 5
        [0, 3, 1, 4, 0, 2, 1, 3, 0, 1, 2, 4, 1, 2, 3, 4, 2, 3],
        # Sorting Network For N = 6, with Depth = 5
        [
            0, 5, 1, 3, 2, 4, 1, 2, 3, 4, 0, 3, 2, 5, 0, 1, 2, 3, 4, 5, 1, 2,
            3, 4
        ],
        # Sorting Network For N = 7, with Depth = 6
        [
            0, 6, 2, 3, 4, 5, 0, 2, 1, 4, 3, 6, 0, 1, 2, 5, 3, 4, 1, 2, 4, 6,
            2, 3, 4, 5, 1, 2, 3, 4, 5, 6
        ],
        # Sorting Network For N = 8, with Depth = 6
        [
            0, 2, 1, 3, 4, 6, 5, 7, 0, 4, 1, 5, 2, 6, 3, 7, 0, 1, 2, 3, 4, 5,
            6, 7, 2, 4, 3, 5, 1, 4, 3, 6, 1, 2, 3, 4, 5, 6
        ],
        # Sorting Network For N = 9, with Depth = 7
        [
            0, 3, 1, 7, 2, 5, 4, 8, 0, 7, 2, 4, 3, 8, 5, 6, 0, 2, 1, 3, 4, 5,
            7, 8, 1, 4, 3, 6, 5, 7, 0, 1, 2, 4, 3, 5, 6, 8, 2, 3, 4, 5, 6, 7,
            1, 2, 3, 4, 5, 6
        ],
        # Sorting Network For N = 10, with Depth = 7
        [
            0, 1, 2, 5, 3, 6, 4, 7, 8, 9, 0, 6, 1, 8, 2, 4, 3, 9, 5, 7, 0, 2,
            1, 3, 4, 5, 6, 8, 7, 9, 0, 1, 2, 7, 3, 5, 4, 6, 8, 9, 1, 2, 3, 4,
            5, 6, 7, 8, 1, 3, 2, 4, 5, 7, 6, 8, 2, 3, 4, 5, 6, 7
        ],
        # Sorting Network For N = 11, with Depth = 8
        [
            0, 9, 1, 6, 2, 4, 3, 7, 5, 8, 0, 1, 3, 5, 4, 10, 6, 9, 7, 8, 1, 3,
            2, 5, 4, 7, 8, 10, 0, 4, 1, 2, 3, 7, 5, 9, 6, 8, 0, 1, 2, 6, 4, 5,
            7, 8, 9, 10, 2, 4, 3, 6, 5, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 2, 3,
            4, 5, 6, 7
        ],
        # Sorting Network For N = 12, with Depth = 8
        [
            0, 8, 1, 7, 2, 6, 3, 11, 4, 10, 5, 9, 0, 2, 1, 4, 3, 5, 6, 8, 7,
            10, 9, 11, 0, 1, 2, 9, 4, 7, 5, 6, 10, 11, 1, 3, 2, 7, 4, 9, 8, 10,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 5, 6, 8, 9, 10, 2,
            4, 3, 6, 5, 8, 7, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        ],
        # Sorting Network For N = 13, with Depth = 9
        [
            0, 11, 1, 7, 2, 4, 3, 5, 8, 9, 10, 12, 0, 2, 3, 6, 4, 12, 5, 7, 8,
            10, 0, 8, 1, 3, 2, 5, 4, 9, 6, 11, 7, 12, 0, 1, 2, 10, 3, 8, 4, 6,
            9, 11, 1, 3, 2, 4, 5, 10, 6, 8, 7, 9, 11, 12, 1, 2, 3, 4, 5, 8, 6,
            9, 7, 10, 2, 3, 4, 7, 5, 6, 8, 11, 9, 10, 4, 5, 6, 7, 8, 9, 10, 11,
            3, 4, 5, 6, 7, 8, 9, 10
        ],
        # Sorting Network For N = 14, with Depth = 9
        [
            0, 3, 1, 9, 2, 6, 4, 12, 5, 10, 7, 11, 8, 13, 0, 2, 3, 12, 4, 5, 6,
            10, 7, 8, 11, 13, 0, 1, 2, 11, 3, 6, 4, 7, 5, 9, 10, 12, 0, 4, 1,
            7, 2, 5, 3, 8, 6, 13, 9, 11, 1, 2, 3, 4, 5, 7, 6, 9, 8, 10, 12, 13,
            1, 3, 2, 4, 5, 9, 6, 10, 7, 8, 11, 12, 2, 3, 4, 5, 6, 7, 8, 11, 9,
            10, 12, 13, 4, 6, 5, 7, 8, 9, 10, 11, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            12
        ],
        # Sorting Network For N = 15, with Depth = 9
        [
            0, 6, 1, 10, 2, 14, 3, 9, 4, 12, 5, 13, 7, 11, 0, 7, 2, 5, 3, 4, 6,
            11, 8, 10, 9, 12, 13, 14, 1, 13, 2, 3, 4, 6, 5, 9, 7, 8, 10, 14,
            11, 12, 0, 3, 1, 4, 5, 7, 6, 13, 8, 9, 10, 11, 12, 14, 0, 2, 1, 5,
            3, 8, 4, 6, 7, 10, 9, 11, 12, 13, 0, 1, 2, 5, 3, 10, 4, 8, 6, 7, 9,
            12, 11, 13, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 12, 3, 5, 4, 6, 7,
            8, 9, 10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        ],
        # Sorting Network For N = 16, with Depth = 9
        [
            0, 5, 1, 4, 2, 12, 3, 13, 6, 7, 8, 9, 10, 15, 11, 14, 0, 2, 1, 10,
            3, 6, 4, 7, 5, 14, 8, 11, 9, 12, 13, 15, 0, 8, 1, 3, 2, 11, 4, 13,
            5, 9, 6, 10, 7, 15, 12, 14, 0, 1, 2, 4, 3, 8, 5, 6, 7, 12, 9, 10,
            11, 13, 14, 15, 1, 3, 2, 5, 4, 8, 6, 9, 7, 11, 10, 13, 12, 14, 1,
            2, 3, 5, 4, 11, 6, 8, 7, 9, 10, 12, 13, 14, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 4, 6, 5, 7, 8, 10, 9, 11, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12
        ],
        # Sorting Network For N = 17, with Depth = 10
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 3, 2, 4,
            5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16, 1, 5, 2, 6, 3, 7, 4, 8,
            9, 13, 10, 14, 11, 15, 12, 16, 0, 3, 1, 13, 2, 10, 4, 7, 5, 11, 6,
            12, 8, 9, 14, 15, 0, 13, 1, 8, 2, 5, 3, 6, 4, 14, 7, 15, 9, 16, 10,
            11, 0, 1, 2, 8, 3, 4, 5, 10, 6, 13, 7, 11, 12, 14, 1, 5, 3, 8, 4,
            10, 6, 7, 9, 12, 11, 13, 1, 2, 4, 6, 5, 8, 7, 10, 9, 11, 12, 14,
            13, 15, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 15, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ],
        # Sorting Network For N = 18, with Depth = 11
        [
            0, 6, 1, 10, 2, 15, 3, 5, 4, 9, 7, 16, 8, 13, 11, 17, 12, 14, 0,
            12, 1, 4, 3, 11, 5, 17, 6, 14, 7, 8, 9, 10, 13, 16, 1, 13, 2, 7, 4,
            16, 6, 9, 8, 11, 10, 15, 0, 1, 2, 3, 4, 12, 5, 13, 7, 9, 8, 10, 14,
            15, 16, 17, 0, 2, 1, 11, 3, 4, 5, 7, 6, 16, 10, 12, 13, 14, 15, 17,
            1, 8, 4, 10, 5, 6, 7, 13, 9, 16, 11, 12, 1, 3, 2, 5, 4, 7, 6, 8, 9,
            11, 10, 13, 12, 15, 14, 16, 1, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 13,
            12, 14, 15, 16, 2, 3, 5, 8, 6, 7, 9, 12, 10, 11, 14, 15, 3, 4, 5,
            6, 7, 8, 9, 10, 11, 12, 13, 14, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
        ],
        # Sorting Network For N = 19, with Depth = 11
        [
            0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 8, 10, 9, 11, 0,
            2, 1, 3, 4, 6, 5, 7, 8, 9, 10, 11, 12, 14, 13, 15, 16, 18, 0, 1, 2,
            3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 0, 4, 1, 12, 2, 16, 3, 17,
            5, 8, 6, 9, 7, 18, 10, 13, 11, 14, 1, 6, 3, 10, 4, 5, 7, 11, 8, 12,
            9, 16, 13, 18, 14, 15, 0, 4, 2, 8, 3, 9, 6, 7, 10, 16, 11, 17, 12,
            13, 15, 18, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 2, 3,
            4, 5, 6, 8, 7, 9, 10, 12, 11, 13, 14, 15, 16, 17, 2, 4, 3, 6, 5, 7,
            8, 10, 9, 11, 12, 14, 13, 16, 15, 17, 1, 2, 3, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 16, 17, 18, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16
        ],
        # Sorting Network For N = 20, with Depth = 11
        [
            0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 10, 9,
            11, 0, 2, 1, 3, 4, 6, 5, 7, 8, 9, 10, 11, 12, 14, 13, 15, 16, 18,
            17, 19, 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 0,
            4, 1, 12, 2, 16, 3, 17, 5, 8, 6, 9, 7, 18, 10, 13, 11, 14, 15, 19,
            1, 6, 3, 10, 4, 5, 7, 11, 8, 12, 9, 16, 13, 18, 14, 15, 0, 4, 2, 8,
            3, 9, 6, 7, 10, 16, 11, 17, 12, 13, 15, 19, 1, 4, 3, 6, 5, 8, 7,
            10, 9, 12, 11, 14, 13, 16, 15, 18, 2, 3, 4, 5, 6, 8, 7, 9, 10, 12,
            11, 13, 14, 15, 16, 17, 2, 4, 3, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13,
            16, 15, 17, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18,
            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ],
        # Sorting Network For N = 21, with Depth = 12
        [
            0, 7, 1, 10, 3, 5, 4, 8, 6, 13, 9, 19, 11, 14, 12, 17, 15, 16, 18,
            20, 0, 11, 1, 15, 2, 12, 3, 4, 5, 8, 6, 9, 7, 14, 10, 16, 13, 19,
            17, 20, 0, 6, 1, 3, 2, 18, 4, 15, 5, 10, 8, 16, 11, 17, 12, 13, 14,
            20, 2, 6, 5, 12, 7, 18, 8, 14, 9, 11, 10, 17, 13, 19, 16, 20, 1, 2,
            4, 7, 5, 9, 6, 17, 10, 13, 11, 12, 14, 19, 15, 18, 0, 2, 3, 6, 4,
            5, 7, 10, 8, 11, 9, 15, 12, 16, 13, 18, 14, 17, 19, 20, 0, 1, 2, 3,
            5, 9, 6, 12, 7, 8, 11, 14, 13, 15, 16, 19, 17, 18, 1, 2, 3, 9, 6,
            13, 10, 11, 12, 15, 16, 17, 18, 19, 1, 4, 2, 5, 3, 7, 6, 10, 8, 9,
            11, 12, 13, 14, 17, 18, 2, 4, 5, 6, 7, 8, 9, 11, 10, 13, 12, 15,
            14, 16, 3, 4, 5, 7, 6, 8, 9, 10, 11, 13, 12, 14, 15, 16, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
        ],
        # Sorting Network For N = 22, with Depth = 12
        [
            0, 14, 1, 8, 2, 4, 3, 5, 6, 11, 7, 21, 9, 12, 10, 15, 13, 20, 16,
            18, 17, 19, 0, 7, 1, 13, 2, 17, 3, 16, 4, 19, 5, 18, 6, 10, 8, 20,
            11, 15, 14, 21, 0, 1, 3, 6, 4, 9, 5, 10, 7, 13, 8, 14, 11, 16, 12,
            17, 15, 18, 20, 21, 0, 3, 1, 8, 2, 4, 7, 11, 9, 12, 10, 14, 13, 20,
            17, 19, 18, 21, 1, 6, 2, 7, 3, 17, 4, 18, 5, 11, 8, 9, 10, 16, 12,
            13, 14, 19, 15, 20, 0, 2, 3, 7, 4, 6, 5, 8, 9, 11, 10, 12, 13, 16,
            14, 18, 15, 17, 19, 21, 1, 4, 3, 5, 6, 13, 7, 9, 8, 15, 12, 14, 16,
            18, 17, 20, 1, 2, 4, 10, 6, 12, 7, 8, 9, 15, 11, 17, 13, 14, 19,
            20, 1, 3, 2, 5, 6, 10, 8, 9, 11, 15, 12, 13, 16, 19, 18, 20, 2, 3,
            4, 8, 5, 7, 6, 9, 10, 11, 12, 15, 13, 17, 14, 16, 18, 19, 4, 5, 6,
            7, 8, 10, 9, 12, 11, 13, 14, 15, 16, 17, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18
        ],
        # Sorting Network For N = 23, with Depth = 12
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15,
            16, 18, 17, 19, 20, 22, 0, 4, 1, 5, 2, 8, 3, 9, 6, 10, 7, 11, 12,
            16, 13, 17, 14, 20, 15, 21, 18, 22, 0, 2, 1, 3, 4, 6, 5, 7, 8, 10,
            9, 11, 12, 14, 13, 15, 16, 18, 17, 19, 20, 22, 0, 12, 1, 13, 2, 4,
            3, 5, 6, 8, 7, 9, 10, 22, 14, 16, 15, 17, 18, 20, 19, 21, 1, 12, 2,
            14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 11, 22, 1, 2,
            3, 14, 4, 6, 5, 7, 8, 13, 9, 20, 10, 15, 16, 18, 17, 19, 21, 22, 3,
            6, 5, 16, 7, 18, 8, 12, 9, 13, 10, 14, 11, 15, 17, 20, 2, 3, 4, 8,
            5, 12, 6, 10, 7, 14, 9, 16, 11, 18, 13, 17, 15, 19, 20, 21, 2, 4,
            5, 8, 7, 9, 10, 12, 11, 13, 14, 16, 15, 18, 19, 21, 3, 5, 6, 8, 7,
            10, 9, 12, 11, 14, 13, 16, 15, 17, 18, 20, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20
        ],
        # Sorting Network For N = 24, with Depth = 12
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14,
            13, 15, 16, 18, 17, 19, 20, 22, 21, 23, 0, 4, 1, 5, 2, 8, 3, 9, 6,
            10, 7, 11, 12, 16, 13, 17, 14, 20, 15, 21, 18, 22, 19, 23, 0, 2, 1,
            3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15, 16, 18, 17, 19, 20,
            22, 21, 23, 0, 12, 1, 13, 2, 4, 3, 5, 6, 8, 7, 9, 10, 22, 11, 23,
            14, 16, 15, 17, 18, 20, 19, 21, 1, 12, 2, 14, 3, 15, 4, 16, 5, 17,
            6, 18, 7, 19, 8, 20, 9, 21, 11, 22, 1, 2, 3, 14, 4, 6, 5, 7, 8, 13,
            9, 20, 10, 15, 16, 18, 17, 19, 21, 22, 3, 6, 5, 16, 7, 18, 8, 12,
            9, 13, 10, 14, 11, 15, 17, 20, 2, 3, 4, 8, 5, 12, 6, 10, 7, 14, 9,
            16, 11, 18, 13, 17, 15, 19, 20, 21, 2, 4, 5, 8, 7, 9, 10, 12, 11,
            13, 14, 16, 15, 18, 19, 21, 3, 5, 6, 8, 7, 10, 9, 12, 11, 14, 13,
            16, 15, 17, 18, 20, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20
        ],
        # Sorting Network For N = 25, with Depth = 13
        [
            0, 13, 1, 6, 2, 8, 3, 20, 4, 7, 5, 22, 9, 16, 10, 15, 11, 14, 17,
            23, 18, 21, 19, 24, 0, 3, 1, 19, 4, 18, 5, 12, 6, 24, 7, 21, 8, 16,
            9, 17, 10, 11, 13, 20, 14, 15, 0, 10, 1, 5, 2, 17, 3, 12, 6, 11, 7,
            16, 8, 23, 9, 18, 13, 22, 14, 19, 20, 24, 0, 1, 2, 9, 3, 14, 4, 8,
            5, 10, 6, 13, 7, 18, 11, 22, 12, 19, 15, 20, 16, 23, 17, 21, 1, 5,
            2, 4, 3, 6, 7, 9, 8, 17, 10, 15, 11, 14, 12, 13, 16, 18, 19, 22,
            20, 24, 21, 23, 0, 2, 1, 3, 4, 7, 5, 6, 8, 10, 9, 16, 11, 12, 13,
            14, 15, 17, 18, 21, 19, 20, 22, 24, 1, 2, 3, 4, 6, 18, 7, 19, 8,
            11, 9, 12, 10, 15, 13, 16, 14, 17, 21, 22, 23, 24, 1, 9, 2, 11, 4,
            6, 5, 7, 10, 12, 13, 15, 14, 23, 18, 20, 19, 21, 1, 3, 2, 8, 6, 14,
            7, 13, 9, 10, 11, 19, 12, 18, 15, 16, 17, 23, 20, 22, 2, 5, 4, 9,
            6, 10, 7, 8, 11, 13, 12, 14, 15, 19, 16, 21, 17, 18, 20, 23, 3, 5,
            4, 7, 6, 11, 8, 9, 10, 12, 13, 15, 14, 19, 16, 17, 18, 21, 22, 23,
            2, 3, 5, 7, 6, 8, 9, 11, 10, 13, 12, 15, 14, 16, 17, 19, 18, 20, 4,
            5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
        ],
        # Sorting Network For N = 26, with Depth = 13
        [
            0, 13, 1, 6, 2, 8, 3, 20, 4, 7, 5, 22, 9, 16, 10, 15, 11, 14, 12,
            25, 17, 23, 18, 21, 19, 24, 0, 3, 1, 19, 4, 18, 5, 12, 6, 24, 7,
            21, 8, 16, 9, 17, 10, 11, 13, 20, 14, 15, 22, 25, 0, 10, 1, 5, 2,
            17, 3, 12, 6, 11, 7, 16, 8, 23, 9, 18, 13, 22, 14, 19, 15, 25, 20,
            24, 0, 1, 2, 9, 3, 14, 4, 8, 5, 10, 6, 13, 7, 18, 11, 22, 12, 19,
            15, 20, 16, 23, 17, 21, 24, 25, 1, 5, 2, 4, 3, 6, 7, 9, 8, 17, 10,
            15, 11, 14, 12, 13, 16, 18, 19, 22, 20, 24, 21, 23, 0, 2, 1, 3, 4,
            7, 5, 6, 8, 10, 9, 16, 11, 12, 13, 14, 15, 17, 18, 21, 19, 20, 22,
            24, 23, 25, 1, 2, 3, 4, 6, 18, 7, 19, 8, 11, 9, 12, 10, 15, 13, 16,
            14, 17, 21, 22, 23, 24, 1, 9, 2, 11, 4, 6, 5, 7, 10, 12, 13, 15,
            14, 23, 16, 24, 18, 20, 19, 21, 1, 3, 2, 8, 6, 14, 7, 13, 9, 10,
            11, 19, 12, 18, 15, 16, 17, 23, 22, 24, 2, 5, 4, 9, 6, 10, 7, 8,
            11, 13, 12, 14, 15, 19, 16, 21, 17, 18, 20, 23, 3, 5, 4, 7, 6, 11,
            8, 9, 10, 12, 13, 15, 14, 19, 16, 17, 18, 21, 20, 22, 2, 3, 5, 7,
            6, 8, 9, 11, 10, 13, 12, 15, 14, 16, 17, 19, 18, 20, 22, 23, 4, 5,
            6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
        ],
        # Sorting Network For N = 27, with Depth = 14
        [
            0, 9, 1, 6, 2, 4, 3, 7, 5, 8, 11, 16, 12, 15, 13, 23, 14, 24, 17,
            18, 19, 20, 21, 26, 22, 25, 0, 1, 3, 5, 4, 10, 6, 9, 7, 8, 11, 13,
            12, 21, 14, 17, 15, 18, 16, 25, 19, 22, 20, 23, 24, 26, 1, 3, 2, 5,
            4, 7, 8, 10, 11, 19, 12, 14, 13, 22, 15, 24, 16, 20, 17, 21, 18,
            26, 23, 25, 0, 4, 1, 2, 3, 7, 5, 9, 6, 8, 11, 12, 13, 15, 14, 19,
            16, 17, 18, 23, 20, 21, 22, 24, 25, 26, 0, 1, 2, 6, 4, 5, 7, 8, 9,
            10, 12, 14, 13, 16, 15, 19, 17, 20, 18, 22, 21, 24, 23, 25, 0, 11,
            2, 4, 3, 6, 5, 7, 8, 9, 12, 13, 14, 16, 15, 22, 17, 19, 18, 20, 21,
            23, 24, 25, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 1, 12, 2, 3, 4, 5, 6, 7, 15, 17, 16, 18, 19, 21,
            20, 22, 2, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 3, 14, 4,
            15, 5, 16, 6, 17, 7, 18, 8, 19, 9, 20, 10, 21, 8, 11, 9, 12, 10,
            13, 14, 22, 15, 23, 16, 24, 17, 25, 18, 26, 4, 8, 5, 9, 6, 10, 7,
            14, 11, 15, 12, 16, 13, 17, 18, 22, 19, 23, 20, 24, 21, 25, 2, 4,
            3, 5, 6, 8, 7, 9, 10, 11, 12, 14, 13, 15, 16, 18, 17, 19, 20, 22,
            21, 23, 24, 26, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
        ],
        # Sorting Network For N = 28, with Depth = 14
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 0, 26, 1, 27, 2, 24, 3, 25, 4,
            22, 5, 23, 6, 20, 7, 21, 8, 18, 9, 19, 10, 16, 11, 17, 12, 14, 13,
            15, 0, 6, 1, 20, 2, 4, 3, 5, 7, 26, 10, 12, 11, 14, 13, 16, 15, 17,
            21, 27, 22, 24, 23, 25, 1, 18, 2, 10, 4, 22, 5, 23, 6, 8, 9, 26,
            12, 13, 14, 15, 17, 25, 19, 21, 0, 6, 3, 22, 4, 12, 5, 24, 7, 9,
            11, 13, 14, 16, 15, 23, 18, 20, 21, 27, 0, 2, 1, 4, 3, 11, 5, 13,
            6, 10, 7, 8, 14, 22, 16, 24, 17, 21, 19, 20, 23, 26, 25, 27, 1, 6,
            3, 7, 4, 11, 5, 10, 8, 15, 9, 14, 12, 19, 13, 18, 16, 23, 17, 22,
            20, 24, 21, 26, 2, 9, 4, 8, 5, 13, 7, 17, 10, 20, 11, 15, 12, 16,
            14, 22, 18, 25, 19, 23, 2, 3, 4, 5, 7, 12, 8, 11, 10, 13, 14, 17,
            15, 20, 16, 19, 22, 23, 24, 25, 1, 2, 5, 6, 8, 10, 9, 12, 11, 14,
            13, 16, 15, 18, 17, 19, 21, 22, 25, 26, 3, 5, 6, 9, 7, 8, 10, 12,
            11, 13, 14, 16, 15, 17, 18, 21, 19, 20, 22, 24, 2, 3, 4, 7, 5, 6,
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 21, 22, 24,
            25, 4, 5, 6, 7, 8, 10, 9, 11, 12, 14, 13, 15, 16, 18, 17, 19, 20,
            21, 22, 23, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24
        ],
        # Sorting Network For N = 29, with Depth = 14
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 0, 2, 1, 3, 4, 6, 5, 7, 8, 10,
            9, 11, 12, 14, 13, 15, 16, 18, 17, 19, 20, 22, 21, 23, 24, 26, 25,
            27, 0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15, 16, 20,
            17, 21, 18, 22, 19, 23, 24, 28, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5,
            13, 6, 14, 7, 15, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 0, 16, 1,
            8, 2, 4, 3, 12, 5, 10, 6, 9, 7, 14, 11, 13, 17, 24, 18, 20, 19, 28,
            21, 26, 22, 25, 23, 27, 1, 2, 3, 5, 4, 8, 6, 22, 7, 11, 9, 25, 10,
            12, 13, 14, 17, 18, 19, 21, 20, 24, 26, 28, 1, 17, 2, 18, 3, 19, 4,
            20, 5, 10, 7, 23, 8, 24, 11, 27, 12, 28, 13, 25, 21, 26, 3, 17, 4,
            16, 5, 21, 6, 18, 7, 9, 8, 20, 10, 26, 11, 23, 14, 28, 15, 27, 22,
            24, 1, 4, 3, 8, 5, 16, 7, 17, 9, 21, 10, 22, 11, 19, 12, 20, 14,
            24, 15, 26, 23, 28, 2, 5, 7, 8, 9, 18, 11, 17, 12, 16, 13, 22, 14,
            20, 15, 19, 23, 24, 2, 4, 6, 12, 9, 16, 10, 11, 13, 17, 14, 18, 15,
            22, 19, 25, 20, 21, 5, 6, 8, 12, 9, 10, 11, 13, 14, 16, 15, 17, 18,
            20, 19, 23, 21, 22, 25, 26, 3, 5, 6, 7, 8, 9, 10, 12, 11, 14, 13,
            16, 15, 18, 17, 20, 19, 21, 22, 23, 24, 25, 26, 28, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28
        ],
        # Sorting Network For N = 30, with Depth = 14
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 0, 2, 1, 3, 4, 6, 5, 7,
            8, 10, 9, 11, 12, 14, 13, 15, 16, 18, 17, 19, 20, 22, 21, 23, 24,
            26, 25, 27, 0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15,
            16, 20, 17, 21, 18, 22, 19, 23, 24, 28, 25, 29, 0, 8, 1, 9, 2, 10,
            3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 16, 24, 17, 25, 18, 26, 19, 27,
            20, 28, 21, 29, 0, 16, 1, 8, 2, 4, 3, 12, 5, 10, 6, 9, 7, 14, 11,
            13, 17, 24, 18, 20, 19, 28, 21, 26, 22, 25, 27, 29, 1, 2, 3, 5, 4,
            8, 6, 22, 7, 11, 9, 25, 10, 12, 13, 14, 17, 18, 19, 21, 20, 24, 23,
            27, 26, 28, 1, 17, 2, 18, 3, 19, 4, 20, 5, 10, 7, 23, 8, 24, 11,
            27, 12, 28, 13, 29, 21, 26, 3, 17, 4, 16, 5, 21, 6, 18, 7, 9, 8,
            20, 10, 26, 11, 23, 13, 25, 14, 28, 15, 27, 22, 24, 1, 4, 3, 8, 5,
            16, 7, 17, 9, 21, 10, 22, 11, 19, 12, 20, 14, 24, 15, 26, 23, 28,
            2, 5, 7, 8, 9, 18, 11, 17, 12, 16, 13, 22, 14, 20, 15, 19, 23, 24,
            26, 29, 2, 4, 6, 12, 9, 16, 10, 11, 13, 17, 14, 18, 15, 22, 19, 25,
            20, 21, 27, 29, 5, 6, 8, 12, 9, 10, 11, 13, 14, 16, 15, 17, 18, 20,
            19, 23, 21, 22, 25, 26, 3, 5, 6, 7, 8, 9, 10, 12, 11, 14, 13, 16,
            15, 18, 17, 20, 19, 21, 22, 23, 24, 25, 26, 28, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28
        ],
        # Sorting Network For N = 31, with Depth = 14
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 0, 2, 1, 3, 4, 6, 5, 7,
            8, 10, 9, 11, 12, 14, 13, 15, 16, 18, 17, 19, 20, 22, 21, 23, 24,
            26, 25, 27, 28, 30, 0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14,
            11, 15, 16, 20, 17, 21, 18, 22, 19, 23, 24, 28, 25, 29, 26, 30, 0,
            8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 16, 24, 17, 25,
            18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 0, 16, 1, 8, 2, 4, 3, 12,
            5, 10, 6, 9, 7, 14, 11, 13, 17, 24, 18, 20, 19, 28, 21, 26, 22, 25,
            23, 30, 27, 29, 1, 2, 3, 5, 4, 8, 6, 22, 7, 11, 9, 25, 10, 12, 13,
            14, 17, 18, 19, 21, 20, 24, 23, 27, 26, 28, 29, 30, 1, 17, 2, 18,
            3, 19, 4, 20, 5, 10, 7, 23, 8, 24, 11, 27, 12, 28, 13, 29, 14, 30,
            21, 26, 3, 17, 4, 16, 5, 21, 6, 18, 7, 9, 8, 20, 10, 26, 11, 23,
            13, 25, 14, 28, 15, 27, 22, 24, 1, 4, 3, 8, 5, 16, 7, 17, 9, 21,
            10, 22, 11, 19, 12, 20, 14, 24, 15, 26, 23, 28, 27, 30, 2, 5, 7, 8,
            9, 18, 11, 17, 12, 16, 13, 22, 14, 20, 15, 19, 23, 24, 26, 29, 2,
            4, 6, 12, 9, 16, 10, 11, 13, 17, 14, 18, 15, 22, 19, 25, 20, 21,
            27, 29, 5, 6, 8, 12, 9, 10, 11, 13, 14, 16, 15, 17, 18, 20, 19, 23,
            21, 22, 25, 26, 3, 5, 6, 7, 8, 9, 10, 12, 11, 14, 13, 16, 15, 18,
            17, 20, 19, 21, 22, 23, 24, 25, 26, 28, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28
        ],
        # Sorting Network For N = 32, with Depth = 14
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 2, 1, 3, 4,
            6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15, 16, 18, 17, 19, 20, 22, 21,
            23, 24, 26, 25, 27, 28, 30, 29, 31, 0, 4, 1, 5, 2, 6, 3, 7, 8, 12,
            9, 13, 10, 14, 11, 15, 16, 20, 17, 21, 18, 22, 19, 23, 24, 28, 25,
            29, 26, 30, 27, 31, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14,
            7, 15, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 23,
            31, 0, 16, 1, 8, 2, 4, 3, 12, 5, 10, 6, 9, 7, 14, 11, 13, 15, 31,
            17, 24, 18, 20, 19, 28, 21, 26, 22, 25, 23, 30, 27, 29, 1, 2, 3, 5,
            4, 8, 6, 22, 7, 11, 9, 25, 10, 12, 13, 14, 17, 18, 19, 21, 20, 24,
            23, 27, 26, 28, 29, 30, 1, 17, 2, 18, 3, 19, 4, 20, 5, 10, 7, 23,
            8, 24, 11, 27, 12, 28, 13, 29, 14, 30, 21, 26, 3, 17, 4, 16, 5, 21,
            6, 18, 7, 9, 8, 20, 10, 26, 11, 23, 13, 25, 14, 28, 15, 27, 22, 24,
            1, 4, 3, 8, 5, 16, 7, 17, 9, 21, 10, 22, 11, 19, 12, 20, 14, 24,
            15, 26, 23, 28, 27, 30, 2, 5, 7, 8, 9, 18, 11, 17, 12, 16, 13, 22,
            14, 20, 15, 19, 23, 24, 26, 29, 2, 4, 6, 12, 9, 16, 10, 11, 13, 17,
            14, 18, 15, 22, 19, 25, 20, 21, 27, 29, 5, 6, 8, 12, 9, 10, 11, 13,
            14, 16, 15, 17, 18, 20, 19, 23, 21, 22, 25, 26, 3, 5, 6, 7, 8, 9,
            10, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 21, 22, 23, 24, 25, 26,
            28, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28
        ]
    ]
    return minimum_pairs


######################################################################
if __name__ == "__main__":
    main()
