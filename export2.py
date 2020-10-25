#! /usr/bin/env python3

import cpufeature
import copy
from enum import Enum
import traceback
import sys


def err_assert(check, msg):
    if check is False:
        print("Error: " + msg)
        traceback.print_stack()
        exit(-1)


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


SIMD_RESTRICTIONS = "AVX512"
ALIGNED_ACCESS = False
EXTRA_MEMORY = False

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
    if sort_bytes == SIMD_m64().sizeof():
        return SIMD_m64()
    elif sort_bytes == SIMD_m128().sizeof():
        return SIMD_m128()
    elif sort_bytes == SIMD_m256().sizeof():
        return SIMD_m256()
    elif sort_bytes == SIMD_m512().sizeof():
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

    def min_value(self):
        cast = "None"
        val = 0
        if self.sign == Sign.SIGNED:
            cast = self.signed_casts[self.size]
            val = (int(1) << (self.sizeof_bits() - 1))
        elif self.sign == Sign.UNSIGNED:
            cast = self.unsigned_casts[self.size]
            val = 0

        err_assert(cast != "None", "no cast for type")
        return "{}({})".format(cast, str(hex(val)))

    def max_value(self):
        cast = "None"
        val = 0
        if self.sign == Sign.SIGNED:
            cast = self.signed_casts[self.size]
            val = (int(1) << (self.sizeof_bits() - 1)) - 1
        elif self.sign == Sign.UNSIGNED:
            cast = self.unsigned_casts[self.size]
            val = (int(1) << (self.sizeof_bits())) - 1

        err_assert(cast != "None", "no cast for type")
        return "{}({})".format(cast, str(hex(val)))


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
        instruction = "__m64 [TMP0] = _mm_cmpgt_pi8([V2], [V1]);"
        instruction += "\n"
        instruction += "_mm_or_si64(_mm_and_si64([TMP0], [V1]), _mm_andnot_si64([TMP0], [V2]))"
        return instruction


class SIMD_Min_Fallback_m64_u16(SIMD_Instruction):
    def __init__(self, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.UNSIGNED, 2, SIMD_m64(), ["MMX"], weight)

    def generate_instruction(self):
        instruction = "__m64 [TMP0] = _mm_set1_pi16(1 << 15);"
        instruction += "\n"
        instruction += "__m64 [TMP1] = _mm_cmpgt_pi16(_mm_xor_si64([V1], [TMP0]), _mm_xor_si64([V2], [TMP0]));"
        instruction += "\n"
        instruction += "_mm_or_si64(_mm_and_si64([TMP1], [V2]), _mm_andnot_si64([TMP1], [V1]))"
        return instruction


class SIMD_Min_Fallback_s64(SIMD_Instruction):
    def __init__(self, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.SIGNED, 8, simd_type, constraints, weight)

    def generate_instruction(self):
        instruction = "{} [TMP0] = {}_cmpgt_epi64([V1], [V2]);".format(
            self.simd_type.to_string(), self.simd_type.prefix())
        instruction += "\n"
        instruction += "{}_blendv_epi8([V1], [V2], [TMP0])".format(
            self.simd_type.prefix())
        return instruction


class SIMD_Min_Fallback_u64(SIMD_Instruction):
    def __init__(self, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.UNSIGNED, 8, simd_type, constraints, weight)

    def generate_instruction(self):
        instruction = "{} [TMP0] = {}_set1_epi64x(1UL) << 63);\n".format(
            self.simd_type.to_string(), self.simd_type.prefix())
        instruction += "{} [TMP1] = {}_cmpgt_epi64({}_xor_{}([V1], [TMP0]), {}_xor_{}([V2], [TMP0]));\n".format(
            self.simd_type.to_string(), self.simd_type.prefix(),
            self.simd_type.prefix(), self.simd_type.postfix(),
            self.simd_type.prefix(), self.simd_type.postfix())
        instruction += "{}_blendv_epi8([V1], [V2], [TMP1])".format(
            self.simd_type.prefix())
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
                         Sign.SIGNED, 1, SIMD_m64, ["MMX"], weight)

    def generate_instruction(self):
        instruction = "__m64 [TMP0] = _mm_cmpgt_pi8([V1], [V2]);"
        instruction += "\n"
        instruction += "_mm_or_si64(_mm_and_si64([TMP0], [V1]), _mm_andnot_si64([TMP0], [V2]))"
        return instruction


class SIMD_Max_Fallback_m64_u16(SIMD_Instruction):
    def __init__(self, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.UNSIGNED, 2, SIMD_m64, ["MMX"], weight)

    def generate_instruction(self):
        instruction = "__m64 [TMP0] = _mm_set1_pi16(1 << 15);"
        instruction += "\n"
        instruction += "__m64 [TMP1] = _mm_cmpgt_pi16(_mm_xor_si64([V1], [TMP0]), _mm_xor_si64([V2], [TMP0]));"
        instruction += "\n"
        instruction += "_mm_or_si64(_mm_and_si64([TMP1], [V1]), _mm_andnot_si64([TMP1], [V2]))"
        return instruction


class SIMD_Max_Fallback_s64(SIMD_Instruction):
    def __init__(self, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.SIGNED, 8, simd_type, constraints, weight)

    def generate_instruction(self):
        instruction = "{} [TMP0] = {}_cmpgt_epi64([V1], [V2]);".format(
            self.simd_type.to_string(), self.simd_type.prefix())
        instruction += "\n"
        instruction += "{}_blendv_epi8([V2], [V1], [TMP0])".format(
            self.simd_type.prefix())
        return instruction


class SIMD_Max_Fallback_u64(SIMD_Instruction):
    def __init__(self, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.UNSIGNED, 8, simd_type, constraints, weight)

    def generate_instruction(self):
        instruction = "{} [TMP0] = {}_set1_epi64x(1UL) << 63);\n".format(
            self.simd_type.to_string(), self.simd_type.prefix())
        instruction += "{} [TMP1] = {}_cmpgt_epi64({}_xor_{}([V1], [TMP0]), {}_xor_{}([V2], [TMP0]));\n".format(
            self.simd_type.to_string(), self.simd_type.prefix(),
            self.simd_type.postfix(), self.simd_type.prefix(),
            self.simd_type.postfix())
        instruction += "{}_blendv_epi8([V2], [V1], [TMP1])".format(
            self.simd_type.prefix())
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
        if self.T_target == 4:
            return blend_mask != (int(-1))
        if self.T_target == 2:
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
            return self.blend_mask_lt_T()
        else:
            return self.blend_mask_ge_T()

    def blend_mask_ge_T(self):
        N = len(self.perm)
        T_size = self.T_size
        T_target = self.T_target

        scale = int(T_size / T_target)
        scaled_mask = (int(1) << scale) - 1
        err_assert(scale == 1 or scale == 2 or scale == 4, "invalid scale")

        blend_mask = int(0)
        for i in range(0, N):
            if self.perm[(N - 1) - i] > i:
                blend_mask |= scaled_mask << (scale * i)

        return blend_mask

    def blend_mask_lt_T(self):
        N = len(self.perm)
        T_size = self.T_size
        T_target = self.T_target

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
            if self.perm[(N - 1) - i] > i:
                run += 1
            else:
                run -= 1
            if run_idx == scale:

                # run can only be scale or -scale if
                # all epi8/epi16 mapped the same way
                if run == scale:
                    blend_mask |= (int(1) << (int(i / 4)))
                elif run == ((-1) * scale):
                    # do nothing
                    blend_mask = blend_mask
                else:
                    return int(-1)
                run = 0
                run_idx = 0

        return blend_mask

    def generate_instruction(self):
        return self.iname.replace("[BLEND_MASK]", str(hex(self.blend_mask())))


class SIMD_Blend_As_Epi8_Generator(SIMD_Instruction):
    def __init__(self, T_size, perm, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)

        self.perm = copy.deepcopy(perm)

    def blend_vec(self):
        N = self.simd_type.sizeof()
        T_size = self.T_size

        err_assert(T_size * len(self.perm) == N, "Invalid permutation map")
        err_assert(T_size * N != 8, "Cant use blend_epi8 for __m64")

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

        return vec

    def generate_instruction(self):
        vec = self.blend_vec()
        list_vec = arr_to_csv(vec, True)

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
        instruction = "{} [TMP0] = (__m64)[BLEND_MASK];".format(
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
            SIMD_Blend_Generator("_mm_blend_epi16([V1], [V2], [BLEND_MASK])",
                                 1, 2, perm, SIMD_m128(), ["SSE4.1"], 1),
            SIMD_Blend_Generator("_mm_mask_mov_epi8([V1], [BLEND_MASK], [V2])",
                                 1, 1, perm, SIMD_m128(),
                                 ["AVX512vl", "AVX512bw"], 2),
            SIMD_Blend_As_Epi8_Generator(1, perm, SIMD_m128(), ["SSE4.1"], 3),
            # __m128i epi16 ordering
            SIMD_Blend_Generator("_mm_blend_epi32([V1], [V2], [BLEND_MASK])",
                                 2, 4, perm, SIMD_m128(), ["AVX2"], 0),
            SIMD_Blend_Generator("_mm_blend_epi16([V1], [V2], [BLEND_MASK])",
                                 2, 2, perm, SIMD_m128(), ["SSE4.1"], 1),
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
            SIMD_Blend_Generator(
                "_mm256_blend_epi16([V1], [V2], [BLEND_MASK])", 1, 2, perm,
                SIMD_m256(), ["AVX2"], 1),
            SIMD_Blend_Generator(
                "_mm256_mask_mov_epi8([V1], [BLEND_MASK], [V2])", 1, 1, perm,
                SIMD_m256(), ["AVX512vl", "AVX512bw"], 2),
            SIMD_Blend_As_Epi8_Generator(1, perm, SIMD_m256(), ["AVX2"], 3),
            # __m256i epi16 ordering
            SIMD_Blend_Generator(
                "_mm256_blend_epi32([V1], [V2], [BLEND_MASK])", 2, 4, perm,
                SIMD_m256(), ["AVX2"], 0),
            SIMD_Blend_Generator(
                "_mm256_blend_epi16([V1], [V2], [BLEND_MASK])", 2, 2, perm,
                SIMD_m256(), ["AVX2"], 1),
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
                "_mm256_blend_epi64([V1], [V2], [BLEND_MASK])", 8, 4, perm,
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
        lower_bound = N - ((i + 1) * ele_per_lane + offset)
        upper_bound = N - (i * ele_per_lane + offset)
        for j in range(0, ele_per_lane):
            p = perm[i * ele_per_lane + j + offset]
            err_assert(
                p >= lower_bound and p < upper_bound,
                "trying to build epi32 shuffle mask across lanes. This should have been checked for earlier"
            )
            idx = p - lower_bound
            slot = slot_bits * ((ele_per_lane - 1) - j)

            err_assert((lane_mask & (int(idx) << slot)) == 0,
                       "overlapping indices in shuffle_mask")
            lane_mask |= int(idx) << slot

        # checking for consistent pattern across lanes
        if i != 0 and mask != lane_mask:
            return int(-1)
        mask = lane_mask

    return mask


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
        err_assert(self.simd_type.sizeof() == 32,
                   "Shuffle_As_Epi64 only valid for __m256i")

        T_size = self.T_size
        if can_shrink(T_size, 8, self.perm) is False:
            return int(-1)

        new_perm = scale_perm(T_size, 8, self.perm)
        err_assert(len(new_perm) == 4, "Any other value should be impossible")

        mask = int(0)
        for i in range(0, len(new_perm)):
            p = new_perm[i]
            err_assert((mask & (int(p) << (2 * i))) == 0,
                       "overlapping indices")
            mask |= int(p) << (2 * i)
        return mask

    def generate_instruction(self):
        return "_mm256_permute4x64_epi64([V], [SHUFFLE_MASK])".replace(
            "[SHUFFLE_MASK]", str(hex(self.shuffle_mask())))


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
        return "{}_shuffle_epi32([V], [SHUFFLE_MASK])".format(
            self.simd_type.prefix()).replace("[SHUFFLE_MASK]",
                                             str(hex(self.shuffle_mask())))


class SIMD_Shuffle_As_Epi16(SIMD_Instruction):
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
        if shuffle_mask_lo == int(0) or shuffle_mask_hi == int(0):
            return int(-1)
        return shuffle_mask_lo | (shuffle_mask_hi << 32)

    def generate_instruction(self):
        mask = self.shuffle_mask()
        shuffle_mask_lo = mask & 0xffffffff
        shuffle_mask_hi = (mask >> 32) & 0xffffffff
        return "{}_shufflehi_epi16({}_shufflelo_epi16([V], [SHUFFLE_MASK_LO]), [SHUFFLE_MASK_HI])".format(
            self.simd_type.prefix(), self.simd_type.prefix()).replace(
                "[SHUFFLE_MASK_LO]",
                str(hex(shuffle_mask_lo))).replace("[SHUFFLE_MASK_HI]",
                                                   str(hex(shuffle_mask_hi)))


class SIMD_Shuffle_As_Epi8(SIMD_Instruction):
    def __init__(self, T_size, perm, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, T_size, simd_type, constraints,
                         weight)
        self.perm = copy.deepcopy(perm)

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and in_same_lanes(16, 1, self.perm)

    def generate_instruction(self):
        scaled_perm = scale_perm(self.T_size, 1, self.perm)
        list_perm = arr_to_csv(scaled_perm)

        instruction = "{}_shuffle_epi8([V], {}_set_epi8([SHUFFLE_VEC]))".format(
            self.simd_type.prefix(),
            self.simd_type.prefix()).replace("[SHUFFLE_VEC]", list_perm)

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
                        vec.append(p)
                else:
                    if cross_lane is True:
                        vec.append(p)
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

        instruction = "__m256i [TMP0] = _mm256_permute4x64_epi64(v, 0x4e);"
        instruction += "\n"
        instruction += "__m256i [TMP1] = _mm256_shuffle_epi8([V], _mm256_set_epi8([SAME_LANE_VEC]));".replace(
            "[SAME_LANE_VEC]", same_lane_list_vec)
        instruction += "\n"
        instruction += "__m256i [TMP2] = _mm256_shuffle_epi8([TMP0], _mm256_set_epi8([OTHER_LANE_VEC]));".replace(
            "[OTHER_LANE_VEC]", other_lane_list_vec)
        instruction += "\n"
        instruction += "_mm256_or_si256([TMP1], [TMP2])"
        return instruction


class SIMD_Permutex_Generate(SIMD_Instruction):
    def __init__(self, iname, T_size, perm, simd_type, constraints, weight):
        super().__init__(iname, Sign.NOT_SIGNED, T_size, simd_type,
                         constraints, weight)
        self.perm = copy.deepcopy(perm)

    def generate_instruction(self):
        return self.iname.replace("[PERM_LIST]", arr_to_csv(self.perm))


class SIMD_Permute():
    def __init__(self, perm):
        self.instructions = [
            ###################################################################
            SIMD_Permutex_Generate(
                "_mm_shuffle_epi8([V], _mm_set_pi8([PERM_LIST]))", 1, perm,
                SIMD_m64(), ["MMX", "SSSE3"], 0),
            SIMD_Shuffle_As_Epi16(2, perm, SIMD_m64(), ["SSE"], 0),
            ###################################################################
            # __m128i epi8 ordering
            SIMD_Shuffle_As_Epi32(1, perm, SIMD_m128(), ["SSE2"], 0),
            SIMD_Shuffle_As_Epi16(1, perm, SIMD_m128(), ["SSE2"], 1),
            SIMD_Shuffle_As_Epi8(1, perm, SIMD_m128(), ["SSSE3"], 2),
            # __m128i epi16 ordering
            SIMD_Shuffle_As_Epi32(2, perm, SIMD_m128(), ["SSE2"], 0),
            SIMD_Shuffle_As_Epi16(2, perm, SIMD_m128(), ["SSE2"], 1),
            SIMD_Shuffle_As_Epi8(2, perm, SIMD_m128(), ["SSSE3"], 2),
            # __m128i epi32 ordering
            SIMD_Shuffle_As_Epi32(4, perm, SIMD_m128(), ["SSE2"], 0),
            # __m128i epi32 ordering
            SIMD_Shuffle_As_Epi32(8, perm, SIMD_m128(), ["SSE2"], 0),
            ###################################################################
            # __m256i epi8 ordering
            SIMD_Shuffle_As_Epi32(1, perm, SIMD_m256(), ["AVX2"], 0),
            SIMD_Shuffle_As_Epi64(1, perm, SIMD_m256(), ["AVX2"], 1),
            SIMD_Shuffle_As_Epi16(1, perm, SIMD_m256(), ["AVX2"], 2),
            SIMD_Shuffle_As_Epi8(1, perm, SIMD_m256(), ["AVX2"], 3),
            SIMD_Permutex_Generate(
                "_mm256_permutexvar_epi8([V], _mm256_set_epi8([PERM_LIST]))",
                1, perm, SIMD_m256(), ["AVX512vbmi", "AVX512vl", "AVX"], 4),
            SIMD_Permutex_Fallback(1, perm, SIMD_m256(), ["AVX2", "AVX"], 5),
            # __m256i epi16 ordering
            SIMD_Shuffle_As_Epi32(2, perm, SIMD_m256(), ["AVX2"], 0),
            SIMD_Shuffle_As_Epi64(1, perm, SIMD_m256(), ["AVX2"], 1),
            SIMD_Shuffle_As_Epi16(2, perm, SIMD_m256(), ["AVX2"], 2),
            SIMD_Shuffle_As_Epi8(2, perm, SIMD_m256(), ["AVX2"], 4),
            SIMD_Permutex_Generate(
                "_mm256_permutexvar_epi16([V], _mm256_set_epi16([PERM_LIST]))",
                2, perm, SIMD_m256(), ["AVX512bw", "AVX512vl", "AVX"], 4),
            SIMD_Permutex_Fallback(2, perm, SIMD_m256(), ["AVX2", "AVX"], 5),
            # __m256i epi32 ordering
            SIMD_Shuffle_As_Epi32(4, perm, SIMD_m256(), ["AVX2"], 0),
            SIMD_Shuffle_As_Epi64(1, perm, SIMD_m256(), ["AVX2"], 1),
            SIMD_Shuffle_As_Epi8(4, perm, SIMD_m256(), ["AVX2"], 2),
            SIMD_Permutex_Generate(
                "_mm256_permutevar8x32_epi32([V], _mm256_set_epi32([PERM_LIST]))",
                4, perm, SIMD_m256(), ["AVX2", "AVX"], 3),
            # __m256i epi64 ordering
            SIMD_Shuffle_As_Epi32(8, perm, SIMD_m256(), ["AVX2"], 0),
            SIMD_Shuffle_As_Epi64(8, perm, SIMD_m256(), ["AVX2"], 1),
            ###################################################################
            # __m512i epi8 ordering
            SIMD_Shuffle_As_Epi32(1, perm, SIMD_m512(), ["AVX512f"], 0),
            SIMD_Shuffle_As_Epi16(1, perm, SIMD_m512(), ["AVX512bw"], 1),
            SIMD_Shuffle_As_Epi8(1, perm, SIMD_m512(), ["AVX512bw"], 2),
            SIMD_Permutex_Generate(
                "_mm512_permutexvar_epi8([V], _mm512_set_epi8([PERM_LIST]))",
                1, perm, SIMD_m512(), ["AVX512vbmi", "AVX512f"], 3),
            # __m512i epi16 ordering
            SIMD_Shuffle_As_Epi32(2, perm, SIMD_m512(), ["AVX512f"], 0),
            SIMD_Shuffle_As_Epi16(2, perm, SIMD_m512(), ["AVX512bw"], 1),
            SIMD_Shuffle_As_Epi8(2, perm, SIMD_m512(), ["AVX512bw"], 2),
            SIMD_Permutex_Generate(
                "_mm512_permutexvar_epi16([V], _mm512_set_epi16([PERM_LIST]))",
                2, perm, SIMD_m512(), ["AVX512bw", "AVX512f"], 3),
            # __m512i epi32 ordering
            SIMD_Shuffle_As_Epi32(4, perm, SIMD_m512(), ["AVX512f"], 0),
            SIMD_Shuffle_As_Epi8(4, perm, SIMD_m512(), ["AVX512bw"], 1),
            SIMD_Permutex_Generate(
                "_mm512_permutexvar_epi32([V], _mm512_set_epi32([PERM_LIST]))",
                4, perm, SIMD_m512(), ["AVX512f"], 2),
            # __m512i epi64 ordering
            SIMD_Shuffle_As_Epi32(8, perm, SIMD_m512(), ["AVX512f"], 0),
            SIMD_Shuffle_As_Epi8(8, perm, SIMD_m512(), ["AVX512bw"], 1),
            SIMD_Permutex_Generate(
                "_mm512_permutexvar_epi64([V], _mm512_set_epi64([PERM_LIST]))",
                8, perm, SIMD_m512(), ["AVX512f"], 2)
        ]


######################################################################
# Load

# A few cases. First aligned vs unaligned. This doesn't make a big
# deal anymore but why not. Second is incomplete loads i.e sorting
# int32s with N = 12 requires an __m256i register but an array of 12
# int32s isnt guranteed to have enough memory to fully load a __m256i
# register so some logic to accomidate that is necessary
######################################################################


class SIMD_Full_Load(SIMD_Instruction):
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


class SIMD_Partial_Load_m64(SIMD_Instruction):
    def __init__(self, iname, sort_type, raw_N, simd_type, constraints,
                 weight):
        super().__init__(iname, Sign.NOT_SIGNED, sort_type.sizeof(), simd_type,
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
        instruction += "memcpy(&[TMP0], [ARR], [SORT_BYTES]);".replace(
            "[SORT_BYTES]", str(self.raw_N * self.sort_type.sizeof()))
        instruction += "\n"
        instruction += "[TMP0]"
        return instruction


class SIMD_Mask_Load(SIMD_Instruction):
    def __init__(self, iname, sort_type, T_size, aligned, raw_N, simd_type,
                 constraints, weight):
        super().__init__(iname, Sign.NOT_SIGNED, T_size, simd_type,
                         constraints, weight)
        self.raw_N = raw_N
        self.sort_type = sort_type
        
        # Booleans
        self.aligned = aligned

    def match(self, match_info):
        return self.has_support() and self.match_sort_type(
            match_info.sort_type) and self.match_simd_type(
                match_info.simd_type) and ((match_info.aligned is self.aligned)
                                           or (self.aligned is False))

    def generate_instruction(self):
        return self.iname.replace("[MAX]", self.sort_type.max_value()).replace(
            "[LOAD_MASK]", str(hex((int(1) << self.raw_N) - 1)))


class SIMD_Mask_Load_Fallback_As_Epi32(SIMD_Instruction):
    def __init__(self, sort_type, raw_N, simd_type, constraints, weight):
        super().__init__("Override Error: " + self.__class__.__name__,
                         Sign.NOT_SIGNED, sort_type.sizeof(), simd_type,
                         constraints, weight)

        self.raw_N = raw_N
        self.sort_type = sort_type

    def build_baseline_vec(self, MIN_VAL):
        T_size = self.sort_type.sizeof()
        err_assert(T_size < 4, "building fallback unnecissarily")

        SIMD_size = self.simd_type.sizeof()

        baseline_vec = []
        for i in range(0, SIMD_size):
            if i < (SIMD_size - self.raw_N):
                if MIN_VAL is True:
                    baseline_vec.append(self.sort_type.min_value())
                else:
                    baseline_vec.append(self.sort_type.max_value())
            else:
                baseline_vec.append("0")

        return baseline_vec

    def build_load_bool_vec(self):
        if self.sort_type.sizeof() < 4:
            return self.build_load_bool_vec_lt()
        else:
            return self.build_load_bool_vec_ge()

    def build_load_bool_vec_ge(self):
        T_size = self.T_size
        err_assert(T_size >= 4, "mask_load requires sizeof(T) >= 4")

        SIMD_size = self.simd_type.sizeof()
        err_assert(SIMD_size > 8, "mask_load not supported by __m64")

        ele_per_register = int(SIMD_size / T_size)
        scale = int(T_size / 4)

        set_arr = []
        for i in range(0, ele_per_register):
            for j in range(0, scale):
                if self.raw_N >= (i + 1):
                    set_arr.append(int(1) << (31))
                else:
                    set_arr.append(0)

        set_arr_ret = []
        for i in range(0, len(set_arr)):
            set_arr_ret.append(set_arr[len(set_arr) - (i + 1)])
            
        return set_arr_ret

    def build_load_bool_vec_lt(self):
        T_size = self.sort_type.sizeof()
        err_assert(T_size < 4, "building fallback unnecissarily")

        SIMD_size = self.simd_type.sizeof()
        err_assert(SIMD_size > 8, "mask_load not supported by __m64")

        ele_per_int32 = int(4 / T_size)
        int32_per_register = int(SIMD_size / 4)

        set_arr = []
        for i in range(0, int32_per_register):
            if self.raw_N >= (i * ele_per_int32):
                set_arr.append(int(1) << 31)
            else:
                set_arr.append(0)

        err_assert(int32_per_register == len(set_arr), "didn't set all values")
        set_arr_ret = []
        for i in range(0, len(set_arr)):
            set_arr_ret.append(set_arr[len(set_arr) - (i + 1)])
        
        return set_arr_ret

    def generate_instruction(self):
        if self.sort_type.sizeof() < 4:
            return self.generate_instruction_lt()
        else:
            return self.generate_instruction_ge()

    def generate_instruction_ge(self):
        load_bool_vec = self.build_load_bool_vec()
        list_load_bool_vec = arr_to_csv(load_bool_vec, True)

        instruction = "{}_maskload_epi32({}_set_epi_32([LOAD_BOOLS]))".format(
            self.simd_type.prefix(),
            self.simd_type.prefix()).replace("[LOAD_BOOLS]",
                                             list_load_bool_vec)
        return instruction

    def generate_instruction_lt(self):
        baseline_max_vec = self.build_baseline_vec(False)
        list_baseline_max_vec = arr_to_csv(baseline_max_vec, False)

        baseline_min_vec = self.build_baseline_vec(True)
        list_baseline_min_vec = arr_to_csv(baseline_min_vec, False)

        load_bool_vec = self.build_load_bool_vec()
        list_load_bool_vec = arr_to_csv(load_bool_vec, True)

        instruction = "{} [TMP0] = {}_set_epi{}([BASELINE_MAX_VEC]);".format(
            self.simd_type.to_string(), self.simd_type.prefix(),
            self.sort_type.sizeof_bits()).replace("[BASELINE_MAX_VEC]",
                                                  list_baseline_max_vec)
        instruction += "\n"

        if self.sort_type.sign == Sign.SIGNED:
            instruction += "{} [TMP2] = {}_set_epi{}([BASELINE_MIN_VEC]);".format(
                self.simd_type.to_string(), self.simd_type.prefix(),
                self.sort_type.sizeof_bits()).replace("[BASELINE_MIN_VEC]",
                                                      list_baseline_min_vec)

        instruction += "{} [TMP1] = {}_maskload_epi32((int32_t * const)[ARR], {}_set_epi32([LOAD_BOOLS]));".format(
            self.simd_type.to_string(), self.simd_type.prefix(),
            self.simd_type.prefix()).replace("[LOAD_BOOLS]",
                                             list_load_bool_vec)
        instruction += "\n"
        if self.sort_type.sign == Sign.SIGNED:
            instruction += "[TMP1] = {}_or_{}([TMP1], [TMP0]);".format(
                self.simd_type.prefix(), self.simd_type.postfix())
            instruction += "\n"
            instruction += "{}_xor_{}([TMP1], [TMP2])".format(
                self.simd_type.prefix(), self.simd_type.postfix())

        elif self.sort_type.sign == Sign.UNSIGNED:
            instruction += "{}_or_{}([TMP1], [TMP0])".format(
                self.simd_type.prefix(), self.simd_type.postfix())

        else:
            err_assert(False, "Invalid sign for sort_type")

        return instruction


class SIMD_Load():
    def __init__(self, raw_N, sort_type):
        self.instructions = [
            ###################################################################
            SIMD_Full_Load("(*((_aliasing_m64_ *)[ARR]))", True, SIMD_m64(),
                           ["MMX"], 0),
            SIMD_Partial_Load_m64("(*((_aliasing_m64_ *)[ARR]))", sort_type,
                                  raw_N, SIMD_m64(), ["MMX"], 1),
            SIMD_Full_Load("(*((_aliasing_m64_ *)[ARR]))", True, SIMD_m64(),
                           ["MMX"], 0),
            SIMD_Partial_Load_m64("(*((_aliasing_m64_ *)[ARR]))", sort_type,
                                  raw_N, SIMD_m64(), ["MMX"], 1),
            ###################################################################
            # These are universal best instructions if applicable
            SIMD_Full_Load("_mm_load_si128((__m128i *)[ARR])", True,
                           SIMD_m128(), ["SSE2"], 0),
            SIMD_Full_Load("_mm_loadu_si128((__m128i *)[ARR])", False,
                           SIMD_m128(), ["SSE2"], 1),

            # This is universal fallback and always lowest priority
            SIMD_Mask_Load_Fallback_As_Epi32(sort_type, raw_N, SIMD_m128(),
                                             ["AVX2", "SSE2"], 100),
            SIMD_Mask_Load(
                "_mm_mask_loadu_epi8(_mm_set1_epi8([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 1, False, raw_N, SIMD_m128(),
                ["AVX512vl", "AVX512bw", "SSE2"], 2),
            SIMD_Mask_Load(
                "_mm_mask_loadu_epi16(_mm_set1_epi16([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 2, False, raw_N, SIMD_m128(),
                ["AVX512vl", "AVX512bw", "SSE2"], 2),
            SIMD_Mask_Load(
                "_mm_mask_load_epi32(_mm_set1_epi32([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 4, True, raw_N, SIMD_m128(),
                ["AVX512vl", "AVX512f", "SSE2"], 2),
            SIMD_Mask_Load(
                "_mm_mask_loadu_epi32(_mm_set1_epi32([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 4, False, raw_N, SIMD_m128(),
                ["AVX512vl", "AVX512f", "SSE2"], 3),
            SIMD_Mask_Load(
                "_mm_mask_load_epi64(_mm_set1_epi64([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 8, True, raw_N, SIMD_m128(),
                ["AVX512vl", "AVX512f", "SSE2"], 2),
            SIMD_Mask_Load(
                "_mm_mask_loadu_epi64(_mm_set1_epi64([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 8, False, raw_N, SIMD_m128(),
                ["AVX512vl", "AVX512f", "SSE2"], 3),
            ###################################################################
            # These are universal best instructions if applicable
            SIMD_Full_Load("_mm256_load_si256((__m256i *)[ARR])", True,
                           SIMD_m256(), ["AVX"], 0),
            SIMD_Full_Load("_mm256_loadu_si256((__m256i *)[ARR])", False,
                           SIMD_m256(), ["AVX"], 1),

            # This is universal fallback and always lowest priority
            SIMD_Mask_Load_Fallback_As_Epi32(sort_type, raw_N, SIMD_m256(),
                                             ["AVX2"], 100),
            SIMD_Mask_Load(
                "_mm256_mask_loadu_epi8(_mm256_set1_epi8([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 1, False, raw_N, SIMD_m256(),
                ["AVX512vl", "AVX512bw", "AVX"], 2),
            SIMD_Mask_Load(
                "_mm256_mask_loadu_epi16(_mm256_set1_epi16([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 2, False, raw_N, SIMD_m256(),
                ["AVX512vl", "AVX512bw", "AVX"], 2),
            SIMD_Mask_Load(
                "_mm256_mask_load_epi32(_mm256_set1_epi32([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 4, True, raw_N, SIMD_m256(),
                ["AVX512vl", "AVX512f", "AVX"], 2),
            SIMD_Mask_Load(
                "_mm256_mask_loadu_epi32(_mm256_set1_epi32([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 4, False, raw_N, SIMD_m256(),
                ["AVX512vl", "AVX512f", "AVX"], 3),
            SIMD_Mask_Load(
                "_mm256_mask_load_epi64(_mm256_set1_epi64([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 8, True, raw_N, SIMD_m256(),
                ["AVX512vl", "AVX512f", "AVX"], 2),
            SIMD_Mask_Load(
                "_mm256_mask_loadu_epi64(_mm256_set1_epi64([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 8, False, raw_N, SIMD_m256(),
                ["AVX512vl", "AVX512f", "AVX"], 3),
            ###################################################################
            # These are universal best instructions if applicable
            SIMD_Full_Load("_mm512_load_si512((__m512i *)[ARR])", True,
                           SIMD_m512(), ["AVX512f"], 0),
            SIMD_Full_Load("_mm512_loadu_si512((__m512i *)[ARR])", False,
                           SIMD_m512(), ["AVX512f"], 1),
            SIMD_Mask_Load(
                "_mm512_mask_loadu_epi8(_mm512_set1_epi8([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 1, False, raw_N, SIMD_m512(),
                ["AVX512f", "AVX512bw", "AVX"], 2),
            SIMD_Mask_Load(
                "_mm512_mask_loadu_epi16(_mm512_set1_epi16([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 2, False, raw_N, SIMD_m512(),
                ["AVX512f", "AVX512bw", "AVX"], 2),
            SIMD_Mask_Load(
                "_mm512_mask_load_epi32(_mm512_set1_epi32([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 4, True, raw_N, SIMD_m512(), ["AVX512f"], 2),
            SIMD_Mask_Load(
                "_mm512_mask_loadu_epi32(_mm512_set1_epi32([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 4, False, raw_N, SIMD_m512(), ["AVX512f"], 3),
            SIMD_Mask_Load(
                "_mm512_mask_load_epi64(_mm512_set1_epi64([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 8, True, raw_N, SIMD_m512(), ["AVX512f"], 2),
            SIMD_Mask_Load(
                "_mm512_mask_loadu_epi64(_mm512_set1_epi64([MAX]), [LOAD_MASK], [ARR])",
                sort_type, 8, False, raw_N, SIMD_m512(), ["AVX512f"], 3),
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
class Compare_Exchange_Generator():
    def __init__(self, pairs, N, sort_type):
        self.pairs = copy.deepcopy(pairs)
        self.sort_N = sort_n(N, sort_type.sizeof())
        self.sort_type = sort_type
        self.simd_type = get_simd_type(self.sort_N * self.sort_type.sizeof())
        err_assert(self.simd_type.sizeof() >= self.sort_N * sort_type.sizeof(),
                   "invalid SIMD type selection")

        self.SIMD_min = instruction_filter(SIMD_Min().instructions,
                                           self.sort_type, self.simd_type)
        self.SIMD_max = instruction_filter(SIMD_Min().instructions,
                                           self.sort_type, self.simd_type)

        do_full_load = EXTRA_MEMORY
        if self.simd_type.sizeof() == N * sort_type.sizeof():
            do_full_load = True

        print("DO_FULL_LOAD: " + str(do_full_load))
        self.SIMD_load = instruction_filter(
            SIMD_Load(N, sort_type).instructions, self.sort_type,
            self.simd_type, ALIGNED_ACCESS, do_full_load)

    def Generate_Instructions(self):
        best_load = best_instruction(self.SIMD_load)
        print(best_load.generate_instruction())
        for i in range(0, int(len(self.pairs) / self.sort_N)):
            self.Compare_Exchange(i)

    def Compare_Exchange(self, cas_idx):
        cas_perm = []
        for i in range(cas_idx * self.sort_N, (cas_idx + 1) * self.sort_N):
            cas_perm.append(self.pairs[i])

        SIMD_blend = instruction_filter(
            SIMD_Blend(cas_perm).instructions, self.sort_type, self.simd_type)
        SIMD_permute = instruction_filter(
            SIMD_Permute(cas_perm).instructions, self.sort_type,
            self.simd_type)

        best_permutate = best_instruction(SIMD_permute)
        best_min = best_instruction(self.SIMD_min)
        best_max = best_instruction(self.SIMD_max)
        best_blend = best_instruction(SIMD_blend)

        print(best_permutate.generate_instruction())
        print(best_min.generate_instruction())
        print(best_max.generate_instruction())
        print(best_blend.generate_instruction())


######################################################################
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
    def __init__(self, N, sort_type, pairs):
        self.N = N
        self.sort_type = sort_type

        self.pairs = copy.deepcopy(pairs)

        self.N_groups = -1

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
        self.N_groups = len(grouped_pairs)
        err_assert(pidx == pairs_len,
                   "not all pairs processed {} != {}".format(pidx, pairs_len))

    def permutation(self):
        err_assert(self.N_groups != -1, "permutating before grouping")

        sort_N = sort_n(self.N, self.sort_type.sizeof())

        perm_arr = []
        for i in range(0, self.N_groups * sort_N):
            perm_arr.append((sort_N - 1) - (i % sort_N))

        current_group = int(0)
        idx = 0

        for i in range(0, self.N_groups):
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

        self.pairs = copy.deepcopy(perm_arr)
        err_assert(len(self.pairs) == len(perm_arr), "bad usage of deep copy")


class Bitonic():
    def __init__(self, N):
        self.name = "bitonic"
        self.N = N
        self.pairs = []

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

    def create_pairs(self):
        self.bitonic_sort(0, self.N, True)

        transformer = Transform(self.N, Sort_Type(1, True), self.pairs)
        transformer.unidirectional()
        self.pairs = copy.deepcopy(transformer.pairs)
        err_assert(
            len(self.pairs) == len(transformer.pairs), "Bad usage of deepcopy")
        return self.pairs


class Batcher():
    def __init__(self, N):
        self.name = "batcher"
        self.N = N
        self.pairs = []


class Oddeven():
    def __init__(self, N):
        self.name = "oddeven"
        self.N = N
        self.pairs = []


class Bosenelson():
    def __init__(self, N):
        self.name = "bosenelson"
        self.N = N
        self.pairs = []


class Minimum():
    def __init__(self, N):
        self.name = "minimum"
        self.N = N
        self.pairs = []


class Algorithms():
    def __init__(self, N):
        self.algorithms = [
            "bitonic", "batcher", "oddeven", "bosenelson", "minimum"
        ]
        self.implementations = [
            Bitonic(N),
            Batcher(N),
            Oddeven(N),
            Bosenelson(N),
            Minimum(N)
        ]
        err_assert(
            len(self.implementations) == len(self.algorithms),
            "algorithms and implementations arrays do not match")

    def get_algorithm(self, algorithm):

        for i in range(0, len(self.algorithms)):
            if algorithm == self.algorithms[i]:
                return self.implementations[i]
        err_assert(False, "No matching algorithm for " + algorithm)

    def valid_algorithm(self, algorithm):
        return algorithm in self.algorithms


class Network():
    def __init__(self, N, sort_type, algorithm_name):
        err_assert(N * sort_type.sizeof() <= 64, "N to large for network size")

        err_assert(
            Algorithms(N).valid_algorithm(algorithm_name),
            algorithm_name + " is unknown")

        self.N = N
        self.sort_type = sort_type
        self.algorithm = Algorithms(N).get_algorithm(algorithm_name)

    def create_pairs(self):
        return self.algorithm.create_pairs()

    def build_network(self):
        transformer = Transform(self.N, self.sort_type, self.create_pairs())
        transformer.group()
        transformer.permutation()
        return transformer.pairs


sizes = [1, 2, 4, 8]
for s in sizes:
    n_max = int(32 / s) + 1
    n_min = 4
    if s == 1:
        n_min = 8
    for i in range(n_min, n_max):
        print("[{}][{}]".format(s, i))
        network = Network(i, Sort_Type(s, Sign.UNSIGNED), "bitonic")

        cas_generator = Compare_Exchange_Generator(network.build_network(),
                                                   network.N,
                                                   network.sort_type)

        cas_generator.Generate_Instructions()
