#! /usr/bin/env python3

import cpufeature
import copy


def err_assert(check, msg):
    if check is False:
        print("Error: " + msg)
        exit(-1)


class Type_Info():
    def __init__(self):
        self.type_list = [
            "uint8_t", "int8_t", "uint16_t", "int16_t", "uint32_t", "int32_t",
            "uint64_t", "int64_t", "float", "double"
        ]
        self.type_sizes = [1, 1, 2, 2, 4, 4, 8, 8, 4, 8]
        self.signed = [
            False, True, False, True, False, True, False, True, True, True
        ]
        err_assert((len(self.type_sizes) == len(self.type_list)
                    and len(self.type_sizes) == len(self.signed)),
                   "meta data array size mismatch")

    def signed(self, T):
        for i in range(0, self.type_list):
            if self.type_list[i] == T:
                return self.signed[i]
        err_assert(False, T + " does not match any type")

    def sizeof(self, T):
        for i in range(0, self.type_list):
            if self.type_list[i] == T:
                return self.type_sizes[i]
        err_assert(False, T + " does not match any type")

    def sizeof_bits(self, T):
        return 8 * self.sizeof(T)

    def valid_type(self, T):
        return T in self.type_list


class SIMD_Constraints():
    def __init__(self, constraints):
        self.constraints = copy.deepcopy(constraints)

    def has_support(self, fields):
        for field in fields:
            if field == "":
                return True
            elif cpufeature.eval(field) is False:
                return False
        return True


class SIMD_m64():
    def __init__(self):
        self.SIMD_constraints = SIMD_Constraints(["MMX"])

    def type_to_string():
        return "__m64"

    def instruction_prefix():
        return "_mm_"

    def instruction_full_postfix():
        return "_si64"

    def sizeof():
        return 8

    def has_support():
        return self.SIMD_constraints.has_support()


class SIMD_m128():
    def __init__(self):
        self.SIMD_constraints = SIMD_Constraints(["SSE2"])

    def type_to_string():
        return "__m128i"

    def instruction_prefix():
        return "_mm_"

    def instruction_full_postfix():
        return "_si128"

    def sizeof():
        return 16

    def has_support():
        return self.SIMD_constraints.has_support()


class SIMD_m256():
    def __init__(self):
        self.SIMD_constraints = SIMD_Constraints(["AVX2"])

    def type_to_string():
        return "__m256i"

    def instruction_prefix():
        return "_mm256_"

    def instruction_full_postfix():
        return "_si256"

    def sizeof():
        return 32

    def has_support():
        return self.SIMD_constraints.has_support()


class SIMD_m512():
    def __init__(self):
        self.SIMD_constraints = SIMD_Constraints(["AVX512f"])

    def type_to_string():
        return "__m512i"

    def instruction_prefix():
        return "_mm512_"

    def instruction_full_postfix():
        return "_si512"

    def sizeof():
        return 32

    def has_support():
        return self.SIMD_constraints.has_support()


class SIMD_Types():
    def __init__(self):
        self.type_list = [SIMD_m64(), SIMD_m128(), SIMD_m256(), SIMD_m512()]

    def get_type(self, sort_bytes):
        for SIMD_T in self.type_list:
            if SIMD_T.sizeof() >= sort_bytes:
                return SIMD_T
        err_assert(False, "No SIMD type to fit " + str(sort_bytes) + " found")


class SIMD_Instruction_Padding():
    def __init__(self, T, SIMD_type, is_signed, is_full):
        self.T = T
        self.SIMD_type = SIMD_type
        self.is_signed = is_signed
        self.is_full = is_full

    def instruction_prefix(self):
        return self.SIMD_type.instruction_prefix()

    def instruction_postfix(self):
        if self.is_full is True:
            return self.SIMD_type.instruction_postfix()
        else:
            type_info = Type_Info()
            base = "_"
            if self.SIMD_type.sizeof() == 8:
                base += "p"
            else:
                base += "ep"

            signed = type_info.signed(self.T)
            if self.is_signed is False or signed is True:
                return base + "i" + str(type_info.sizeof_bits(self.T))
            else:
                return base + "u" + str(type_info.sizeof_bits(self.T))


class SIMD_Instruction_Base():
    def __init__(self, T, is_signed, is_full, SIMD_constraint, SIMD_type):
        self.attributes = SIMD_Constraints(SIMD_constraint)
        self.SIMD_type = SIMD_type
        self.T = T
        self.SIMD_instruction_padding = SIMD_Instruction_Padding(
            T, is_signed, is_full)

        def has_support(self):
            return (self.SIMD_type.has_support()
                    and self.attributes.has_support())

        def instruction_prefix(self):
            return self.SIMD_instruction_padding.instruction_prefix()

        def instruction_postfix(self):
            return self.SIMD_instruction_padding.instruction_postfix()


######################################################################
###############################   Min   ##############################


class SIMD_Min():
    def __init__(self, T, SIMD_constraint, SIMD_type):
        self.SIMD_instruction_base = SIMD_Instruction_Base(
            T, True, False, SIMD_constraint, SIMD_type)

    def has_support(self):
        return self.SIMD_instruction_base.has_support()

    def generate_instruction(self):
        err_assert(
            self.has_support(),
            "Necessary instructions not supported\n\tMissing: " +
            str(self.SIMD_constraint))

        return "[V] = " + self.SIMD_instruction_base.instruction_prefix(
        ) + "min" + self.SIMD_instruction_base.instruction_postfix(
        ) + "([V1], [V2]);"


# Min __m64
class SIMD_Min_m64_t8_u(SIMD_Min):
    def __init__(self):
        super().__init__("int8_t", ["SSE"], SIMD_m64())


class SIMD_Min_m64_t8_s(SIMD_Min):
    def __init__(self):
        self.SIMD_required = ["MMX"]
        super().__init__("uint8_t", self.SIMD_constraint, SIMD_m64())

    def generate_instruction(self):
        err_assert(
            self.has_support(),
            "Necessary instructions not supported\n\tMissing: " +
            str(self.SIMD_required))

        instruction = "__m64 _tmp_min0 = _mm_cmpgt_pi8([V2], [V1]);\n"
        instruction += "[V] = _mm_or_si64(_mm_and_si64(_tmp_min0, [V1]), _mm_andnot_si64(_tmp_min0, [V2]));"
        return instruction


class SIMD_Min_m64_t16_u(SIMD_Min):
    def __init__(self):
        self.SIMD_required = ["MMX"]
        super().__init__("uint16_t", self.SIMD_required, SIMD_m64())

    def generate_instruction(self):
        err_assert(
            self.has_support(),
            "Necessary instructions not supported\n\tMissing: " +
            str(self.SIMD_required))

        instruction = "__m64 _tmp_min0 = _mm_set1_pi16(1 << 15);\n"
        instruction += "__m64 _tmp_min1 = _mm_cmpgt_pi16(_mm_xor_si64([V1], _tmp_min0), _mm_xor_si64([V2], _tmp_min0));\n"
        instruction += "[V] = _mm_or_si64(_mm_and_si64(_tmp_min1, [V2]), _mm_andnot_si64(_tmp_min1, [V1]));"
        return instruction


class SIMD_Min_m64_t16_s(SIMD_Min):
    def __init__(self):
        super().__init__("int16_t", ["SSE"], SIMD_m64())


# Min __m128i
class SIMD_Min_m128_t8_u(SIMD_Min):
    def __init__(self):
        super().__init__("uint8_t", ["SSE2"], SIMD_m128())


class SIMD_Min_m128_t8_s(SIMD_Min):
    def __init__(self):
        super().__init__("int8_t", ["SSE4.1"], SIMD_m128())


class SIMD_Min_m128_t16_u(SIMD_Min):
    def __init__(self):
        super().__init__("uint16_t", ["SSE4.1"], SIMD_m128())


class SIMD_Min_m128_t16_s(SIMD_Min):
    def __init__(self):
        super().__init__("int16_t", ["SSE2"], SIMD_m128())


class SIMD_Min_m128_t32_u(SIMD_Min):
    def __init__(self):
        super().__init__("uint32_t", ["SSE4.1"], SIMD_m128())


class SIMD_Min_m128_t32_s(SIMD_Min):
    def __init__(self):
        super().__init__("int32_t", ["SSE4.1"], SIMD_m128())


# Min __m256i
class SIMD_Min_m256_t8_u(SIMD_Min):
    def __init__(self):
        super().__init__("uint8_t", ["AVX2"], SIMD_m256())


class SIMD_Min_m256_t8_s(SIMD_Min):
    def __init__(self):
        super().__init__("int8_t", ["AVX2"], SIMD_m256())


class SIMD_Min_m256_t16_u(SIMD_Min):
    def __init__(self):
        super().__init__("uint16_t", ["AVX2"], SIMD_m256())


class SIMD_Min_m256_t16_s(SIMD_Min):
    def __init__(self):
        super().__init__("int16_t", ["AVX2"], SIMD_m256())


class SIMD_Min_m256_t32_u(SIMD_Min):
    def __init__(self):
        super().__init__("uint32_t", ["AVX2"], SIMD_m256())


class SIMD_Min_m256_t32_s(SIMD_Min):
    def __init__(self):
        super().__init__("int32_t", ["AVX2"], SIMD_m256())


class SIMD_Min_m256_t64_u(SIMD_Min):
    def __init__(self):
        T = "uint64_t"

        # these are minimum required constraints
        self.SIMD_required = ["AVX", "AVX2"]
        self.SIMD_min = [
            SIMD_Min(T, ["AVX512vl", "AVX512f"], SIMD_m128()),
            SIMD_Min(T, self.SIMD_required, SIMD_m128())
        ]

    def generate_instruction(self):
        if self.SIMD_min[0].has_support():
            return self.SIMD_min[0].generate_instruction()
        else:
            err_assert(
                self.has_support(),
                "Necessary instructions not supported\n\tMissing: " +
                str(self.SIMD_required))
            instruction = "__m128i _tmp_min0 = _mm_set1_epi64x(1UL) << 63);\n"
            instruction += "__m128i _tmp_min1 = _mm256_cmpgt_epi64(_mm256_xor_si256([V1], _tmp_min0), _mm256_xor_si256([V2], _tmp_min0));\n"
            instruction += "[V] = _mm256_blendv_epi8([V1], [V2], _tmp_min1);"
            return instruction


class SIMD_Min_m256_t64_s(SIMD_Min):
    def __init__(self):
        T = "uint64_t"

        # these are minimum required constraints
        self.SIMD_required = ["AVX2"]
        self.SIMD_min = [
            SIMD_Min(T, ["AVX512vl", "AVX512f"], SIMD_m128()),
            SIMD_Min(T, self.SIMD_required, SIMD_m128())
        ]

    def generate_instruction(self):
        if self.SIMD_min[0].has_support():
            return self.SIMD_min[0].generate_instruction()
        else:
            err_assert(
                self.has_support(),
                "Necessary instructions not supported\n\tMissing: " +
                str(self.SIMD_required))
            instruction = "__m128i _tmp_min0 = _mm256_cmpgt_epi64([V1], [V2]);\n"
            instruction += "[V] = _mm256_blendv_epi8([V1], [V2], _tmp_min0);"
            return instruction


# Min __m512i
class SIMD_Min_m512_t8_u(SIMD_Min):
    def __init__(self):
        super().__init__("uint8_t", ["AVX512bw"], SIMD_m512())


class SIMD_Min_m512_t8_s(SIMD_Min):
    def __init__(self):
        super().__init__("int8_t", ["AVX512bw"], SIMD_m512())


class SIMD_Min_m512_t16_u(SIMD_Min):
    def __init__(self):
        super().__init__("uint16_t", ["AVX512bw"], SIMD_m512())


class SIMD_Min_m512_t16_s(SIMD_Min):
    def __init__(self):
        super().__init__("int16_t", ["AVX512bw"], SIMD_m512())


class SIMD_Min_m512_t32_u(SIMD_Min):
    def __init__(self):
        super().__init__("uint32_t", ["AVX512f"], SIMD_m512())


class SIMD_Min_m512_t32_s(SIMD_Min):
    def __init__(self):
        super().__init__("int32_t", ["AVX512f"], SIMD_m512())


class SIMD_Min_m512_t64_u(SIMD_Min):
    def __init__(self):
        super().__init__("uint64_t", ["AVX512f"], SIMD_m512())


class SIMD_Min_m512_t64_s(SIMD_Min):
    def __init__(self):
        super().__init__("int64_t", ["AVX512f"], SIMD_m512())


######################################################################
###############################   Max   ##############################


class SIMD_Max():
    def __init__(self, T, SIMD_constraint, SIMD_type):
        self.SIMD_instruction_base = SIMD_Instruction_Base(
            T, True, False, SIMD_constraint, SIMD_type)

    def has_support(self):
        return self.SIMD_instruction_base.has_support()

    def generate_instruction(self):
        err_assert(
            self.has_support(),
            "Necessary instructions not supported\n\tMissing: " +
            str(self.SIMD_constraint))

        return "[V] = " + self.SIMD_instruction_base.instruction_prefix(
        ) + "max" + self.SIMD_instruction_base.instruction_postfix(
        ) + "([V1], [V2]);"


# Max __m64
class SIMD_Max_m64_t8_u(SIMD_Max):
    def __init__(self):
        super().__init__("int8_t", ["SSE"], SIMD_m64())


class SIMD_Max_m64_t8_s(SIMD_Max):
    def __init__(self):
        self.SIMD_required = ["MMX"]
        super().__init__("uint8_t", self.SIMD_constraint, SIMD_m64())

    def generate_instruction(self):
        err_assert(
            self.has_support(),
            "Necessary instructions not supported\n\tMissing: " +
            str(self.SIMD_required))

        instruction = "__m64 _tmp_max0 = _mm_cmpgt_pi8([V1], [V2]);\n"
        instruction += "[V] = _mm_or_si64(_mm_and_si64(_tmp_max0, [V1]), _mm_andnot_si64(_tmp_max0, [V2]));"
        return instruction


class SIMD_Max_m64_t16_u(SIMD_Max):
    def __init__(self):
        self.SIMD_required = ["MMX"]
        super().__init__("uint16_t", self.SIMD_required, SIMD_m64())

    def generate_instruction(self):
        err_assert(
            self.has_support(),
            "Necessary instructions not supported\n\tMissing: " +
            str(self.SIMD_required))

        instruction = "__m64 _tmp_max0 = _mm_set1_pi16(1 << 15);\n"
        instruction += "__m64 _tmp_max1 = _mm_cmpgt_pi16(_mm_xor_si64([V1], _tmp_max0), _mm_xor_si64([V2], _tmp_max0));\n"
        instruction += "[V] = _mm_or_si64(_mm_and_si64(_tmp_max1, [V1]), _mm_andnot_si64(_tmp_max1, [V2]));"
        return instruction


class SIMD_Max_m64_t16_s(SIMD_Max):
    def __init__(self):
        super().__init__("int16_t", ["SSE"], SIMD_m64())


# Max __m128i
class SIMD_Max_m128_t8_u(SIMD_Max):
    def __init__(self):
        super().__init__("uint8_t", ["SSE2"], SIMD_m128())


class SIMD_Max_m128_t8_s(SIMD_Max):
    def __init__(self):
        super().__init__("int8_t", ["SSE4.1"], SIMD_m128())


class SIMD_Max_m128_t16_u(SIMD_Max):
    def __init__(self):
        super().__init__("uint16_t", ["SSE4.1"], SIMD_m128())


class SIMD_Max_m128_t16_s(SIMD_Max):
    def __init__(self):
        super().__init__("int16_t", ["SSE2"], SIMD_m128())


class SIMD_Max_m128_t32_u(SIMD_Max):
    def __init__(self):
        super().__init__("uint32_t", ["SSE4.1"], SIMD_m128())


class SIMD_Max_m128_t32_s(SIMD_Max):
    def __init__(self):
        super().__init__("int32_t", ["SSE4.1"], SIMD_m128())


# Max __m256i
class SIMD_Max_m256_t8_u(SIMD_Max):
    def __init__(self):
        super().__init__("uint8_t", ["AVX2"], SIMD_m256())


class SIMD_Max_m256_t8_s(SIMD_Max):
    def __init__(self):
        super().__init__("int8_t", ["AVX2"], SIMD_m256())


class SIMD_Max_m256_t16_u(SIMD_Max):
    def __init__(self):
        super().__init__("uint16_t", ["AVX2"], SIMD_m256())


class SIMD_Max_m256_t16_s(SIMD_Max):
    def __init__(self):
        super().__init__("int16_t", ["AVX2"], SIMD_m256())


class SIMD_Max_m256_t32_u(SIMD_Max):
    def __init__(self):
        super().__init__("uint32_t", ["AVX2"], SIMD_m256())


class SIMD_Max_m256_t32_s(SIMD_Max):
    def __init__(self):
        super().__init__("int32_t", ["AVX2"], SIMD_m256())


class SIMD_Max_m256_t64_u(SIMD_Max):
    def __init__(self):
        T = "uint64_t"

        # these are maximum required constraints
        self.SIMD_required = ["AVX", "AVX2"]
        self.SIMD_max = [
            SIMD_Max(T, ["AVX512vl", "AVX512f"], SIMD_m128()),
            SIMD_Max(T, self.SIMD_required, SIMD_m128())
        ]

    def generate_instruction(self):
        if self.SIMD_max[0].has_support():
            return self.SIMD_max[0].generate_instruction()
        else:
            err_assert(
                self.has_support(),
                "Necessary instructions not supported\n\tMissing: " +
                str(self.SIMD_required))
            instruction = "__m128i _tmp_max0 = _mm_set1_epi64x(1UL) << 63);\n"
            instruction += "__m128i _tmp_max1 = _mm256_cmpgt_epi64(_mm256_xor_si256([V1], _tmp_max0), _mm256_xor_si256([V2], _tmp_max0));\n"
            instruction += "[V] = _mm256_blendv_epi8([V2], [V1], _tmp_max1);"
            return instruction


class SIMD_Max_m256_t64_s(SIMD_Max):
    def __init__(self):
        T = "uint64_t"

        # these are maximum required constraints
        self.SIMD_required = ["AVX2"]
        self.SIMD_max = [
            SIMD_Max(T, ["AVX512vl", "AVX512f"], SIMD_m128()),
            SIMD_Max(T, self.SIMD_required, SIMD_m128())
        ]

    def generate_instruction(self):
        if self.SIMD_max[0].has_support():
            return self.SIMD_max[0].generate_instruction()
        else:
            err_assert(
                self.has_support(),
                "Necessary instructions not supported\n\tMissing: " +
                str(self.SIMD_required))
            instruction = "__m128i _tmp_max0 = _mm256_cmpgt_epi64([V1], [V2]);\n"
            instruction += "[V] = _mm256_blendv_epi8([V2], [V1], _tmp_max0);"
            return instruction


# Max __m512i
class SIMD_Max_m512_t8_u(SIMD_Max):
    def __init__(self):
        super().__init__("uint8_t", ["AVX512bw"], SIMD_m512())


class SIMD_Max_m512_t8_s(SIMD_Max):
    def __init__(self):
        super().__init__("int8_t", ["AVX512bw"], SIMD_m512())


class SIMD_Max_m512_t16_u(SIMD_Max):
    def __init__(self):
        super().__init__("uint16_t", ["AVX512bw"], SIMD_m512())


class SIMD_Max_m512_t16_s(SIMD_Max):
    def __init__(self):
        super().__init__("int16_t", ["AVX512bw"], SIMD_m512())


class SIMD_Max_m512_t32_u(SIMD_Max):
    def __init__(self):
        super().__init__("uint32_t", ["AVX512f"], SIMD_m512())


class SIMD_Max_m512_t32_s(SIMD_Max):
    def __init__(self):
        super().__init__("int32_t", ["AVX512f"], SIMD_m512())


class SIMD_Max_m512_t64_u(SIMD_Max):
    def __init__(self):
        super().__init__("uint64_t", ["AVX512f"], SIMD_m512())


class SIMD_Max_m512_t64_s(SIMD_Max):
    def __init__(self):
        super().__init__("int64_t", ["AVX512f"], SIMD_m512())


######################################################################


class Bitonic:
    def __init__(self, N):
        self.name = "bitonic"
        self.N = N
        self.pairs = []


class Batcher:
    def __init__(self, N):
        self.name = "batcher"
        self.N = N
        self.pairs = []


class Oddeven:
    def __init__(self, N):
        self.name = "oddeven"
        self.N = N
        self.pairs = []


class Bosenelson:
    def __init__(self, N):
        self.name = "bosenelson"
        self.N = N
        self.pairs = []


class Minimum:
    def __init__(self, N):
        self.name = "minimum"
        self.N = N
        err_assert(self.N <= 32, "32 is max N for minimum")

        self.pairs = []


class Algorithms:
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


class Network:
    def __init__(self, N, T, algorithm_name, SIMD, BUILTIN):
        err_assert(N * Type_info.sizeof(T) <= 64,
                   "N to large for network size")
        err_assert(Type_info.valid_type(T), T + " is unknown")

        err_assert(Algorithms().valid_algorithm(algorithm_name),
                   algorithm_name + " is unknown")

        self.N = N
        self.T = T
        self.algorithm = Algorithms().get_algorithm(algorithm_name)
        self.SIMD = SIMD
        self.BUILTIN = BUILTIN
