#ifndef _INSTRUCTION_SETS_H_
#define _INSTRUCTION_SETS_H_

#ifndef AVX_512_VMBI
#if defined __AVX512VBMI__
#define AVX_512_VBMI 1
#else
#define AVX_512_VBMI 0
#endif
#endif

#ifndef AVX_512_BW
#if defined __AVX512BW__
#define AVX_512_BW 1
#else
#define AVX_512_BW 0
#endif
#endif

#ifndef AVX_512_F
#if defined __AVX512F__
#define AVX_512_F 1
#else
#define AVX_512_F 0
#endif
#endif

#ifndef AVX_512_VL
#if defined __AVX512VL__
#define AVX_512_VL 1
#else
#define AVX_512_VL 0
#endif
#endif


#ifndef AVX_256
#if defined __AVX2__
#define AVX_256 1
#else
#define AVX_256 0
#endif
#endif

#ifndef AVX_128
#if defined __AVX__
#define AVX_128 1
#else
#define AVX_128 0
#endif
#endif

#ifndef SSE_2
#if defined __SSE2__
#define SSE_2 1
#else
#define SSE_2 0
#endif
#endif

#ifndef SSE_3
#if defined __SSE3__
#define SSE_3 1
#else
#define SSE_3 0
#endif
#endif

#ifndef SSE_4_1
#if defined __SSE4_1__
#define SSE_4_1 1
#else
#define SSE_4_1 0
#endif
#endif

#ifndef SSE_4_2
#if defined __SSE4_2__
#define SSE_4_2 1
#else
#define SSE_4_2 0
#endif
#endif


#define VERIFY_SSE_4_1()                                                       \
    if constexpr (SSE_4_1 == 0) {                                              \
        CONSTEXPR_FAIL("Error SSE4.1 support required\n");                     \
    }

#define VERIFY_SSE_4_2()                                                       \
    if constexpr (SSE_4_2 == 0) {                                              \
        CONSTEXPR_FAIL("Error SSE4.2 support required\n");                     \
    }

#define VERIFY_SSE_3()                                                         \
    if constexpr (SSE_3 == 0) {                                                \
        CONSTEXPR_FAIL("Error SSE3 support required\n");                       \
    }

#define VERIFY_SSE_2()                                                         \
    if constexpr (SSE_2 == 0) {                                                \
        CONSTEXPR_FAIL("Error SSE2 support required\n");                       \
    }

#define VERIFY_AVX_128()                                                       \
    if constexpr (AVX_128 == 0) {                                              \
        CONSTEXPR_FAIL("Error AVX support required\n");                        \
    }
#define VERIFY_AVX_256()                                                       \
    if constexpr (AVX_256 == 0) {                                              \
        CONSTEXPR_FAIL("Error AVX2 support required\n");                       \
    }
#define VERIFY_AVX_512_F()                                                     \
    if constexpr (AVX_512_F == 0) {                                            \
        CONSTEXPR_FAIL("Error AVX512F support required\n");                    \
    }

#define VERIFY_AVX_512_VL()                                                    \
    if constexpr (AVX_512_VL == 0) {                                           \
        CONSTEXPR_FAIL("Error AVX512F support required\n");                    \
    }

#define VERIFY_AVX_512_BW()                                                    \
    if constexpr (AVX_512_BW == 0) {                                           \
        CONSTEXPR_FAIL("Error AVX512BW support required\n");                   \
    }

#define VERIFY_AVX_512_VMBI()                                                  \
    if constexpr (AVX_512_VMBI == 0) {                                         \
        CONSTEXPR_FAIL("Error AVX512F support required\n");                    \
    }


#endif
