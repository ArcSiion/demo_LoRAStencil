#include "defines.h"
//强制开启512位，否则error
#ifndef __AVX512F__
#error "defines_512.h requires AVX-512F. Compile with -mavx512f."
#endif


#define SET_COFF_512 \
    __m512d vc0 = _mm512_set1_pd(C0); \
    __m512d vc1 = _mm512_set1_pd(C1)

#if defined(heat)
#define Compute_1vector_512(v0, v1, v2) \
    (v0) = _mm512_mul_pd((vc0), _mm512_add_pd(_mm512_fmadd_pd((vc1), (v1), (v2)), (v0)))

#define Compute_1vector_tmp_512(out, v0, v1, v2) \
    (out) = _mm512_mul_pd((vc0), _mm512_add_pd(_mm512_fmadd_pd((vc1), (v1), (v2)), (v0)))
#else
#define Compute_1vector_512(v0, v1, v2) \
    (v0) = _mm512_fmadd_pd((v1), (vc0), _mm512_mul_pd(_mm512_add_pd((v0), (v2)), (vc1)))

#define Compute_1vector_tmp_512(out, v0, v1, v2) \
    (out) = _mm512_fmadd_pd((v1), (vc0), _mm512_mul_pd(_mm512_add_pd((v0), (v2)), (vc1)))
#endif

#define loadv_512(b) _mm512_loadu_pd((b))
#define storev_512(a, b) _mm512_storeu_pd((a), (b))

// 需要传地址，错误写法：vload_512(v0, tmp[0]);
// 正确写法：vload_512(v0, &tmp[0]);
#define vload_512(a, b) (a) = _mm512_loadu_pd((b))
#define vstore_512(a, b) _mm512_storeu_pd((a), (b))

#define vloadset_512(a, b, c, d, e, f, g, h, i) \
    (a) = _mm512_set_pd((b), (c), (d), (e), (f), (g), (h), (i))

#define vallset_512(a, b) \
    (a) = _mm512_set1_pd((b))

// a0 a1 a2 a3 a4 a5 a6 a7 -> a1 a2 a3 a4 a5 a6 a7 a0
#define vrotate_512_low2high(a) \
    _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(a), _mm512_castpd_si512(a), 1))

// a0 a1 a2 a3 a4 a5 a6 a7 -> a7 a0 a1 a2 a3 a4 a5 a6
#define vrotate_512_high2low(a) \
    _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(a), _mm512_castpd_si512(a), 7))

#define blend_512_7012345(out, in2, in1) \
    (out) = vrotate_512_high2low(_mm512_mask_blend_pd(0x80, (in1), (in2)))

#define blend_512_1234567(out, in1, in2) \
    (out) = vrotate_512_low2high(_mm512_mask_blend_pd(0x01, (in1), (in2)))

#define Compute_3vector_512(v0, v1, v2, v3, v4) \
    Compute_1vector_512((v0), (v1), (v2)); \
    Compute_1vector_512((v1), (v2), (v3)); \
    Compute_1vector_512((v2), (v3), (v4))

#define Compute_4vector_512(v0, v1, v2, v3, v4, v5) \
    Compute_1vector_512((v0), (v1), (v2)); \
    Compute_1vector_512((v1), (v2), (v3)); \
    Compute_1vector_512((v2), (v3), (v4)); \
    Compute_1vector_512((v3), (v4), (v5))

#define Compute_7vector_512(v0, v1, v2, v3, v4, v5, v6, v7, v8) \
    Compute_1vector_512((v0), (v1), (v2)); \
    Compute_1vector_512((v1), (v2), (v3)); \
    Compute_1vector_512((v2), (v3), (v4)); \
    Compute_1vector_512((v3), (v4), (v5)); \
    Compute_1vector_512((v4), (v5), (v6)); \
    Compute_1vector_512((v5), (v6), (v7)); \
    Compute_1vector_512((v6), (v7), (v8))

#define Compute_8vector_512(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9) \
    Compute_1vector_512((v0), (v1), (v2)); \
    Compute_1vector_512((v1), (v2), (v3)); \
    Compute_1vector_512((v2), (v3), (v4)); \
    Compute_1vector_512((v3), (v4), (v5)); \
    Compute_1vector_512((v4), (v5), (v6)); \
    Compute_1vector_512((v5), (v6), (v7)); \
    Compute_1vector_512((v6), (v7), (v8)); \
    Compute_1vector_512((v7), (v8), (v9))

#define transpose8x8_pd(r0, r1, r2, r3, r4, r5, r6, r7) \
    { \
        __m512d t0 = _mm512_unpacklo_pd((r0), (r1)); \
        __m512d t1 = _mm512_unpackhi_pd((r0), (r1)); \
        __m512d t2 = _mm512_unpacklo_pd((r2), (r3)); \
        __m512d t3 = _mm512_unpackhi_pd((r2), (r3)); \
        __m512d t4 = _mm512_unpacklo_pd((r4), (r5)); \
        __m512d t5 = _mm512_unpackhi_pd((r4), (r5)); \
        __m512d t6 = _mm512_unpacklo_pd((r6), (r7)); \
        __m512d t7 = _mm512_unpackhi_pd((r6), (r7)); \
        const __m512i idx0 = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13); \
        const __m512i idx1 = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15); \
        __m512d tt0 = _mm512_permutex2var_pd(t0, idx0, t2); \
        __m512d tt1 = _mm512_permutex2var_pd(t0, idx1, t2); \
        __m512d tt2 = _mm512_permutex2var_pd(t1, idx0, t3); \
        __m512d tt3 = _mm512_permutex2var_pd(t1, idx1, t3); \
        __m512d tt4 = _mm512_permutex2var_pd(t4, idx0, t6); \
        __m512d tt5 = _mm512_permutex2var_pd(t4, idx1, t6); \
        __m512d tt6 = _mm512_permutex2var_pd(t5, idx0, t7); \
        __m512d tt7 = _mm512_permutex2var_pd(t5, idx1, t7); \
        (r0) = _mm512_shuffle_f64x2(tt0, tt4, 0x44); \
        (r1) = _mm512_shuffle_f64x2(tt2, tt6, 0x44); \
        (r2) = _mm512_shuffle_f64x2(tt1, tt5, 0x44); \
        (r3) = _mm512_shuffle_f64x2(tt3, tt7, 0x44); \
        (r4) = _mm512_shuffle_f64x2(tt0, tt4, 0xEE); \
        (r5) = _mm512_shuffle_f64x2(tt2, tt6, 0xEE); \
        (r6) = _mm512_shuffle_f64x2(tt1, tt5, 0xEE); \
        (r7) = _mm512_shuffle_f64x2(tt3, tt7, 0xEE); \
    }


//==vectime_stride7_512====================================================

#define Input_Output_1_512(out, v1, in) \
    (v1) = vrotate_512_high2low(v1); \
    (out) = (v1); \
    (v1) = _mm512_mask_blend_pd(0x01, (v1), (in)); \
    (in) = vrotate_512_low2high(in)

#define Input_Output_2_512(out, v1, in) \
    (v1) = vrotate_512_high2low(v1); \
    (out) = _mm512_mask_blend_pd(0x02, (out), _mm512_broadcastsd_pd(_mm512_castpd512_pd128(v1))); \
    (v1) = _mm512_mask_blend_pd(0x01, (v1), (in)); \
    (in) = vrotate_512_low2high(in)

#define Input_Output_3_512(out, v1, in) \
    (v1) = vrotate_512_high2low(v1); \
    (out) = _mm512_mask_blend_pd(0x04, (out), _mm512_broadcastsd_pd(_mm512_castpd512_pd128(v1))); \
    (v1) = _mm512_mask_blend_pd(0x01, (v1), (in)); \
    (in) = vrotate_512_low2high(in)

#define Input_Output_4_512(out, v1, in) \
    (v1) = vrotate_512_high2low(v1); \
    (out) = _mm512_mask_blend_pd(0x08, (out), _mm512_broadcastsd_pd(_mm512_castpd512_pd128(v1))); \
    (v1) = _mm512_mask_blend_pd(0x01, (v1), (in)); \
    (in) = vrotate_512_low2high(in)

#define Input_Output_5_512(out, v1, in) \
    (v1) = vrotate_512_high2low(v1); \
    (out) = _mm512_mask_blend_pd(0x10, (out), _mm512_broadcastsd_pd(_mm512_castpd512_pd128(v1))); \
    (v1) = _mm512_mask_blend_pd(0x01, (v1), (in)); \
    (in) = vrotate_512_low2high(in)

#define Input_Output_6_512(out, v1, in) \
    (v1) = vrotate_512_high2low(v1); \
    (out) = _mm512_mask_blend_pd(0x20, (out), _mm512_broadcastsd_pd(_mm512_castpd512_pd128(v1))); \
    (v1) = _mm512_mask_blend_pd(0x01, (v1), (in)); \
    (in) = vrotate_512_low2high(in)

#define Input_Output_7_512(out, v1, in) \
    (v1) = vrotate_512_high2low(v1); \
    (out) = _mm512_mask_blend_pd(0x40, (out), _mm512_broadcastsd_pd(_mm512_castpd512_pd128(v1))); \
    (v1) = _mm512_mask_blend_pd(0x01, (v1), (in)); \
    (in) = vrotate_512_low2high(in)

#define Input_Output_8_512(out, v1, in) \
    (v1) = vrotate_512_high2low(v1); \
    (out) = _mm512_mask_blend_pd(0x80, (out), _mm512_broadcastsd_pd(_mm512_castpd512_pd128(v1))); \
    (v1) = _mm512_mask_blend_pd(0x01, (v1), (in))


//==vectime_transpose_inout_512============================================

#define InOut_POS_0_512(out, v, in) \
    (out) = _mm512_mask_blend_pd(0x01, (out), (v)); \
    (v) = _mm512_mask_blend_pd(0x01, (v), (in))

#define InOut_POS_1_512(out, v, in) \
    (out) = _mm512_mask_blend_pd(0x02, (out), (v)); \
    (v) = _mm512_mask_blend_pd(0x02, (v), (in))

#define InOut_POS_2_512(out, v, in) \
    (out) = _mm512_mask_blend_pd(0x04, (out), (v)); \
    (v) = _mm512_mask_blend_pd(0x04, (v), (in))

#define InOut_POS_3_512(out, v, in) \
    (out) = _mm512_mask_blend_pd(0x08, (out), (v)); \
    (v) = _mm512_mask_blend_pd(0x08, (v), (in))

#define InOut_POS_4_512(out, v, in) \
    (out) = _mm512_mask_blend_pd(0x10, (out), (v)); \
    (v) = _mm512_mask_blend_pd(0x10, (v), (in))

#define InOut_POS_5_512(out, v, in) \
    (out) = _mm512_mask_blend_pd(0x20, (out), (v)); \
    (v) = _mm512_mask_blend_pd(0x20, (v), (in))

#define InOut_POS_6_512(out, v, in) \
    (out) = _mm512_mask_blend_pd(0x40, (out), (v)); \
    (v) = _mm512_mask_blend_pd(0x40, (v), (in))

#define InOut_POS_7_512(out, v, in) \
    (out) = _mm512_mask_blend_pd(0x80, (out), (v)); \
    (v) = _mm512_mask_blend_pd(0x80, (v), (in))

#define Compute_Inout_Pos_0_Vector_512(v0, v1, v2, out0, in1) \
    Compute_1vector_512((v0), (v1), (v2)); \
    InOut_POS_0_512((out0), (v0), (in1))

#define Compute_Inout_Pos_1_Vector_512(v0, v1, v2, out0, in1) \
    Compute_1vector_512((v0), (v1), (v2)); \
    InOut_POS_1_512((out0), (v0), (in1))

#define Compute_Inout_Pos_2_Vector_512(v0, v1, v2, out0, in1) \
    Compute_1vector_512((v0), (v1), (v2)); \
    InOut_POS_2_512((out0), (v0), (in1))

#define Compute_Inout_Pos_3_Vector_512(v0, v1, v2, out0, in1) \
    Compute_1vector_512((v0), (v1), (v2)); \
    InOut_POS_3_512((out0), (v0), (in1))

#define Compute_Inout_Pos_4_Vector_512(v0, v1, v2, out0, in1) \
    Compute_1vector_512((v0), (v1), (v2)); \
    InOut_POS_4_512((out0), (v0), (in1))

#define Compute_Inout_Pos_5_Vector_512(v0, v1, v2, out0, in1) \
    Compute_1vector_512((v0), (v1), (v2)); \
    InOut_POS_5_512((out0), (v0), (in1))

#define Compute_Inout_Pos_6_Vector_512(v0, v1, v2, out0, in1) \
    Compute_1vector_512((v0), (v1), (v2)); \
    InOut_POS_6_512((out0), (v0), (in1))

#define Compute_Inout_Pos_7_Vector_512(v0, v1, v2, out0, in1) \
    Compute_1vector_512((v0), (v1), (v2)); \
    InOut_POS_7_512((out0), (v0), (in1))

#define PROCESS_POS(pos, _v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7, _v8, _v9, out0, in0, in1, in2, in3, in4, in5, in6, in7, tail) \
    Shift_right_XSLOPE_vectors_512((_v8), (_v9)); \
    Compute_Inout_Pos_##pos##_Vector_512((_v0), (_v1), (_v2), (out0), (in0)); \
    Shift_left_XSLOPE_vectors_512((_v0), (tail)); \
    Compute_Inout_Pos_##pos##_Vector_512((_v1), (_v2), (_v3), (in0), (in1)); \
    Compute_Inout_Pos_##pos##_Vector_512((_v2), (_v3), (_v4), (in1), (in2)); \
    Compute_Inout_Pos_##pos##_Vector_512((_v3), (_v4), (_v5), (in2), (in3)); \
    Compute_Inout_Pos_##pos##_Vector_512((_v4), (_v5), (_v6), (in3), (in4)); \
    Compute_Inout_Pos_##pos##_Vector_512((_v5), (_v6), (_v7), (in4), (in5)); \
    Compute_Inout_Pos_##pos##_Vector_512((_v6), (_v7), (_v8), (in5), (in6)); \
    Compute_Inout_Pos_##pos##_Vector_512((_v7), (_v8), (tail), (in6), (in7))

#define Load_8_Vectors_512(addr, v0, v1, v2, v3, v4, v5, v6, v7) \
    (v0) = _mm512_loadu_pd((addr)); \
    (v1) = _mm512_loadu_pd((addr) + VVECLEN); \
    (v2) = _mm512_loadu_pd((addr) + 2 * VVECLEN); \
    (v3) = _mm512_loadu_pd((addr) + 3 * VVECLEN); \
    (v4) = _mm512_loadu_pd((addr) + 4 * VVECLEN); \
    (v5) = _mm512_loadu_pd((addr) + 5 * VVECLEN); \
    (v6) = _mm512_loadu_pd((addr) + 6 * VVECLEN); \
    (v7) = _mm512_loadu_pd((addr) + 7 * VVECLEN)

#define Store_8_Vectors_512(v0, v1, v2, v3, v4, v5, v6, v7, addr) \
    _mm512_storeu_pd((addr), (v0)); \
    _mm512_storeu_pd((addr) + VVECLEN, (v1)); \
    _mm512_storeu_pd((addr) + 2 * VVECLEN, (v2)); \
    _mm512_storeu_pd((addr) + 3 * VVECLEN, (v3)); \
    _mm512_storeu_pd((addr) + 4 * VVECLEN, (v4)); \
    _mm512_storeu_pd((addr) + 5 * VVECLEN, (v5)); \
    _mm512_storeu_pd((addr) + 6 * VVECLEN, (v6)); \
    _mm512_storeu_pd((addr) + 7 * VVECLEN, (v7))

#define Shift_left_XSLOPE_vectors_512(v0, v1) \
    (v1) = vrotate_512_low2high(v0)

#define Shift_right_XSLOPE_vectors_512(v0, v1) \
    (v1) = vrotate_512_high2low(v0)


//==vectime_stride7_512_inreg=============================================

// 将原本 7 次串行 in rotate 改成一次性生成 lane0 视图。
// inN 消费后会被复用为对应 vN 的输出候选。
#define Input_Output_inreg_prepare_512(in0, in1, in2, in3, in4, in5, in6, in7) \
    (in2) = _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(in0), _mm512_castpd_si512(in0), 2)); \
    (in4) = _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(in0), _mm512_castpd_si512(in0), 4)); \
    (in6) = _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(in0), _mm512_castpd_si512(in0), 6)); \
    (in1) = _mm512_permute_pd((in0), 0x55); \
    (in3) = _mm512_permute_pd((in2), 0x55); \
    (in5) = _mm512_permute_pd((in4), 0x55); \
    (in7) = _mm512_permute_pd((in6), 0x55)

#define Input_Output_inreg_512(tmp, v, in_src) \
    (tmp) = vrotate_512_high2low(v); \
    (v) = _mm512_mask_blend_pd(0x01, (tmp), (in_src)); \
    (in_src) = (tmp)

#define Input_Output_inreg_4_512(out, out0, out1, out2, out3) \
    { \
        __m128d _out01 = _mm_unpacklo_pd(_mm512_castpd512_pd128(out0), _mm512_castpd512_pd128(out1)); \
        __m128d _out23 = _mm_unpacklo_pd(_mm512_castpd512_pd128(out2), _mm512_castpd512_pd128(out3)); \
        (out) = _mm256_castpd128_pd256(_out01); \
        (out) = _mm256_insertf128_pd((out), _out23, 1); \
    }

#define Input_Output_inreg_8_512(out, out_lo, out_hi) \
    (out) = _mm512_castpd256_pd512(out_lo); \
    (out) = _mm512_insertf64x4((out), (out_hi), 1)


//==vectime_transpose_inout_512_inreg=======================================

// inreg variant of PROCESS_POS: batch 7 of 8 independent stencil computes, with
// v0 chain serial to produce correct tail for v7's stencil (tail = rotate(InOut(v0))).
#define PROCESS_POS_inreg(pos, _v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7, _v8, _v9, out0, in0, in1, in2, in3, in4, in5, in6, in7, tail) \
    Shift_right_XSLOPE_vectors_512((_v8), (_v9)); \
    /* v0 chain (serial): compute → InOut → tail capture */ \
    Compute_1vector_512((_v0), (_v1), (_v2)); \
    InOut_POS_##pos##_512((out0), (_v0), (in0)); \
    Shift_left_XSLOPE_vectors_512((_v0), (tail)); \
    /* Phase 1: batch remaining 7 independent stencil computations */ \
    Compute_1vector_512((_v1), (_v2), (_v3)); \
    Compute_1vector_512((_v2), (_v3), (_v4)); \
    Compute_1vector_512((_v3), (_v4), (_v5)); \
    Compute_1vector_512((_v4), (_v5), (_v6)); \
    Compute_1vector_512((_v5), (_v6), (_v7)); \
    Compute_1vector_512((_v6), (_v7), (_v8)); \
    Compute_1vector_512((_v7), (_v8), (tail)); \
    /* Phase 2: batch remaining 7 InOut lane operations */ \
    InOut_POS_##pos##_512((in0),  (_v1), (in1)); \
    InOut_POS_##pos##_512((in1),  (_v2), (in2)); \
    InOut_POS_##pos##_512((in2),  (_v3), (in3)); \
    InOut_POS_##pos##_512((in3),  (_v4), (in4)); \
    InOut_POS_##pos##_512((in4),  (_v5), (in5)); \
    InOut_POS_##pos##_512((in5),  (_v6), (in6)); \
    InOut_POS_##pos##_512((in6),  (_v7), (in7))




void multiload_512_unroll1(double *A, int NX, int T);
void multiload_512_unroll2(double *A, int NX, int T);
void multiload_512_unroll4(double *A, int NX, int T);
void reg_assemble_512_unroll1(double *A, int NX, int T);
void reg_assemble_512_unroll2(double *A, int NX, int T);
void reg_assemble_512_unroll4(double *A, int NX, int T);
void dlt_512_unroll1(double *A, int NX, int T);
void dlt_512_unroll2(double *A, int NX, int T);
void dlt_512_unroll4(double *A, int NX, int T);
void vectime_stride7_512(double *A, int NX, int T);
void vectime_stride7_512_inreg(double *A, int NX, int T);
void vectime_stride7_512_hybrid(double *A, int NX, int T);
void vectime_stride7_512_hybridReg(double *A, int NX, int T);
void vectime_stride7_512_reg_hybrid(double *A, int NX, int T);
void vectime_stride7_512_inreg_hybrid(double *A, int NX, int T);
void vectime_stride15_512(double *A, int NX, int T);
void vectime_stride15_512_inreg(double *A, int NX, int T);
void vectime_stride15_512_hybrid(double *A, int NX, int T);
void vectime_stride15_512_inreg_hybrid(double *A, int NX, int T);
void vectime_stride7_midstride2_512(double *A, int NX, int T);
void vectime_transpose_inout_512(double *A, int NX, int T);
void vectime_transpose_inout_512_inreg(double *A, int NX, int T);
void vectime_transpose_inout_512_hybrid(double *A, int NX, int T);
void vectime_transpose_inout_512_inreg_hybrid(double *A, int NX, int T);
