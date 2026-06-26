#include "../common.h"

#define XSTART 4
#define XXSTART 8
#define XSLOPE 1

#define STRIDE2 2
#define STRIDE3 3
#define STRIDE4 4
#define STRIDE 7
#define STRIDE11 11
#define STRIDE15 15
#define LANESTRIDE 2
#define LANESTRIDE4 4

#define INIT 1.0 * (rand() % 1024)

#if defined(heat)
#define C0 0.50
#define C1 -2.0
#define Compute_scalar(A, t, x) A[(t+1)%2][x] = C0 * ((A[t%2][x+1] +  C1 * A[t%2][x]) + A[t%2][x-1])
#define Compute_1vector(v0,v1,v2)  v0=_mm256_mul_pd(vc0,_mm256_add_pd(_mm256_fmadd_pd(vc1, v1, v2),v0))
#define Compute_1vector_tmp(out, v0, v1, v2) \
   						 out = _mm256_mul_pd(vc0, _mm256_add_pd(_mm256_fmadd_pd(vc1, v1, v2), v0))

#else

#define C0 0.25
#define C1 0.75

#define Compute_scalar(A, t, x) A[(t+1)%2][x] = C0 * A[t%2][x] +  C1 * (A[t%2][x+1] + A[t%2][x-1])
#define Compute_1vector(v0,v1,v2)  v0 = _mm256_fmadd_pd(v1,vc0,_mm256_mul_pd(_mm256_add_pd(v0,v2),vc1))
#define Compute_1vector_tmp(out, v0, v1, v2) \
					   out = _mm256_fmadd_pd(v1, vc0, _mm256_mul_pd(_mm256_add_pd(v0, v2), vc1))//向量化stencil，但不就地修改

#endif

#define SET_COFF __m256d vc0 = _mm256_set1_pd(C0); __m256d vc1 = _mm256_set1_pd(C1)

int checkresult(int NX, double * A_correct, double * A) ;
void naive_scalar(double * A, int NX, int T);
void naive_vector(double * A, int NX, int T);
void vectime(double * A, int NX, int T);
void vectime_stride3(double * A, int NX, int T);
void vectime_stride3_unroll2(double * A, int NX, int T);
void vectime_stride7(double* A, int NX, int T);
void vectime_stride11(double* A, int NX, int T);
void vectime_stride7_midstride2(double* A, int NX, int T);
void vectime_transpose_inout(double* A, int NX, int T);
void dlt_unroll1(double* A, int NX, int T);
void dlt_unroll2(double* A, int NX, int T);
void dlt_unroll4(double* A, int NX, int T);
void multiload_unroll1(double* A, int NX, int T);
void multiload_unroll2(double* A, int NX, int T);
void multiload_unroll4(double* A, int NX, int T);
void reg_assemble_unroll1(double* A, int NX, int T);
void reg_assemble_unroll2(double* A, int NX, int T);
void reg_assemble_unroll4(double* A, int NX, int T);



/*------------------------------*/
// MergeTop
#define MergeTop(vtop, v3, k)                     \
do {                                              \
    __m256d tmp = Broadcast_high(v3);            \
    switch ((k) & 3) {                           \
        case 0: vtop = _mm256_blend_pd(vtop, tmp, 0b0001); break; \
        case 1: vtop = _mm256_blend_pd(vtop, tmp, 0b0010); break; \
        case 2: vtop = _mm256_blend_pd(vtop, tmp, 0b0100); break; \
        case 3: vtop = _mm256_blend_pd(vtop, tmp, 0b1000); break; \
    }                                            \
} while(0)

// MergeDown：合并 v_down 和 v3 到 v2
#define MergeDown(v3, vdown, v2)                                   \
    __m256d _md_v3r = rotate_right_1(v3);                           \
    __m256d _md_b   = Broadcast_low(vdown);                         \
    (v2) = _mm256_blend_pd(_md_v3r, _md_b, 0b0001);                 \
    (vdown) = rotate_left_1(vdown)


#define rotate_left_1(a) \
        				_mm256_permute4x64_pd((a), 0b00111001) // a0 a1 a2 a3 → a1 a2 a3 a0（左移） 
#define rotate_right_1(a) \
 				       _mm256_permute4x64_pd((a), 0b10010011) // a0 a1 a2 a3 → a3 a0 a1 a2（右移）

#define Broadcast_low(a) \
    				    _mm256_permute4x64_pd((a), 0b00000000)// 把向量最低位广播到整个向量
		
#define Broadcast_high(a) \
    _mm256_permute4x64_pd((a), 0b11111111)  // 广播最高位


#define Compute_4vector(v0,v1,v2,v3,v4,v5)  Compute_1vector(v0,v1,v2);\
											Compute_1vector(v1,v2,v3);\
											Compute_1vector(v2,v3,v4);\
											Compute_1vector(v3,v4,v5) 


#define Compute_3vector(v0,v1,v2,v3,v4) Compute_1vector(v0,v1,v2);\
										Compute_1vector(v1,v2,v3);\
										Compute_1vector(v2,v3,v4)



//将v的通道0复制到out的通道0，将in的通道0复制到v的通道0
#define InOut_POS_0(out, v, in)	out = _mm256_blend_pd(out, v, 0b0001); v = _mm256_blend_pd(v, in, 0b0001)
#define InOut_POS_1(out, v, in)	out = _mm256_blend_pd(out, v, 0b0010); v = _mm256_blend_pd(v, in, 0b0010)
#define InOut_POS_2(out, v, in)	out = _mm256_blend_pd(out, v, 0b0100); v = _mm256_blend_pd(v, in, 0b0100)
#define InOut_POS_3(out, v, in)	out = _mm256_blend_pd(out, v, 0b1000); v = _mm256_blend_pd(v, in, 0b1000)


#define Load_4_Vectors(addr, in0,in1,in2,in3)	in0 = _mm256_loadu_pd(addr);\
													in1 = _mm256_loadu_pd(addr + VECLEN);\
													in2 = _mm256_loadu_pd(addr + 2 * VECLEN);\
													in3 = _mm256_loadu_pd(addr + 3 * VECLEN)

#define Store_4_Vectors(out0,in0,in1,in2,addr)	_mm256_storeu_pd(addr,out0);\
													_mm256_storeu_pd(addr + VECLEN,in0);\
													_mm256_storeu_pd(addr + 2 * VECLEN,in1);\
													_mm256_storeu_pd(addr + 3 * VECLEN,in2)

//用 v0/v1/v2 执行一次 stencil，按位置把 v0 的第 0 个位置写入 out0， 用来自 in1 的新数据补充回 v0 的第 0 个位置
#define Compute_Inout_Pos_0_Vector(v0,v1,v2,out0,in1)			Compute_1vector(v0,v1,v2);\
													InOut_POS_0(out0, v0, in1)
#define Compute_Inout_Pos_1_Vector(v0,v1,v2,out0,in1)			Compute_1vector(v0,v1,v2);\
													InOut_POS_1(out0, v0, in1)
													
#define Compute_Inout_Pos_2_Vector(v0,v1,v2,out0,in1)			Compute_1vector(v0,v1,v2);\
													InOut_POS_2(out0, v0, in1)
#define Compute_Inout_Pos_3_Vector(v0,v1,v2,out0,in1)			Compute_1vector(v0,v1,v2);\
													InOut_POS_3(out0, v0, in1)									




#define Load_STRIDE_Vectors(addr, in0,in1,in2,in3)	Load_4_Vectors(addr, in0,in1,in2,in3)

#define Store_STRIDE_Vectors(out0,in0,in1,in2,addr)	Store_4_Vectors(out0,in0,in1,in2,addr)

#define Compute_STRIDE_Minus_XSLOPE_Vectors(v0,v1,v2,v3,v4) Compute_3vector(v0,v1,v2,v3,v4)
#define Compute_XSLOPE_vectors(v0,v1,v2) 					Compute_1vector(v0,v1,v2)

#define Shift_left_XSLOPE_vectors(v0,v1)  v1 = _mm256_permute4x64_pd(v0,0b00111001) //	a0 a1 a2 a3 --> a1 a2 a3 a0
#define Shift_right_XSLOPE_vectors(v0,v1) v1 = _mm256_permute4x64_pd(v0,0b10010011) //	a0 a1 a2 a3 --> a3 a0 a1 a2

#define InOut_Pos_0_STRIDE_Minus_XSLOPE_Vectors(v0,v1,v2)	InOut_POS_0(out0, v0, in0);\
															InOut_POS_0(in0, v1, in1);\
															InOut_POS_0(in1, v2, in2)

#define InOut_POS_0_XSLOPE_Vectors(v0)						InOut_POS_0(in2, v0, in3)


#define InOut_Pos_1_STRIDE_Minus_XSLOPE_Vectors(v0,v1,v2)	InOut_POS_1(out0, v0, in0);\
															InOut_POS_1(in0, v1, in1);\
															InOut_POS_1(in1, v2, in2)

#define InOut_POS_1_XSLOPE_Vectors(v0)						InOut_POS_1(in2, v0, in3)




#define InOut_Pos_2_STRIDE_Minus_XSLOPE_Vectors(v0,v1,v2)	InOut_POS_2(out0, v0, in0);\
															InOut_POS_2(in0, v1, in1);\
															InOut_POS_2(in1, v2, in2)

#define InOut_POS_2_XSLOPE_Vectors(v0)						InOut_POS_2(in2, v0, in3)




#define InOut_Pos_3_STRIDE_Minus_XSLOPE_Vectors(v0,v1,v2)	InOut_POS_3(out0, v0, in0);\
															InOut_POS_3(in0, v1, in1);\
															InOut_POS_3(in1, v2, in2)

#define InOut_POS_3_XSLOPE_Vectors(v0)						InOut_POS_3(in2, v0, in3)

