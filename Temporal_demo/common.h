#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#include <immintrin.h>
//  #include <avx2intrin.h>
#include "omp.h"
#include <string.h>

#define VECLEN 4
#define VVECLEN 8

#ifdef __linux__ 
#include <malloc.h>
#define	alloc_extra_array(a) _mm_malloc(a, 64) 
#define free_extra_array(a) _mm_free(a)
#else
#define	alloc_extra_array(a) malloc(a) 
#define free_extra_array(a) free(a)
#endif

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))
#define myabs(x,y)  ((x) > (y)? ((x)-(y)) : ((y)-(x))) 
#define myceil(x,y)  (int)ceil(((double)x)/((double)y)) // if x and y are integers, myceil(x,y) = (x-1)/y + 1
#define myfloor(x,y)  (int)floor(((double)x)/((double)y)) // if x and y are integers, myceil(x,y) = (x-1)/y + 1

#define transpose4x4(a,b,c,d,i,j,in,out) 	out = _mm256_permute2f128_pd(b,d, 0x31); \
											j = _mm256_permute2f128_pd(a,c, 0x31); \
											i = _mm256_permute2f128_pd(a,c, 0x20); \
											in = _mm256_permute2f128_pd(b,d, 0x20); \
											d = _mm256_unpackhi_pd(j,out); \
											c = _mm256_unpacklo_pd(j,out); \
											b = _mm256_unpackhi_pd(i,in); \
											a = _mm256_unpacklo_pd(i,in)
#define transpose(a,b,c,d,i,j) 	i = _mm256_permute2f128_pd(a,c, 0x20); \
								j = _mm256_permute2f128_pd(a,c, 0x31); \
								a = _mm256_permute2f128_pd(b,d, 0x20); \
								c = _mm256_permute2f128_pd(b,d, 0x31); \
								d = _mm256_unpackhi_pd(j,c); \
								c = _mm256_unpacklo_pd(j,c); \
								b = _mm256_unpackhi_pd(i,a); \
								a = _mm256_unpacklo_pd(i,a)


#define transposei(r0, r1, r2, r3, r4, r5, r6, r7, t0, t1, t2, t3, t4, t5, t6, t7) t0 = _mm256_unpackhi_epi32(r0, r1);\
  t1 = _mm256_unpacklo_epi32(r0, r1);\
  t2 = _mm256_unpackhi_epi32(r2, r3);\
  t3 = _mm256_unpacklo_epi32(r2, r3);\
  t4 = _mm256_unpackhi_epi32(r4, r5);\
  t5 = _mm256_unpacklo_epi32(r4, r5);\
  t6 = _mm256_unpackhi_epi32(r6, r7);\
  t7 = _mm256_unpacklo_epi32(r6, r7);\
  r0 = (__m256i)_mm256_shuffle_ps((__m256)t0,(__m256)t2,_MM_SHUFFLE(1,0,1,0));  \
  r1 = (__m256i)_mm256_shuffle_ps((__m256)t0,(__m256)t2,_MM_SHUFFLE(3,2,3,2));\
  r2 = (__m256i)_mm256_shuffle_ps((__m256)t1,(__m256)t3,_MM_SHUFFLE(1,0,1,0));\
  r3 = (__m256i)_mm256_shuffle_ps((__m256)t1,(__m256)t3,_MM_SHUFFLE(3,2,3,2));\
  r4 = (__m256i)_mm256_shuffle_ps((__m256)t4,(__m256)t6,_MM_SHUFFLE(1,0,1,0));\
  r5 = (__m256i)_mm256_shuffle_ps((__m256)t4,(__m256)t6,_MM_SHUFFLE(3,2,3,2));\
  r6 = (__m256i)_mm256_shuffle_ps((__m256)t5,(__m256)t7,_MM_SHUFFLE(1,0,1,0));\
  r7 = (__m256i)_mm256_shuffle_ps((__m256)t5,(__m256)t7,_MM_SHUFFLE(3,2,3,2));\
  t2 = _mm256_permute2f128_si256(r0, r4, 0x20);\
  t3 = _mm256_permute2f128_si256(r1, r5, 0x20);\
  t0 = _mm256_permute2f128_si256(r2, r6, 0x20);\
  t1 = _mm256_permute2f128_si256(r3, r7, 0x20);\
  t6 = _mm256_permute2f128_si256(r0, r4, 0x31);\
  t7 = _mm256_permute2f128_si256(r1, r5, 0x31);\
  t4 = _mm256_permute2f128_si256(r2, r6, 0x31);\
  t5 = _mm256_permute2f128_si256(r3, r7, 0x31)


#define vload(a,b) a=_mm256_loadu_pd((&b))
#define vstore(a,b) _mm256_storeu_pd((&a),b)

#define vloada(a,b) a=_mm256_load_pd((&b))
#define vstorea(a,b) _mm256_store_pd((&a),b)

#define loadv(b) _mm256_loadu_pd(&b)
#define storev(a,b) _mm256_storeu_pd((&a),b)

#define vloadset(a,b,c,d,e) {a=_mm256_set_pd(b,c,d,e);}
#define vallset(a,b) {a = _mm256_set1_pd(b);}

#define vloadseti(v, B, t, x, y)             v = _mm256_set_epi32( B[ (t + 1)%2 ][ x         ][ y ], \
																B[ (t)%2  ][ x + STRIDE  ][ y ], \
																B[ (t + 1)%2 ][ x + STRIDE * 2 ][ y ], \
																B[ (t)%2 ]  [ x + STRIDE * 3 ][ y ], \
																B[ (t + 1)%2 ][ x + STRIDE * 4 ][ y ], \
																B[ (t)%2 ]  [ x + STRIDE * 5 ][ y ], \
																B[ (t + 1)%2 ][ x + STRIDE * 6 ][ y ], \
																B[ (t)%2  ][ x + STRIDE * 7 ][ y ]) 
#define vloadseti_blk(v, B, t, x, y)             v = _mm256_set_epi32(	B[ (t + 1)%2 ][ x         ][ y ], \
																		B[ (t)%2  ][ x + STRIDE  ][ y + 1 ], \
																		B[ (t + 1)%2 ][ x + STRIDE * 2 ][ y + 2 ], \
																		B[ (t)%2 ]  [ x + STRIDE * 3 ][ y + 3 ], \
																		B[ (t + 1)%2 ][ x + STRIDE * 4 ][ y + 4 ], \
																		B[ (t)%2 ]  [ x + STRIDE * 5 ][ y + 5 ], \
																		B[ (t + 1)%2 ][ x + STRIDE * 6 ][ y + 6 ], \
																		B[ (t)%2  ][ x + STRIDE * 7 ][ y + 7 ]) 
		#define vloadi(a) _mm256_loadu_si256((__m256i * )&a)
#define vloadi2(b,a) b=_mm256_loadu_si256((__m256i * )&a)
#define vstorei(a,b) _mm256_storeu_si256((__m256i * ) &a, b)
#define vrotate_high2lowi(a)  _mm256_permutevar8x32_epi32(a,vrotatei_high2low)


#define vrotate_low2high(a) _mm256_permute4x64_pd(a,0b00111001) //	a0 a1 a2 a3 --> a1 a2 a3 a0
#define vrotate_high2low(a) _mm256_permute4x64_pd(a,0b10010011) //	a0 a1 a2 a3 --> a3 a0 a1 a2


#define blend_3456(out, in2, in1) {out = _mm256_blend_pd(in1, in2, 0b1000); out = vrotate_high2low(out);} // out: in2[ 3 ] in1[ 0 ] in1[ 1 ] in1[ 2 ]
#define blend_upvec2(out, in1, in2) {out = _mm256_blend_pd(in1, in2, 0b1100); out = _mm256_permute4x64_pd(out, 0b01001110);} // out: in2[ 2 ] in2[ 3 ] in1[ 0 ] in1[ 1 ]
#define blend_1234(out, in1, in2) {out = _mm256_blend_pd(in1, in2, 0b0001); out = vrotate_low2high(out);} // out: in1[ 1 ] in1[ 2 ] in1[ 3 ] in2[ 0 ]
#define blend_downvec2(out, in1, in2) {out = _mm256_blend_pd(in1, in2, 0b0011); out = _mm256_permute4x64_pd(out, 0b01001110);} // out: in1[ 2 ] in1[ 3 ] in2[ 0 ] in2[ 1 ]

#define Output_1(out, v1) out = v1
#define Output_2(out, v1) out = _mm256_shuffle_pd(out, v1, 0b0000)
#define Output_3(out, v1) out = _mm256_blend_pd(out, v1, 0b1000); out = _mm256_permute_pd(out, 0b0110)
#define Output_4(out, v1) out = _mm256_blend_pd(out,v1,0b1000)


#define Input_1(v1, v2, in) 			v1 = vrotate_high2low(v2);\
										v1 = _mm256_blend_pd(v1, in, 0b0001);\
										in = _mm256_permute_pd(in, 0b0101)
#define Input_2(v1, v2, in)				v1 = vrotate_high2low(v2);\
										v1 = _mm256_blend_pd(v1, in, 0b0001)
#define Input_3(v1, v2, in)				v1 = _mm256_blend_pd(v2, in, 0b1000);\
										v1 = vrotate_high2low(v1);	\
										in = _mm256_permute_pd(in, 0b0101)
#define Input_4(v1, v2, in)				v1 = _mm256_blend_pd(v2, in, 0b1000);\
										v1 = vrotate_high2low(v1)

#define	Input_Output_1(out,v1,in)		v1 = vrotate_high2low(v1);\
										Output_1(out, v1);\
										v1 = _mm256_blend_pd(v1, in, 0b0001);\
										in = _mm256_permute_pd(in, 0b0101)
#define	Input_Output_2(out,v1,in)		v1 = vrotate_high2low(v1);\
										Output_2(out, v1);\
										v1 = _mm256_blend_pd(v1, in, 0b0001)
#define	Input_Output_3(out,v1,in)		Output_3(out, v1); Input_3(v1, v1, in)
#define	Input_Output_4(out,v1,in)		Output_4(out, v1); Input_4(v1, v1, in)


#define	In_Out(v1, a)                	vstore(BV3[(y - YSTART) + a][1], v1);\
                						B[(t)%2][x][y + a] = BV3[(y - YSTART) + a + 1][0];\
                						BV3[(y - YSTART) + a][0] = B[(t)%2][x+STRIDE * VECLEN][y + a]


#define Input_high_1(v1, v2, in) 		v1 = _mm256_blend_pd(v2, in, 0b0001); \
										v1 = vrotate_low2high(v1); \
										in = _mm256_permute_pd(in, 0b0101)

#define Input_high_2(v1, v2, in) 		v1 = _mm256_blend_pd(v2, in, 0b0001);\
										v1 = vrotate_low2high(v1)

#define Input_high_3(v1, v2, in) 		v1 = vrotate_low2high(v2);\
										v1 = _mm256_blend_pd(v1, in, 0b1000);\
										in = _mm256_permute_pd(in, 0b0101)

#define Input_high_4(v1, v2, in) 		v1 = vrotate_low2high(v2);\
										v1 = _mm256_blend_pd(v1, in, 0b1000)


#define	Input_Output_keep_src_1(out,v1,v2,in)		v1 = vrotate_high2low(v2);\
													Output_1(out, v1);\
													v1 = _mm256_blend_pd(v1, in, 0b0001);\
													in = _mm256_permute_pd(in, 0b0101)

#define	Input_Output_keep_src_2(out,v1,v2,in)		v1 = vrotate_high2low(v2);\
													Output_2(out, v1);\
													v1 = _mm256_blend_pd(v1, in, 0b0001)

#define	Input_Output_keep_src_3(out,v1,v2,in)		Output_3(out, v2); Input_3(v1, v2, in)
										

#define	Input_Output_keep_src_4(out,v1,v2,in)		Output_4(out, v2); Input_4(v1, v2, in)


#define Input_Output_double_strides_02(out, v0, in)			out = v0;\
															v0 = _mm256_shuffle_pd(in, v0, 0b0000)

#define Input_Output_double_strides_13(out, v0, in)			out = _mm256_shuffle_pd(out, v0, 0b1111);\
															v0 = _mm256_shuffle_pd(in, v0, 0b0101)


#define Output_i_1(out, v1) out =   v1; \
                            out =   _mm256_shuffle_epi32(out, 0b00111001)//----0---

#define Output_i_2(out, v1) out =   _mm256_blend_epi32(out, v1, 0b00000001); \
                            out =   _mm256_shuffle_epi32(out, 0b00111001)//----10--

#define Output_i_3(out, v1) out =   _mm256_blend_epi32(out, v1, 0b00000001); \
                            out =   _mm256_shuffle_epi32(out, 0b00111001)// ----210-

#define Output_i_4(out, v1) out =   _mm256_blend_epi32(out, v1, 0b00000001); \
                            out =   _mm256_shuffle_epi32(out, 0b01001110)// ----0321

#define Output_i_5(out, v1) out =   _mm256_blend_epi32(out, v1, 0b10000000); \
                            out =   _mm256_shuffle_epi32(out, 0b00111001)// -4--1032

#define Output_i_6(out, v1) out =   _mm256_blend_epi32(out, v1, 0b10000000); \
                            out =   _mm256_shuffle_epi32(out, 0b00111001)// -54-2103

#define Output_i_7(out, v1) out =   _mm256_blend_epi32(out, v1, 0b10000000); \
                            out =   _mm256_shuffle_epi32(out, 0b00111001)// -6543210

#define Output_i_8(out, v1) out =   _mm256_blend_epi32(out, v1, 0b10000000)// 76543210




#define	Input_Output_i_1(out,v1,in)		v1 = vrotate_high2lowi(v1);\
										Output_i_1(out, v1);\
										v1 = _mm256_blend_epi32(v1, in, 0b00000001);\
										in = _mm256_shuffle_epi32(in, 0b00111001)
                                        
#define	Input_Output_i_2(out,v1,in)		v1 = vrotate_high2lowi(v1);\
										Output_i_2(out, v1);\
										v1 = _mm256_blend_epi32(v1, in, 0b00000001);\
										in = _mm256_shuffle_epi32(in, 0b00111001)

#define	Input_Output_i_3(out,v1,in)		v1 = vrotate_high2lowi(v1);\
										Output_i_3(out, v1);\
										v1 = _mm256_blend_epi32(v1, in, 0b00000001);\
										in = _mm256_shuffle_epi32(in, 0b00111001)

#define	Input_Output_i_4(out,v1,in)		v1 = vrotate_high2lowi(v1);\
										Output_i_4(out, v1);\
										v1 = _mm256_blend_epi32(v1, in, 0b00000001);\
										in = _mm256_shuffle_epi32(in, 0b01001110)

#define	Input_Output_i_5(out,v1,in)		Output_i_5(out, v1);\
										v1 = _mm256_blend_epi32(v1, in, 0b10000000);\
										in = _mm256_shuffle_epi32(in, 0b00111001);\
                                        v1 = vrotate_high2lowi(v1)

#define	Input_Output_i_6(out,v1,in)		Output_i_6(out, v1);\
										v1 = _mm256_blend_epi32(v1, in, 0b10000000);\
										in = _mm256_shuffle_epi32(in, 0b00111001);\
                                        v1 = vrotate_high2lowi(v1)

#define	Input_Output_i_7(out,v1,in)		Output_i_7(out, v1);\
										v1 = _mm256_blend_epi32(v1, in, 0b10000000);\
										in = _mm256_shuffle_epi32(in, 0b00111001);\
                                        v1 = vrotate_high2lowi(v1)

#define	Input_Output_i_8(out,v1,in)		Output_i_8(out, v1);\
										v1 = _mm256_blend_epi32(v1, in, 0b10000000);\
                                        v1 = vrotate_high2lowi(v1)





#define	Input_Output_i_keep_src_1(out,v2,v1,in)		v2 = vrotate_high2lowi(v1);\
                                                    Output_i_1(out, v2);\
                                                    v2 = _mm256_blend_epi32(v2, in, 0b00000001);\
                                                    in = _mm256_shuffle_epi32(in, 0b00111001)
                                        
#define	Input_Output_i_keep_src_2(out,v2,v1,in)		v2 = vrotate_high2lowi(v1);\
                                                    Output_i_2(out, v2);\
                                                    v2 = _mm256_blend_epi32(v2, in, 0b00000001);\
                                                    in = _mm256_shuffle_epi32(in, 0b00111001)

#define	Input_Output_i_keep_src_3(out,v2,v1,in)		v2 = vrotate_high2lowi(v1);\
                                                    Output_i_3(out, v2);\
                                                    v2 = _mm256_blend_epi32(v2, in, 0b00000001);\
                                                    in = _mm256_shuffle_epi32(in, 0b00111001)

#define	Input_Output_i_keep_src_4(out,v2,v1,in)		v2 = vrotate_high2lowi(v1);\
                                                    Output_i_4(out, v2);\
                                                    v2 = _mm256_blend_epi32(v2, in, 0b00000001);\
                                                    in = _mm256_shuffle_epi32(in, 0b01001110)

#define	Input_Output_i_keep_src_5(out,v2,v1,in)		Output_i_5(out, v1);\
                                                    v2 = _mm256_blend_epi32(v1, in, 0b10000000);\
                                                    in = _mm256_shuffle_epi32(in, 0b00111001);\
                                                    v2 = vrotate_high2lowi(v2)

#define	Input_Output_i_keep_src_6(out,v2,v1,in)		Output_i_6(out, v1);\
                                                    v2 = _mm256_blend_epi32(v1, in, 0b10000000);\
                                                    in = _mm256_shuffle_epi32(in, 0b00111001);\
                                                    v2 = vrotate_high2lowi(v2)

#define	Input_Output_i_keep_src_7(out,v2,v1,in)		Output_i_7(out, v1);\
                                                    v2 = _mm256_blend_epi32(v1, in, 0b10000000);\
                                                    in = _mm256_shuffle_epi32(in, 0b00111001);\
                                                    v2 = vrotate_high2lowi(v2)

#define	Input_Output_i_keep_src_8(out,v2,v1,in)		Output_i_8(out, v1);\
                                                    v2 = _mm256_blend_epi32(v1, in, 0b10000000);\
                                                    v2 = vrotate_high2lowi(v2)





#define	Input_i_1(v1,in)		    v1 = vrotate_high2lowi(v1);\
										v1 = _mm256_blend_epi32(v1, in, 0b00000001);\
										in = _mm256_shuffle_epi32(in, 0b00111001)
                                        
#define	Input_i_2(v1,in)		    v1 = vrotate_high2lowi(v1);\
										v1 = _mm256_blend_epi32(v1, in, 0b00000001);\
										in = _mm256_shuffle_epi32(in, 0b00111001)

#define	Input_i_3(v1,in)		    v1 = vrotate_high2lowi(v1);\
										v1 = _mm256_blend_epi32(v1, in, 0b00000001);\
										in = _mm256_shuffle_epi32(in, 0b00111001)

#define	Input_i_4(v1,in)		    v1 = vrotate_high2lowi(v1);\
										v1 = _mm256_blend_epi32(v1, in, 0b00000001);\
										in = _mm256_shuffle_epi32(in, 0b01001110)

#define	Input_i_5(v1,in)		    v1 = _mm256_blend_epi32(v1, in, 0b10000000);\
										in = _mm256_shuffle_epi32(in, 0b00111001);\
                                        v1 = vrotate_high2lowi(v1)

#define	Input_i_6(v1,in)		    v1 = _mm256_blend_epi32(v1, in, 0b10000000);\
										in = _mm256_shuffle_epi32(in, 0b00111001);\
                                        v1 = vrotate_high2lowi(v1)

#define	Input_i_7(v1,in)			v1 = _mm256_blend_epi32(v1, in, 0b10000000);\
										in = _mm256_shuffle_epi32(in, 0b00111001);\
                                        v1 = vrotate_high2lowi(v1)

#define	Input_i_8(v1,in)			v1 = _mm256_blend_epi32(v1, in, 0b10000000);\
                                        v1 = vrotate_high2lowi(v1)


#define setv_2d(x, y) 			_mm256_set_pd(	B[ (t + 1) % 2 ][ x + STRIDE * 0 ][ y ], \
												B[ (t) % 2 ]    [ x + STRIDE * 1 ][ y ], \
												B[ (t + 1) % 2 ][ x + STRIDE * 2 ][ y ], \
												B[ (t) % 2 ]    [ x + STRIDE * 3 ][ y ])

#define setv_2d_blk(x, y)		_mm256_set_pd(	B[ (t + 1) % 2 ][ x + STRIDE * 0 ][ y + 0 ], \
												B[ (t) % 2 ]    [ x + STRIDE * 1 ][ y + 1 ], \
												B[ (t + 1) % 2 ][ x + STRIDE * 2 ][ y + 2 ], \
												B[ (t) % 2 ]    [ x + STRIDE * 3 ][ y + 3 ])

#define setv_3d(x, y, z)		_mm256_set_pd(	B[ (t + 1) % 2 ][ x + STRIDE * 0 ][ y ][ z ], \
												B[ (t) % 2 ]    [ x + STRIDE * 1 ][ y ][ z ], \
												B[ (t + 1) % 2 ][ x + STRIDE * 2 ][ y ][ z ], \
												B[ (t) % 2 ]    [ x + STRIDE * 3 ][ y ][ z ])

#define setv_3d_blk(x, y, z)	_mm256_set_pd(	B[ (t + 1) % 2 ][ x + STRIDE * 0 ][ y + 0 ][ z + 0 ], \
												B[ (t) % 2 ]    [ x + STRIDE * 1 ][ y + 1 ][ z + 1 ], \
												B[ (t + 1) % 2 ][ x + STRIDE * 2 ][ y + 2 ][ z + 2 ], \
												B[ (t) % 2 ]    [ x + STRIDE * 3 ][ y + 3 ][ z + 3 ])

#define vset_2d(v, x, y) 		v = setv_2d(x, y)
#define vset_2d_blk(v, x, y) 	v = setv_2d_blk(x, y)
#define vset_3d(v, x, y, z) 	v = setv_3d(x, y, z)
#define vset_3d_blk(v, x, y, z) v = setv_3d_blk(x, y, z)

#define vstore_set_2d(x, y, v)				vstore(tmp[ 0 ], v);\
											B[ (t + 1)%2 ][ x + STRIDE * 3 ][ y ] = tmp[ 0 ];\
											B[ t%2       ][ x + STRIDE * 2 ][ y ] = tmp[ 1 ];\
											B[ (t + 1)%2 ][ x + STRIDE * 1 ][ y ] = tmp[ 2 ];\
											B[ t%2       ][ x + STRIDE * 0 ][ y ] = tmp[ 3 ]

#define vstore_set_2d_blk(x, y, v)			vstore(tmp[ 0 ], v);\
											B[ (t + 1)%2 ][ x + STRIDE * 3 ][ y + 3 ] = tmp[ 0 ];\
											B[ t%2       ][ x + STRIDE * 2 ][ y + 2 ] = tmp[ 1 ];\
											B[ (t + 1)%2 ][ x + STRIDE * 1 ][ y + 1 ] = tmp[ 2 ];\
											B[ t%2       ][ x + STRIDE * 0 ][ y + 0 ] = tmp[ 3 ]

#define vstore_set_2d_x_m_blk(x, y, v) 		vstore(tmp[ 0 ], v);\
											B[ (t)%2     ][ x + STRIDE * 3 ][ y + 3 ] = tmp[ 0 ];\
											B[ (t + 1)%2 ][ x + STRIDE * 2 ][ y + 2 ] = tmp[ 1 ];\
											B[ (t)%2     ][ x + STRIDE * 1 ][ y + 1 ] = tmp[ 2 ];\
											B[ (t + 1)%2 ][ x + STRIDE * 0 ][ y + 0 ] = tmp[ 3 ]

#define vstore_set_3d(x, y, z, v)	 	   	vstore(tmp[ 0 ], v);\
											B[ (t + 1)%2 ][ x + STRIDE * 3 ][ y ][ z ] = tmp[ 0 ];\
											B[ t%2       ][ x + STRIDE * 2 ][ y ][ z ] = tmp[ 1 ];\
											B[ (t + 1)%2 ][ x + STRIDE * 1 ][ y ][ z ] = tmp[ 2 ];\
											B[ t%2       ][ x + STRIDE * 0 ][ y ][ z ] = tmp[ 3 ]

#define vstore_set_3d_blk(x, y, z, v)	  	vstore(tmp[ 0 ], v);\
											B[ (t + 1)%2 ][ x + STRIDE * 3 ][ y + 3 ][ z + 3 ] = tmp[ 0 ];\
											B[ t%2       ][ x + STRIDE * 2 ][ y + 2 ][ z + 2 ] = tmp[ 1 ];\
											B[ (t + 1)%2 ][ x + STRIDE * 1 ][ y + 1 ][ z + 1 ] = tmp[ 2 ];\
											B[ t%2       ][ x + STRIDE * 0 ][ y + 0 ][ z + 0 ] = tmp[ 3 ]


#define loadv_2d_x_m(y) 		loadv(BV0[ y - YSTART ][ 0 ])
#define loadv_2d_x_c(y)			loadv(BV1[ y - YSTART ][ 0 ])
#define loadv_2d_x_p(y) 		loadv(BV2[ y - YSTART ][ 0 ])

#define loadv_2d_x_m_blk(y) 	loadv(BV0[ y - (myybeg - VECLEN + 1) ][ 0 ])
#define loadv_2d_x_p_blk(y) 	loadv(BV2[ y - (myybeg - VECLEN + 1) ][ 0 ])
#define loadv_2d_x_c_blk(y) 	loadv(BV1[ y - (myybeg - VECLEN + 1) ][ 0 ])

#define vstore_2d_x_pp_blk(v, y) vstore(BV3[y - (myybeg - VECLEN + 1) ][ 0 ], v)

#define loadv_3d_x_m(y, z) 		loadv(BV0[ y - YSTART ][ z - ZSTART ][ 0 ])
#define loadv_3d_x_c(y, z)		loadv(BV1[ y - YSTART ][ z - ZSTART ][ 0 ])
#define loadv_3d_x_p(y, z) 		loadv(BV2[ y - YSTART ][ z - ZSTART ][ 0 ])

#define loadv_3d_x_m_blk(y, z) 	loadv(BV0[ y - (myybeg - VECLEN + 1) ][ z - (myzbeg - VECLEN + 1) ][ 0 ])
#define loadv_3d_x_p_blk(y, z) 	loadv(BV2[ y - (myybeg - VECLEN + 1) ][ z - (myzbeg - VECLEN + 1) ][ 0 ])
#define loadv_3d_x_c_blk(y, z) 	loadv(BV1[ y - (myybeg - VECLEN + 1) ][ z - (myzbeg - VECLEN + 1) ][ 0 ])

#define vstore_3d_x_pp_blk(v, y, z) vstore(BV3[y - (myybeg - VECLEN + 1) ][ z - (myzbeg - VECLEN + 1) ][ 0 ], v)







#define load_x_m_blocking(x, y, z, yshift, zshift) loadv_3d_x_m_blk((y) + (yshift), (z) + (zshift))
#define load_x_p_blocking(x, y, z, yshift, zshift) loadv_3d_x_p_blk((y) + (yshift), (z) + (zshift))
#define load_x_c_blocking(x, y, z, yshift, zshift) loadv_3d_x_c_blk((y) + (yshift), (z) + (zshift))


#define	load_v(x, y, z, xshift, yshift, zshift)	setv_3d_blk((x) + (xshift), (y) + (yshift), (z) + (zshift))
#define store_v(v, x, y, z, xshift, yshift, zshift)      vstore_set_3d_blk((x) + (xshift), (y) + (yshift), (z) + (zshift), v)


#define loadv_x_m_3d(y, z) loadv_3d_x_m(y, z)
#define loadv_x_p_3d(y, z) loadv_3d_x_p(y, z)
#define loadv_x_c_3d(y, z) loadv_3d_x_c(y, z)

#define load_x_m(x, y, z, yshift, zshift) loadv_3d_x_m(yshift + y + YSLOPE, zshift + z)
#define load_x_p(x, y, z, yshift, zshift) loadv_3d_x_p(yshift + y + YSLOPE, zshift + z)
#define load_x_c(x, y, z, yshift, zshift) loadv_3d_x_c(yshift + y + YSLOPE, zshift + z)

#define store_x_pp(v, x, y, z, yshift, zshift) vstore_3d_x_pp_blk(v, (y) + (yshift), (z) + (zshift))




//#define shuffle(a,b,c) { a = _mm256_shuffle_pd(a, b, c);}
