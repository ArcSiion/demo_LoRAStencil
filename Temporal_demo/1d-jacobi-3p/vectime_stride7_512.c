#include "defines_512.h"

void vectime_stride7_512(double* A, int NX, int T){

	double (* B)[NX + 2 * XSTART] = (double(*)[NX + 2 * XSTART]) A;

	int x, t = 0, tt, xx;

	__m512d v0, v1, v2, v3, v4, v5, v6, v7;
	__m512d out, in;	
	SET_COFF_512;

	for (tt = 0; tt <= T - VVECLEN; tt += VVECLEN){	

		for( t = tt ; t < tt + VVECLEN - 1 ; t++){
			#pragma vector always
			#pragma ivdep
			for ( x = XSTART; x < XSTART + STRIDE * (VVECLEN - 1 - (t - tt)) ; x++){
				Compute_scalar(B, t, x);	
			}
		}
		t = tt;

		x = XSTART - XSLOPE;

		// v7 是最旧的时间层 (x), v0 是最新的时间层 (x + 7*STRIDE)
		vload_512(v7, &B[(t + 1) % 2][x]);
		vload_512(v6, &B[(t) % 2]    [x + STRIDE]);
		vload_512(v5, &B[(t + 1) % 2][x + 2*STRIDE]);
		vload_512(v4, &B[(t) % 2]    [x + 3*STRIDE]);
		vload_512(v3, &B[(t + 1) % 2][x + 4*STRIDE]);
		vload_512(v2, &B[(t) % 2]    [x + 5*STRIDE]);
		vload_512(v1, &B[(t + 1) % 2][x + 6*STRIDE]);
		vload_512(v0, &B[(t) % 2]    [x + 7*STRIDE]);
	
		// 将空间向量转置为时间向量,lane0是时间的低位
		transpose8x8_pd(v0,v1,v2,v3,v4,v5,v6,v7);

		// Main loop: 处理中间平行四边形区域
		for ( x = XSTART; x < NX + XSTART - STRIDE * VVECLEN; x += STRIDE + XSLOPE){				
			// in 做 vdown 功能，加载新的输入数据
			vload_512(in, &B[t % 2][x + STRIDE * VVECLEN]);
			
			// 按照 stride3 的逻辑，分段计算并更新 IO
			// 第一组更新：v0-v3
			Compute_4vector_512(v0,v1,v2,v3,v4,v5);

			// 数据推进：IO 1-4
			Input_Output_1_512(out, v0, in);
			Input_Output_2_512(out, v1, in);
			Input_Output_3_512(out, v2, in);	
			Input_Output_4_512(out, v3, in);

			// 第二组更新：v4-v7
			Compute_4vector_512(v4,v5,v6,v7,v0,v1);

			// 数据推进：IO 5-8
			Input_Output_5_512(out, v4, in);
			Input_Output_6_512(out, v5, in);
			Input_Output_7_512(out, v6, in);	
			Input_Output_8_512(out, v7, in);

			// 存储 8 个 double 结果 (512-bit)
			_mm512_storeu_pd(&B[t % 2][x], out);				
		}

		// 转置回空间向量
		transpose8x8_pd(v0,v1,v2,v3,v4,v5,v6,v7);

		// Store back to memory (8 time steps x 8 spatial positions)
		vstore_512(&B[(t + 1) % 2][x - XSLOPE], v7);
		vstore_512(&B[(t) % 2]    [x - XSLOPE + STRIDE], v6);
		vstore_512(&B[(t + 1) % 2][x - XSLOPE + 2 * STRIDE], v5);
		vstore_512(&B[(t) % 2]    [x - XSLOPE + 3 * STRIDE], v4);
		vstore_512(&B[(t + 1) % 2][x - XSLOPE + 4 * STRIDE], v3);
		vstore_512(&B[(t) % 2]    [x - XSLOPE + 5 * STRIDE], v2);
		vstore_512(&B[(t + 1) % 2][x - XSLOPE + 6 * STRIDE], v1);
		// 注意: 最后一个不存，因为这个位置会在epilogue中计算

		// Epilogue: 处理后导三角区域
		xx = x + STRIDE * (VVECLEN - 1);
		for(t = tt ; t < tt + VVECLEN ; t++, xx -= STRIDE){
			#pragma vector always
			#pragma ivdep
			for ( x = xx; x < NX + XSTART; x ++){	
				Compute_scalar(B, t, x);
			}
		}
	}
	// 处理剩余时间层
	for (; t < T; t++){
		#pragma vector always
		#pragma ivdep
		for (x = XSTART; x < NX + XSTART; x++) {
			Compute_scalar(B, t, x);
		}
	}	
}

