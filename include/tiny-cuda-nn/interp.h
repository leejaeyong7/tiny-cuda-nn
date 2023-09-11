#pragma once
#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>
TCNN_NAMESPACE_BEGIN
static inline  __device__ uint32_t powu(const uint32_t base, const uint32_t exp) {
    uint32_t val = 1;
    for(int i = 0; i < exp; i++){
        val *= base;
    }
    return val;
}

template <typename T, uint32_t D, uint32_t C>
__device__ void nlinear_interp(
    const T * __restrict__ features, 
    const uint32_t R, 
	const float sc[D],
	MatrixView<T> outputs,
	const uint32_t b,
	const uint32_t out_offset
) {
	float w[D];
	float p0[D];
	float p1[D];
	uint32_t o[D];

	TCNN_PRAGMA_UNROLL
	for(uint32_t i = 0; i < D; i++){
		const float p = ((sc[i] + 1) * 0.5) * (R -1);
		p0[i] = min(max((uint32_t)floor(p), 0), R-1);
		p1[i] = max(min((uint32_t)ceil(p), R-1), 0);
		w[i] = p - p0[i];
		o[i] = powu(R, i);
	}

	float results[C] = {0};
	TCNN_PRAGMA_UNROLL
	for(int l = 0; l < 8; l++) {
		uint32_t offset = 0;
		float weight = 1;

		TCNN_PRAGMA_UNROLL
		for(int i = 0; i < D; i++){
			const uint32_t inv_i = D - i - 1;
			offset += (l & (1 << inv_i) ? p1[i] : p0[i]) * o[i];
			weight *= (l & (1 << inv_i) ? w[i] : 1 - w[i]);
		}

		TCNN_PRAGMA_UNROLL
		for(int c = 0; c < C; c++){
			results[c] += (float)(*(features + offset *C + c)) * weight;
		}
	}
	TCNN_PRAGMA_UNROLL
	for(int c = 0; c < C; c++){
		outputs(out_offset + c, b) = (T)results[c];
	}
}

template <typename GRAD_T, typename T, uint32_t D, uint32_t C>
__device__ void grad_nlinear_interp(
    GRAD_T * __restrict__ grad_features, 
    const uint32_t R, 
    const float sc[D], 
    MatrixView<const T> grad_output,
	const uint32_t b,
	const uint32_t out_offset 
){
	float w[D];
	float p0[D];
	float p1[D];
	uint32_t o[D];

	TCNN_PRAGMA_UNROLL
	for(uint32_t i = 0; i < D; i++){
		const float p = ((sc[i] + 1) * 0.5) * (R -1);
		p0[i] = min(max((uint32_t)floor(p), 0), R-1);
		p1[i] = max(min((uint32_t)ceil(p), R-1), 0);
		w[i] = p - p0[i];
		o[i] = powu(R, i);
	}

	TCNN_PRAGMA_UNROLL
	for(int l = 0; l < powu(2, D); l++) {
		uint32_t offset = 0;
		float weight = 1;

		TCNN_PRAGMA_UNROLL
		for(int i = 0; i < D; i++){
			const uint32_t inv_i = D - i - 1;
			offset += (l & (1 << inv_i) ? p1[i] : p0[i]) * o[i];
			weight *= (l & (1 << inv_i) ? w[i] : 1 - w[i]);
		}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (C > 1 && std::is_same<GRAD_T, __half>::value) {
			TCNN_PRAGMA_UNROLL
			for(uint32_t c = 0; c < C; c+=2){
				__half2 v = {
					(__half)((float)grad_output(out_offset + c, b) * weight),
					(__half)((float)grad_output(out_offset + c + 1, b) * weight)
				};
				atomicAdd((__half2*)(grad_features + offset * C + c), v);
			}
		} else
#endif
		{
			TCNN_PRAGMA_UNROLL
			for(int c = 0; c < C; c++){
				float go = (float)grad_output(out_offset + c, b);
				atomicAdd(grad_features + offset *C + c, (GRAD_T)(go * weight));
			}
		}
	}
}

TCNN_NAMESPACE_END