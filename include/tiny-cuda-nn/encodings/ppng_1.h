#pragma once

#include <tiny-cuda-nn/encodings/ppng.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

namespace tcnn {

template <typename T, uint32_t D, uint32_t C, uint32_t R>
__device__ void linear_interp(
    const T * __restrict__ features, // Fx2x3xCxQxR, but we only have to care about last 4 : 3CQR
    const uint32_t Q,
	const float sc[D],
	MatrixView<T> outputs, // BxF2C
	const uint32_t b,
	const uint32_t out_offset
) {
	float w[D];
	uint32_t p0[D];
	uint32_t p1[D];

	TCNN_PRAGMA_UNROLL
	for(uint32_t i = 0; i < D; i++){
		const float p = ((sc[i] + 1) * 0.5) * (Q -1);
		p0[i] = std::min(std::max((uint32_t)floor(p), (uint32_t) 0), Q - 1);
		p1[i] = std::max(std::min((uint32_t)ceil(p), Q-1), (uint32_t) 0);
		w[i] = p - (float)p0[i];
	}

	TCNN_PRAGMA_UNROLL
	for(uint32_t c = 0; c < C; c++){
		// iterate over rank
		float fs = 0;

		TCNN_PRAGMA_UNROLL
		for(uint32_t r = 0; r < R; r++){
			float f = 1;

			TCNN_PRAGMA_UNROLL
			for(uint32_t i = 0; i < D; i++){
				float f0 = (float)*(features + (i * C * Q * R) + (c * Q * R) + (p0[i] * R) + r);
				float f1 = (float)*(features + (i * C * Q * R) + (c * Q * R) + (p1[i] * R) + r);
				f *= (w[i] * f1) + ((1 - w[i]) * f0);
			}
			fs += f;
		}
		outputs(out_offset + c, b) = (T)fs;
	}
}


template <typename GRAD_T, typename T, uint32_t D, uint32_t C, uint32_t R>
__device__ void grad_linear_interp(
    GRAD_T * __restrict__ grad_features, // Fx2x3xCxQxR, but we only have to care about last 4 : 3CQR
    const T * __restrict__ features, // Fx2x3xCxQxR, but we only have to care about last 4 : 3CQR
    const uint32_t Q,
	const float sc[D],
	MatrixView<const T> grad_outputs, // BxF2C
	const uint32_t b,
	const uint32_t out_offset
) {
	float w[D];
	uint32_t p0[D];
	uint32_t p1[D];

	TCNN_PRAGMA_UNROLL
	for(uint32_t i = 0; i < D; i++){
		const float p = ((sc[i] + 1) * 0.5) * (Q -1);
		p0[i] = std::min(std::max((uint32_t)floor(p), (uint32_t) 0), Q - 1);
		p1[i] = std::max(std::min((uint32_t)ceil(p), Q-1), (uint32_t) 0);
		w[i] = p - (float)p0[i];
	}
	/**
	 * Given sx, query for interpolated values at that location
	 *
	 * for c in C
	 *   var fs = 0
	 *   for r in R
	 *     var f = 1
	 *     for i in D
	 *       f *= sample(sx)
	 *	   fs += f
	 *    results[c] = fs
	 */

	TCNN_PRAGMA_UNROLL
	for(int c = 0; c < C; c++){
		// iterate over rank
		float go = (float)grad_outputs(out_offset + c, b);

		float grad_cache[R*D];

		TCNN_PRAGMA_UNROLL
		for(int r = 0; r < R; r++){

			TCNN_PRAGMA_UNROLL
			for(int i = 0; i < D; i++){
				grad_cache[r * D + i] = 1;
			}

			TCNN_PRAGMA_UNROLL
			for(int i = 0; i < D; i++){
				// 3CQR
				float f0 = (float)*(features + (i * C * Q * R) + (c * Q * R) + (p0[i] * R) + r);
				float f1 = (float)*(features + (i * C * Q * R) + (c * Q * R) + (p1[i] * R) + r);
				float fa = (w[i] * f1) + ((1 - w[i]) * f0);
				for(int j = 0; j < D; j++){
					grad_cache[r * D + j] *= (i == j) ? 1 : fa;
				}
			}
		}

		TCNN_PRAGMA_UNROLL
		for(int i = 0; i < D; i++){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
			if (R > 1 && std::is_same<GRAD_T, __half>::value) {
				TCNN_PRAGMA_UNROLL
				for(uint32_t r = 0; r < R; r+=2){
					__half2 v0 = {
						(__half)(go * grad_cache[r*D + i] * (1 -w[i])),
						(__half)(go * grad_cache[(r + 1)*D + i] * (1 -w[i])),
					};
					__half2 v1 = {
						(__half)(go * grad_cache[r*D + i] * (w[i])),
						(__half)(go * grad_cache[(r + 1)*D + i] * (w[i])),
					};

					atomicAdd((__half2*)(grad_features + (i * C*Q*R) + (c * Q*R) + (p0[i] * R) + r), v0);
					atomicAdd((__half2*)(grad_features + (i * C*Q*R) + (c * Q*R) + (p1[i] * R) + r), v1);
				}
			} else
#endif
			TCNN_PRAGMA_UNROLL
			for(int r = 0; r < R; r++){
				atomicAdd(grad_features + (i * C*Q*R) + (c * Q*R) + (p0[i] * R) + r, (GRAD_T)(go * grad_cache[r*D + i] * (1 -w[i])));
				atomicAdd(grad_features + (i * C*Q*R) + (c * Q*R) + (p1[i] * R) + r, (GRAD_T)(go * grad_cache[r*D + i] * w[i]));
			}
		}
	}
}


template <typename T, uint32_t D, uint32_t C, uint32_t R>
__global__ void kernel_ppng_1_forward(
    const uint32_t B, // batch size
	const uint32_t F, // freq size
    const uint32_t Q, // quantization size
    const int32_t min_log2_freq,
    const int32_t max_log2_freq,
    const uint32_t P, // padding (not used)
    MatrixView<const float> points,      // Bx3
    const T * __restrict__ features,     // Fx2x3xCxQxR
	MatrixView<T> outputs             	 // BxF2C
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b>= B) return;
    const uint32_t f = blockIdx.y;
    const uint32_t s = blockIdx.z;
	const float freq_base = ((float)(f * (max_log2_freq - min_log2_freq))) / ((float) (F - 1)) + min_log2_freq;
    const float freq = powf(2.0, freq_base) * 3.1415926535;

	// skip freq / sc
    features += f*2*D*C*Q*R + s*D*C*Q*R;

	float sc[D];

	TCNN_PRAGMA_UNROLL
	for(int i = 0; i < D; i++){
		float p = points(i, b);
		sc[i] = sinf(freq * (p - 0.5) + s * M_HI);
	}
	linear_interp<T, D, C, R>(features, Q, sc, outputs, b, f*2*C + s*C);
}
template <typename GRAD_T, typename T, uint32_t D, uint32_t C, uint32_t R>
__global__ void kernel_ppng_1_backward(
    const uint32_t B, // batch size
	const uint32_t F, // freq size
    const uint32_t Q, // quantization size
    const int32_t min_log2_freq,
    const int32_t max_log2_freq,
    const uint32_t P, // padding (not used)
    MatrixView<const float> points,      // Bx3
    T * __restrict__ features,     // Fx2x3xCxQxR
    GRAD_T * __restrict__ grad_features,     // Fx2x3xCxQxR
    MatrixView<const T> grad_outputs      // BxF2C
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b>= B) return;
    const uint32_t f = blockIdx.y;
    const uint32_t s = blockIdx.z;
	const float freq_base = ((float)(f * (max_log2_freq - min_log2_freq))) / ((float) (F - 1)) + min_log2_freq;
    const float freq = powf(2.0, freq_base) * 3.1415926535;

	// skip freq / sc
    grad_features 	+= f*2*D*C*Q*R + s*D*C*Q*R;
    features 		+= f*2*D*C*Q*R + s*D*C*Q*R;

	float sc[D];

	TCNN_PRAGMA_UNROLL
	for(int i = 0; i < D; i++){
		float p = points(i, b);
		sc[i] = sinf(freq * (p - 0.5) + s * M_HI);
	}
	grad_linear_interp<GRAD_T, T, D, C, R>(grad_features, features, Q, sc, grad_outputs, b, f*2*C + s*C);
}

/////////////////////
// Class definition
/**
 * @brief PPNG1 encoding for ND inputs.
 * T: float or double
 * D: number of dimensions to encode (2 or 3)
 * C: number of features per frequency (1, 2, 4 or 8)
 * R: number of correlations per level (1, 2, 4 or 8) (Rank)
 */
template <typename T, uint32_t D, uint32_t C, uint32_t R>
class PPNG1 : public PPNG<T, D, C, R> {
#if TCNN_MIN_GPU_ARCH >= 62 || TCNN_MIN_GPU_ARCH == 60
	using grad_t = std::conditional_t<R == 1, float, T>;
#else
	using grad_t = float;
#endif
public:
	PPNG1(int32_t log2_min_freq,
		 int32_t log2_max_freq,
		 uint32_t n_quants,
		 uint32_t n_frequencies)
	: PPNG<T, D, C, R>(log2_min_freq, log2_max_freq, n_quants, n_frequencies)
	{
		this->m_n_params = this->m_n_frequencies * 2 * D * C * n_quants * R;
	}

	std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		auto forward = std::make_unique<Context>();

		if ((!output && !prepare_input_gradients) || this->padded_output_width() == 0) {
			return forward;
		}

		static constexpr uint32_t N_THREADS = 512;
		const dim3 blocks_ppng = { div_round_up(input.n(), N_THREADS), this->m_n_frequencies, 2};
		kernel_ppng_1_forward<T, D, C, R><<<blocks_ppng, N_THREADS, 0, stream>>>(
			input.n(), // B
			this->m_n_frequencies, // F
			this->m_n_quants, // Q
			this->m_log2_min_freq, // I
			this->m_log2_max_freq, // X
			this->m_n_to_pad, // P
			input.view(),
			use_inference_params ? this->inference_params() : this->params(),
			output->view()
		);

		return forward;
	}

	void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override {
        const uint32_t num_elements = input.n();
        if ((!dL_dinput && param_gradients_mode == GradientMode::Ignore) || this->padded_output_width() == 0 || num_elements == 0) {
			return;
		}

		static constexpr uint32_t N_THREADS = 512;
		const dim3 blocks_ppng = { div_round_up(input.n(), N_THREADS), this->m_n_frequencies, 2};


        // If not, accumulate in a temporary buffer and cast later.
		if(param_gradients_mode != GradientMode::Ignore){
			grad_t * grad_array;
			GPUMemoryArena::Allocation grad_array_tmp;
			if (!std::is_same<grad_t, T>::value) {
				grad_array_tmp = allocate_workspace(stream, this->m_n_params * sizeof(grad_t));
				grad_array = (grad_t*)grad_array_tmp.data();
			} else {
				grad_array = (grad_t*)this->gradients();
			}


			if (param_gradients_mode == GradientMode::Overwrite) {
				CUDA_CHECK_THROW(cudaMemsetAsync(grad_array, 0, this->m_n_params * sizeof(grad_t), stream));
			}


			kernel_ppng_1_backward<grad_t, T, D, C, R><<<blocks_ppng, N_THREADS, 0, stream>>>(
				input.n(), // B
				this->m_n_frequencies, // F
				this->m_n_quants, // Q
				this->m_log2_min_freq, // I
				this->m_log2_max_freq, // X
				this->m_n_to_pad, // P
				input.view(),
				use_inference_params ? this->inference_params() : this->params(),
				grad_array,
				dL_doutput.view()
			);

			if (!std::is_same<grad_t, T>::value) {
				parallel_for_gpu(stream, this->n_params(), [grad=this->gradients(), grad_tmp=grad_array] __device__ (size_t i) {
					grad[i] = (T)grad_tmp[i];
				});
			}
		}
	}
	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
		// Initialize the hashgrid from the GPU, because the number of parameters can be quite large.
		generate_random_uniform<float>(rnd, this->n_params(), params_full_precision, -0.7f * scale, 0.7f * scale);
	}

	std::string otype() const override {
		return "PPNG1";
	}
};

///////////////
// Templating

template <typename T, uint32_t D, uint32_t C>
Encoding<T>* create_ppng_1_encoding_with_dim_and_feat(const json& encoding) {

#define TCNN_PPNG_PARAMS \
	encoding.value("log2_min_freq", 0), \
	encoding.value("log2_max_freq", 6), \
	encoding.value("n_quants", 64u), \
	encoding.value("n_frequencies", 6u), \

	const uint32_t n_corrs = encoding.value("rank", 4u);

	switch (n_corrs) {
		// case 1: return new PPNG1<T, D, C, 1>{ TCNN_PPNG_PARAMS };
		case 2: return new PPNG1<T, D, C, 2>{ TCNN_PPNG_PARAMS };
		case 4: return new PPNG1<T, D, C, 4>{ TCNN_PPNG_PARAMS };
		case 8: return new PPNG1<T, D, C, 8>{ TCNN_PPNG_PARAMS };
		case 16: return new PPNG1<T, D, C, 16>{ TCNN_PPNG_PARAMS };
		default: throw std::runtime_error{"PPNG1: rank must be 1, 2, 4, 8 or 16"};
	}
#undef TCNN_PPNG_PARAMS
}

template <typename T, uint32_t D>
Encoding<T>* create_ppng_1_encoding_with_dim(const json& encoding) {
	const uint32_t n_feats = encoding.value("n_features", 4u);
	switch (n_feats) {
		// case 1: return create_ppng_1_encoding_with_dim_and_feat<T, D, 1>(encoding);
		case 2: return create_ppng_1_encoding_with_dim_and_feat<T, D, 2>(encoding);
		case 4: return create_ppng_1_encoding_with_dim_and_feat<T, D, 4>(encoding);
		case 8: return create_ppng_1_encoding_with_dim_and_feat<T, D, 8>(encoding);
		default: throw std::runtime_error{"PPNG1: number of features must be 1, 2, 4 or 8"};
	}
}

template <typename T>
Encoding<T>* create_ppng_1_encoding(uint32_t n_dims_to_encode, const json& encoding) {
	switch (n_dims_to_encode) {
		// case 2: return create_ppng_1_encoding_with_dim<T, 2>(encoding);
		case 3: return create_ppng_1_encoding_with_dim<T, 3>(encoding);
		default: throw std::runtime_error{"PPNG1: number of input dims must be 2 or 3"};
	}
}

}