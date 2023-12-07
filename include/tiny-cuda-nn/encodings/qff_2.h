#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/interp.h>
#include <tiny-cuda-nn/encodings/qff.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

template <typename T, uint32_t D, uint32_t C, uint32_t R>
__device__ void bilinear_interp(
    const T * __restrict__ features, // Fx2x3xCxQxQxR, but we only have to care about last 4 : 3CQQR
    const uint32_t Q,
	const float sc[D],
	MatrixView<T> outputs, // BxF2C
	const uint32_t b,
	const uint32_t out_offset
) {
	uint32_t p0[D];
	uint32_t p1[D];
	float w[D];

	TCNN_PRAGMA_UNROLL
	for(uint32_t i = 0; i < D; i++){
		const float p = ((sc[i] + 1) * 0.5) * (Q -1);
		p0[i] = min(max((uint32_t)floor(p), 0), Q - 1);
		p1[i] = max(min((uint32_t)ceil(p), Q-1), 0);
		w[i] = p - (float)p0[i];
	}

	TCNN_PRAGMA_UNROLL
	for(uint32_t c = 0; c < C; c++){
		// iterate over rank
		float fs = 0;

		TCNN_PRAGMA_UNROLL
		for(uint32_t r = 0; r < R; r++){
			float f = 1;

			// bilinear interpolation, so we want to interpolate between 4 points
			TCNN_PRAGMA_UNROLL
			for(uint32_t i = 0; i < D; i++){
				uint32_t jj = (i + 1) % D;
				uint32_t kk = (i + 2) % D;
				uint32_t j = min(jj, kk);
				uint32_t k = max(jj, kk);

				float f00 = (float)*(features + (i * C * Q * Q* R) + (c * Q * Q* R) + (p0[k] * Q* R) + (p0[j] * R)+r);
				float f01 = (float)*(features + (i * C * Q * Q* R) + (c * Q * Q* R) + (p0[k] * Q* R) + (p1[j] * R)+r);
				float f10 = (float)*(features + (i * C * Q * Q* R) + (c * Q * Q* R) + (p1[k] * Q* R) + (p0[j] * R)+r);
				float f11 = (float)*(features + (i * C * Q * Q* R) + (c * Q * Q* R) + (p1[k] * Q* R) + (p1[j] * R)+r);
				float w00 = (1 - w[j]) * (1 - w[k]);
				float w01 = w[j] * (1 - w[k]);
				float w10 = (1 - w[j]) * w[k];
				float w11 = w[j] * w[k];
				f += (w00 * f00) + (w01 * f01) + (w10 * f10) + (w11 * f11);
			}
			fs += f;
		}
		outputs(out_offset + c, b) = (T)fs;
	}
}


template <typename GRAD_T, typename T, uint32_t D, uint32_t C, uint32_t R>
__device__ void grad_bilinear_interp(
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
		p0[i] = min(max((uint32_t)floor(p), 0), Q - 1);
		p1[i] = max(min((uint32_t)ceil(p), Q-1), 0);
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

		TCNN_PRAGMA_UNROLL
		for(int i = 0; i < D; i++){
			uint32_t jj = (i + 1) % D;
			uint32_t kk = (i + 2) % D;
			uint32_t j = min(jj, kk);
			uint32_t k = max(jj, kk);
			uint32_t o00 = (i * C * Q * Q* R) + (c * Q * Q* R) + (p0[k] * Q* R) + (p0[j] * R);
			uint32_t o01 = (i * C * Q * Q* R) + (c * Q * Q* R) + (p0[k] * Q* R) + (p1[j] * R);
			uint32_t o10 = (i * C * Q * Q* R) + (c * Q * Q* R) + (p1[k] * Q* R) + (p0[j] * R);
			uint32_t o11 = (i * C * Q * Q* R) + (c * Q * Q* R) + (p1[k] * Q* R) + (p1[j] * R);
			float w00 = (1 - w[j]) * (1 - w[k]);
			float w01 = w[j] * (1 - w[k]);
			float w10 = (1 - w[j]) * w[k];
			float w11 = w[j] * w[k];

// atomicAdd(__half2) is only supported with compute capability 60 and above
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
			if (R > 1 && std::is_same<GRAD_T, __half>::value) {
				TCNN_PRAGMA_UNROLL
				for(uint32_t r = 0; r < R; r+=2){
					__half2 v00 = {
						(__half)(go * w00),
						(__half)(go * w00)
					};
					__half2 v01 = {
						(__half)(go * w01),
						(__half)(go * w01)
					};
					__half2 v10 = {
						(__half)(go * w10),
						(__half)(go * w10)
					};
					__half2 v11 = {
						(__half)(go * w11),
						(__half)(go * w11)
					};
					atomicAdd((__half2*)(grad_features + o00 + r), v00);
					atomicAdd((__half2*)(grad_features + o01 + r), v01);
					atomicAdd((__half2*)(grad_features + o10 + r), v10);
					atomicAdd((__half2*)(grad_features + o11 + r), v11);
				}
			} else
#endif

			TCNN_PRAGMA_UNROLL
			for(int r = 0; r < R; r++){
				atomicAdd(grad_features + o00 + r, (GRAD_T)(go * w00));
				atomicAdd(grad_features + o01 + r, (GRAD_T)(go * w01));
				atomicAdd(grad_features + o10 + r, (GRAD_T)(go * w10));
				atomicAdd(grad_features + o11 + r, (GRAD_T)(go * w11));
			}
		}
	}
}


template <typename T, uint32_t D, uint32_t C, uint32_t R>
__global__ void kernel_qff_2_forward(
    const uint32_t B, // batch size
	const uint32_t F, // freq size
    const uint32_t Q, // quantization size
    const uint32_t min_log2_freq,
    const uint32_t max_log2_freq,
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
    const float freq = powf(2.0, freq_base);

	// skip freq / sc
    features += f*2*D*C*Q*Q*R + s*D*C*Q*Q*R;

	float sc[D];

	TCNN_PRAGMA_UNROLL
	for(int i = 0; i < D; i++){
		float p = points(i, b);
		sc[i] = (s == 0) ? __sinf(freq * p) : __cosf(freq * p);
	}
	bilinear_interp<T, D, C, R>(features, Q, sc, outputs, b, f*2*C + s*C);
}
template <typename GRAD_T, typename T, uint32_t D, uint32_t C, uint32_t R>
__global__ void kernel_qff_2_backward(
    const uint32_t B, // batch size
	const uint32_t F, // freq size
    const uint32_t Q, // quantization size
    const uint32_t min_log2_freq,
    const uint32_t max_log2_freq,
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
    const float freq = powf(2.0, freq_base);

	// skip freq / sc
    grad_features 	+= f*2*D*C*Q*Q*R + s*D*C*Q*Q*R;
    features 		+= f*1*D*C*Q*Q*R + s*D*C*Q*Q*R;

	float sc[D];

	TCNN_PRAGMA_UNROLL
	for(int i = 0; i < D; i++){
		float p = points(i, b);
		sc[i] = (s == 0) ? __sinf(freq * p) : __cosf(freq * p);
	}
	grad_bilinear_interp<GRAD_T, T, D, C, R>(grad_features, features, Q, sc, grad_outputs, b, f*2*C + s*C);
}

/////////////////////
// Class definition
/**
 * @brief QFF2 encoding for ND inputs.
 * T: float or double
 * D: number of dimensions to encode (2 or 3)
 * C: number of features per frequency (1, 2, 4 or 8)
 * R: number of correlations per level (1, 2, 4 or 8) (Rank)
 */
template <typename T, uint32_t D, uint32_t C, uint32_t R>
class QFF2 : public QFF<T, D, C, R> {
#if TCNN_MIN_GPU_ARCH >= 62 || TCNN_MIN_GPU_ARCH == 60
	using grad_t = std::conditional_t<R == 1, float, T>;
#else
	using grad_t = float;
#endif
public:
	QFF2(uint32_t log2_min_freq,
		 uint32_t log2_max_freq,
		 uint32_t n_quants,
		 uint32_t n_frequencies)
	: QFF<T, D, C, R>(log2_min_freq, log2_max_freq, n_quants, n_frequencies)
	{
		this->m_n_params = this->m_n_frequencies * 2 * D * C * n_quants * n_quants * R;
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
		const dim3 blocks_qff = { div_round_up(input.n(), N_THREADS), this->m_n_frequencies, 2};
		kernel_qff_2_forward<T, D, C, R><<<blocks_qff, N_THREADS, 0, stream>>>(
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
		EGradientMode param_gradients_mode = EGradientMode::Overwrite
	) override {
        const uint32_t num_elements = input.n();
        if ((!dL_dinput && param_gradients_mode == EGradientMode::Ignore) || this->padded_output_width() == 0 || num_elements == 0) {
			return;
		}

		static constexpr uint32_t N_THREADS = 512;
		const dim3 blocks_qff = { div_round_up(input.n(), N_THREADS), this->m_n_frequencies, 2};


        // If not, accumulate in a temporary buffer and cast later.
		if(param_gradients_mode != EGradientMode::Ignore){
			grad_t * grad_array;
			GPUMemoryArena::Allocation grad_array_tmp;
			if (!std::is_same<grad_t, T>::value) {
				grad_array_tmp = allocate_workspace(stream, this->m_n_params * sizeof(grad_t));
				grad_array = (grad_t*)grad_array_tmp.data();
			} else {
				grad_array = (grad_t*)this->gradients();
			}


			if (param_gradients_mode == EGradientMode::Overwrite) {
				CUDA_CHECK_THROW(cudaMemsetAsync(grad_array, 0, this->m_n_params * sizeof(grad_t), stream));
			}


			kernel_qff_2_backward<grad_t, T, D, C, R><<<blocks_qff, N_THREADS, 0, stream>>>(
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
		generate_random_uniform<float>(rnd, this->n_params(), params_full_precision, -0.2f * scale, 0.2f * scale);
	}

	std::string otype() const override {
		return "QFF2";
	}
};

///////////////
// Templating

template <typename T, uint32_t D, uint32_t C>
Encoding<T>* create_qff_2_encoding_with_dim_and_feat(const json& encoding) {

#define TCNN_QFF_PARAMS \
	encoding.value("log2_min_freq", 0u), \
	encoding.value("log2_max_freq", 6u), \
	encoding.value("n_quants", 64u), \
	encoding.value("n_frequencies", 6u), \

	const uint32_t n_corrs = encoding.value("rank", 4u);

	switch (n_corrs) {
		// case 1: return new QFF1<T, D, C, 1>{ TCNN_QFF_PARAMS };
		case 2: return new QFF2<T, D, C, 2>{ TCNN_QFF_PARAMS };
		case 4: return new QFF2<T, D, C, 4>{ TCNN_QFF_PARAMS };
		case 8: return new QFF2<T, D, C, 8>{ TCNN_QFF_PARAMS };
		case 16: return new QFF2<T, D, C, 16>{ TCNN_QFF_PARAMS };
		default: throw std::runtime_error{"QFF1: rank must be 1, 2, 4, 8 or 16"};
	}
#undef TCNN_QFF_PARAMS
}

template <typename T, uint32_t D>
Encoding<T>* create_qff_2_encoding_with_dim(const json& encoding) {
	const uint32_t n_feats = encoding.value("n_features", 4u);
	switch (n_feats) {
		// case 1: return create_qff_2_encoding_with_dim_and_feat<T, D, 1>(encoding);
		case 2: return create_qff_2_encoding_with_dim_and_feat<T, D, 2>(encoding);
		case 4: return create_qff_2_encoding_with_dim_and_feat<T, D, 4>(encoding);
		case 8: return create_qff_2_encoding_with_dim_and_feat<T, D, 8>(encoding);
		default: throw std::runtime_error{"QFF1: number of features must be 1, 2, 4 or 8"};
	}
}

template <typename T>
Encoding<T>* create_qff_2_encoding(uint32_t n_dims_to_encode, const json& encoding) {
	switch (n_dims_to_encode) {
		// case 2: return create_qff_2_encoding_with_dim<T, 2>(encoding);
		case 3: return create_qff_2_encoding_with_dim<T, 3>(encoding);
		default: throw std::runtime_error{"QFF1: number of input dims must be 2 or 3"};
	}
}

TCNN_NAMESPACE_END
