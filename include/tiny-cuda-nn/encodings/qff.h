#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN
static inline  __device__ uint32_t pow(const uint32_t base, const uint32_t exp) {
    uint32_t val = 1;
    for(int i = 0; i < exp; i++){
        val *= base;
    }
    return val;
}


template <typename T, uint32_t N_POS_DIMS, uint32_t C>
__device__ void nlinear_interp(
    const T * __restrict__ features, 
    const uint32_t R, 
	const float sc[N_POS_DIMS],
	MatrixView<T> outputs,
	const uint32_t b,
	const uint32_t out_offset
) {
	float w[N_POS_DIMS];
	float p0[N_POS_DIMS];
	float p1[N_POS_DIMS];
	uint32_t o[N_POS_DIMS];

	TCNN_PRAGMA_UNROLL
	for(uint32_t i = 0; i < N_POS_DIMS; i++){
		const float p = ((sc[i] + 1) * 0.5) * (R -1);
		p0[i] = min(max((uint32_t)floor(p), 0), R-1);
		p1[i] = max(min((uint32_t)ceil(p), R-1), 0);
		w[i] = p - p0[i];
		o[i] = pow(R, i);
	}

	float results[C] = {0};
	TCNN_PRAGMA_UNROLL
	for(int l = 0; l < 8; l++) {
		uint32_t offset = 0;
		float weight = 1;

		TCNN_PRAGMA_UNROLL
		for(int i = 0; i < N_POS_DIMS; i++){
			const uint32_t inv_i = N_POS_DIMS - i - 1;
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

template <typename GRAD_T, typename T, uint32_t N_POS_DIMS, uint32_t C>
__device__ void grad_nlinear_interp(
    GRAD_T * __restrict__ grad_features, 
    const uint32_t R, 
    const float sc[N_POS_DIMS], 
    MatrixView<const T> grad_output,
	const uint32_t b,
	const uint32_t out_offset 
){
	float w[N_POS_DIMS];
	float p0[N_POS_DIMS];
	float p1[N_POS_DIMS];
	uint32_t o[N_POS_DIMS];

	TCNN_PRAGMA_UNROLL
	for(uint32_t i = 0; i < N_POS_DIMS; i++){
		const float p = ((sc[i] + 1) * 0.5) * (R -1);
		p0[i] = min(max((uint32_t)floor(p), 0), R-1);
		p1[i] = max(min((uint32_t)ceil(p), R-1), 0);
		w[i] = p - p0[i];
		o[i] = pow(R, i);
	}

	TCNN_PRAGMA_UNROLL
	for(int l = 0; l < 8; l++) {
		uint32_t offset = 0;
		float weight = 1;

		TCNN_PRAGMA_UNROLL
		for(int i = 0; i < N_POS_DIMS; i++){
			const uint32_t inv_i = N_POS_DIMS - i - 1;
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

// grad_point_helper<T, N_POS_DIMS, C>(grad_points, features, R, sc, dsc, grad_output, b,b*F*2*C+ f*2*C + s*C);
template <typename T, uint32_t N_POS_DIMS, uint32_t C>
__device__ void grad_point_helper(
    MatrixView<float> grad_points, 
	const T * __restrict__ features, 
    const uint32_t R, 
    const float sc[N_POS_DIMS], 
    const float dsc[N_POS_DIMS], 
    MatrixView<const T> grad_output,
	const uint32_t b,
	const uint32_t out_offset 
){
    // for b, r, r, rxc
    float w[N_POS_DIMS];
    float dw[N_POS_DIMS];
	float p0[N_POS_DIMS];
	float p1[N_POS_DIMS];
	uint32_t o[N_POS_DIMS];

	#pragma unroll
	for(uint32_t i = 0; i < N_POS_DIMS; i++){
		const float p = ((sc[i] + 1) * 0.5) * (R -1);
		p0[i] = min(max((uint32_t)floor(p), 0), R-1);
		p1[i] = max(min((uint32_t)ceil(p), R-1), 0);
		w[i] = p - p0[i];
		o[i] = pow(R, i);
		dw[i] = dsc[i] * 0.5 * (R - 1);
	}

    float results[N_POS_DIMS*C] = {0};

    #pragma unroll
    for(uint32_t l = 0; l < pow(2, N_POS_DIMS); l++){
        uint32_t offset = 0;
        float weights[N_POS_DIMS] = {0};

		#pragma unroll
		for(uint32_t k = 0; k < N_POS_DIMS; k++){
			weights[k] = 1;
		}

        #pragma unroll
        for(uint32_t i = 0; i < N_POS_DIMS; i++){
            const uint32_t inv_i = N_POS_DIMS - i - 1;
            offset += (l & (1 << inv_i) ? p1[i] : p0[i]) * o[i];
            #pragma unroll
            for(uint32_t k  =0; k < N_POS_DIMS; k++){
                weights[k] *= (i == k) ? ((l & (1 << inv_i)) ? dw[i] : -dw[i]) : 
                                         ((l & (1 << inv_i)) ? w[i] : 1 - w[i]);
            }
        }
        
        #pragma unroll
        for(int c = 0; c < C; c++){
            float feature = (float)*(features + offset*C + c);
            #pragma unroll
            for (int k = 0; k < N_POS_DIMS; k++){
                results[k*C + c] += feature * weights[k];
            }
        }
    }
    #pragma unroll
    for(int c = 0; c < C; c++){
		float go = (float)grad_output(out_offset + c, b);
        #pragma unroll
        for (int k = 0; k < N_POS_DIMS; k++){
            atomicAdd(&grad_points(k, b), go * results[k*C + c]);
        }
    }
}


// grad_grad_helper<GRAD_T, T, N_POS_DIMS, C>(features, grad2_features, grad_grad_outputs, R, sc, dsc, dps, grad_output, b, b*F*2*C + f*2*C + s*C);
template <typename GRAD_T, typename T, uint32_t N_POS_DIMS, uint32_t C>
__device__ void grad_grad_helper(
    const T * __restrict__ features, 
    GRAD_T * __restrict__ grad2_features, 
    MatrixView<T> grad_grad_output, 
    const uint32_t R, 
    const float sc[N_POS_DIMS], 
    const float dsc[N_POS_DIMS], 
    const float dps[N_POS_DIMS], 
    MatrixView<const T>grad_outputs,
    const uint32_t b,
    const uint32_t out_offset
){
    // for b, r, r, rxc
    float w[N_POS_DIMS];
    float dw[N_POS_DIMS];
	float p0[N_POS_DIMS];
	float p1[N_POS_DIMS];
	uint32_t o[N_POS_DIMS];

	#pragma unroll
	for(uint32_t i = 0; i < N_POS_DIMS; i++){
		const float p = ((sc[i] + 1) * 0.5) * (R -1);
		p0[i] = min(max((uint32_t)floor(p), 0), R-1);
		p1[i] = max(min((uint32_t)ceil(p), R-1), 0);
		w[i] = p - p0[i];
		o[i] = pow(R, i);
		dw[i] = dsc[i] * 0.5 * (R - 1) * dps[i];
	}

    T results[N_POS_DIMS*C] = {0};
    #pragma unroll
    for(uint32_t l = 0; l < pow(2, N_POS_DIMS); l++){
        uint32_t offset = 0;
        float weights[N_POS_DIMS] = {0};

		#pragma unroll
		for(uint32_t k = 0; k < N_POS_DIMS; k++){
			weights[k] = 1;
		}

        #pragma unroll
        for(uint32_t i = 0; i < N_POS_DIMS; i++){
            const uint32_t inv_i = N_POS_DIMS - i - 1;
            offset += (l & (1 << inv_i) ? p1[i] : p0[i]) * o[i];

            #pragma unroll
            for(uint32_t k = 0; k < N_POS_DIMS; k++){
                weights[k] *= (i == k) ? ((l & (1 << inv_i)) ? dw[i] : -dw[i]) : 
                                         ((l & (1 << inv_i)) ? w[i] : 1 - w[i]);
            }
        }
        #pragma unroll
        for(int c = 0; c < C; c++){
            T feature = *(features + offset*C + c);
            #pragma unroll
            for (int k = 0; k < N_POS_DIMS; k++){
                results[k*C + c] += feature * (T)weights[k];
            }
		}

		float g2f = 0;
		#pragma unroll
		for (int k = 0; k < N_POS_DIMS; k++){
			g2f += weights[k];
		}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (C > 1 && std::is_same<GRAD_T, __half>::value) {
			TCNN_PRAGMA_UNROLL
			for(uint32_t c = 0; c < C; c+=2){
				__half2 v = {
					(__half)((float)grad_outputs(out_offset + c, b) * g2f),
					(__half)((float)grad_outputs(out_offset + c + 1, b) * g2f)
				};
				atomicAdd((__half2*)(grad2_features + offset * C + c), v);
			}
		} else
#endif
		{
			TCNN_PRAGMA_UNROLL
			for(int c = 0; c < C; c++){
				float go = (float)grad_outputs(out_offset + c, b);
				atomicAdd(grad2_features + offset *C + c, (GRAD_T)(go * g2f));
			}
		}
    }
    #pragma unroll
    for(int c = 0; c < C; c++){
        T ggo = 0;
        #pragma unroll
        for (int k = 0; k < N_POS_DIMS; k++){
            ggo += results[k * C + c];
        }
        grad_grad_output(out_offset + c, b) = ggo;
    }
}





template <typename T, uint32_t N_POS_DIMS, uint32_t C>
__global__ void kernel_qff_forward(
    const uint32_t B, 
	const uint32_t F, 
    const uint32_t R,
    const uint32_t min_log2_freq, 
    const uint32_t max_log2_freq, 
    const uint32_t P,
    MatrixView<const float> points,      // Bx3
    const T * __restrict__ features,     // Fx2xRsxC
	MatrixView<T> outputs             	 // BxF2C
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b>= B) return;
    const uint32_t f = blockIdx.y;
    const uint32_t s = blockIdx.z;
	const float freq_base = (float) (f * (max_log2_freq - min_log2_freq)) / (float) (F - 1);
    const float freq = scalbnf(1.0, freq_base);
    const uint32_t Rs = pow(R, N_POS_DIMS);
    features += f*2*Rs*C + s*Rs*C;

	float sc[N_POS_DIMS];

	TCNN_PRAGMA_UNROLL
	for(int i = 0; i < N_POS_DIMS; i++){
		float p = points(i, b);
		sc[i] = (s == 0) ? __sinf(freq * p) : __cosf(freq * p);
	}
	nlinear_interp<T, N_POS_DIMS, C>(features, R, sc, outputs, b, f*2*C + s*C);
}


template <typename GRAD_T, typename T, uint32_t N_POS_DIMS, uint32_t C>
__global__ void kernel_qff_backward_features(
	const uint32_t B, 
	const uint32_t F, 
    const uint32_t R,
    const uint32_t min_log2_freq, 
    const uint32_t max_log2_freq, 
    const uint32_t P,
	MatrixView<const T> grad_output,
    MatrixView<const float> points,      // Bx3
    GRAD_T * __restrict__ grad_features       // Fx2xRsxC
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b>= B) return;
    const uint32_t f = blockIdx.y;
    const uint32_t s = blockIdx.z;

	const float freq_base = (float) (f * (max_log2_freq - min_log2_freq)) / (float) (F - 1);
    const float freq = scalbnf(1.0, freq_base);
    const uint32_t Rs = pow(R, N_POS_DIMS);

    grad_features += f*2*C*Rs + s*C*Rs;

	float sc[N_POS_DIMS];

	TCNN_PRAGMA_UNROLL
	for(int i = 0; i < N_POS_DIMS; i++){
		float p = points(i, b);
		sc[i] = (s == 0) ? __sinf(freq * p) : __cosf(freq * p);
	}
	grad_nlinear_interp<GRAD_T, T, N_POS_DIMS, C>(grad_features, R, sc, grad_output, b, f*2*C + s*C);
}

template <typename T, uint32_t N_POS_DIMS, uint32_t C>
__global__ void kernel_qff_backward_input(
	const uint32_t B, 
	const uint32_t F, 
    const uint32_t R,
    const uint32_t min_log2_freq, 
    const uint32_t max_log2_freq, 
    const uint32_t P,
	MatrixView<const T> grad_output,
    MatrixView<const float> points,      // Bx3
    MatrixView<float> grad_points,       // Bx3
	T * __restrict__ features       // Fx2xRsxC
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b>= B) return;
    const uint32_t f = blockIdx.y;
    const uint32_t s = blockIdx.z;

	const float freq_base = (float) (f * (max_log2_freq - min_log2_freq)) / (float) (F - 1);
    const float freq = scalbnf(1.0, freq_base);
    const uint32_t Rs = pow(R, N_POS_DIMS);

	// grad_points += b*N_POS_DIMS;
    features += f*2*C*Rs + s*C*Rs;

	float sc[N_POS_DIMS];
	float dsc[N_POS_DIMS];

	TCNN_PRAGMA_UNROLL
	for(int i = 0; i < N_POS_DIMS; i++){
		float p = points(i, b);
		sc[i] = (s == 0) ? __sinf(freq * p) : __cosf(freq * p);
		dsc[i] = ((s == 0) ? __cosf(freq * p) : -__sinf(freq * p)) * freq;
	}

	grad_point_helper<T, N_POS_DIMS, C>(grad_points, features, R, sc, dsc, grad_output, b,b*F*2*C+ f*2*C + s*C);
}




template <typename GRAD_T, typename T, uint32_t N_POS_DIMS, uint32_t C>
__global__ void kernel_qff_backward_input_backward(
	const uint32_t B, 
	const uint32_t F, 
    const uint32_t R,
    const uint32_t min_log2_freq, 
    const uint32_t max_log2_freq, 
    const uint32_t P,
	MatrixView<const T> grad_output,
    MatrixView<const float> points,      // Bx3
    MatrixView<const float> grad_grad_points,  // Bx3
	const T * __restrict__ features,       // Fx2xRsxC
	GRAD_T* __restrict__ grad2_features,
	MatrixView<T> grad_grad_outputs
) {
	const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b>= B) return;
    const uint32_t f = blockIdx.y;
    const uint32_t s = blockIdx.z;

	const float freq_base = (float) (f * (max_log2_freq - min_log2_freq)) / (float) (F - 1);
    const float freq = scalbnf(1.0, freq_base);
    const uint32_t Rs = pow(R, N_POS_DIMS);

    features += f*2*C*Rs + s*C*Rs;
    grad2_features += f*2*C*Rs + s*C*Rs;


	float sc[N_POS_DIMS];
	float dsc[N_POS_DIMS];
	float dps[N_POS_DIMS];

	TCNN_PRAGMA_UNROLL
	for(int i = 0; i < N_POS_DIMS; i++){
		float p = points(i, b);
		float dp = grad_grad_points(i, b);
		sc[i] = (s == 0) ? __sinf(freq * p) : __cosf(freq * p);
		dsc[i] = ((s == 0) ? __cosf(freq * p) : -__sinf(freq * p)) * freq;
		dps[i] = dp;
	}

    grad_grad_helper<GRAD_T, T, N_POS_DIMS, C>(features, grad2_features, grad_grad_outputs, R, sc, dsc, dps, grad_output, b, b*F*2*C + f*2*C + s*C);
}





template <typename T>
class QFF: public Encoding<T> {
public:
	virtual uint32_t n_pos_dims() const = 0;
	virtual uint32_t n_features() const = 0;
};

template <typename T, uint32_t N_POS_DIMS, uint32_t C>
class QFFTemplated : public QFF<T> {
#if TCNN_MIN_GPU_ARCH >= 62 || TCNN_MIN_GPU_ARCH == 60
	using grad_t = std::conditional_t<C == 1, float, T>;
#else
	using grad_t = float;
#endif
public:
	QFFTemplated(uint32_t log2_min_freq,
				 uint32_t log2_max_freq,
				 uint32_t n_quants,
				 uint32_t n_frequencies)
	: m_log2_min_freq{log2_min_freq}, 
	  m_log2_max_freq{log2_max_freq}, 
	  m_n_quants{n_quants}, 
	  m_n_frequencies{n_frequencies} 
	{
		m_n_output_dims = m_n_frequencies * 2 * C;
		m_n_params = n_quants * n_quants * n_quants * 2 * m_n_frequencies * C;
		// if (N_POS_DIMS == 2){
		// 	m_n_params = n_quants * n_quants * 2 * m_n_frequencies * C;
		// } else{
		// 	m_n_params = n_quants * n_quants * n_quants * 2 * m_n_frequencies * C;
		// }
	}

	std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		auto forward = std::make_unique<ForwardContext>();

		if (!output || padded_output_width() == 0) {
			return forward;
		}

		if (prepare_input_gradients) {
			forward->dy_dx = GPUMatrix<float>{N_POS_DIMS * m_n_frequencies * 2, input.n(), stream};
		}
		static constexpr uint32_t N_THREADS = 512;
		const dim3 blocks_qff = { div_round_up(input.n(), N_THREADS), m_n_frequencies, 2};
		kernel_qff_forward<T, N_POS_DIMS, C><<<blocks_qff, N_THREADS, 0, stream>>>(
			input.n(), // B
			m_n_frequencies, // F
			m_n_quants, // Q
			m_log2_min_freq, // I
			m_log2_max_freq, // X
			m_n_to_pad, // P
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
        if ((!dL_dinput && param_gradients_mode == EGradientMode::Ignore) || padded_output_width() == 0 || num_elements == 0) {
			return;
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

        // If not, accumulate in a temporary buffer and cast later.
		grad_t * grad_array;
		GPUMemoryArena::Allocation grad_array_tmp;
		if (!std::is_same<grad_t, T>::value) {
			grad_array_tmp = allocate_workspace(stream, m_n_params * sizeof(grad_t));
			grad_array = (grad_t*)grad_array_tmp.data();
		} else {
			grad_array = (grad_t*)this->gradients();
		}


        if (param_gradients_mode == EGradientMode::Overwrite) {
            CUDA_CHECK_THROW(cudaMemsetAsync(grad_array, 0, m_n_params * sizeof(grad_t), stream));
        }


		static constexpr uint32_t N_THREADS = 512;
		const dim3 blocks_qff = { div_round_up(input.n(), N_THREADS), m_n_frequencies, 2};
		kernel_qff_backward_features<grad_t, T, N_POS_DIMS, C><<<blocks_qff, N_THREADS, 0, stream>>>(
			input.n(), // B
			m_n_frequencies, // F
			m_n_quants, // Q
			m_log2_min_freq, // I
			m_log2_max_freq, // X
			m_n_to_pad, // P
			dL_doutput.view(),
			input.view(),
            grad_array 
        );

		if (!std::is_same<grad_t, T>::value) {
			parallel_for_gpu(stream, n_params(), [grad=this->gradients(), grad_tmp=grad_array] __device__ (size_t i) {
				grad[i] = (T)grad_tmp[i];
			});
		}

		if (!dL_dinput) {
			return;
		}

		linear_kernel(kernel_qff_backward_input<T, N_POS_DIMS, C>, 0, stream,
			input.n(), // B
			m_n_frequencies, // F
			m_n_quants, // Q
			m_log2_min_freq, // I
			m_log2_max_freq, // X
			m_n_to_pad, // P
			dL_doutput.view(),
			input.view(),
			dL_dinput->view(),
			use_inference_params ? this->inference_params() : this->params()
		);
	}


	void backward_backward_input_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<float>& dL_ddLdinput,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_ddLdoutput = nullptr,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		EGradientMode param_gradients_mode = EGradientMode::Overwrite
	) override {
        const uint32_t num_elements = input.n();
		if ((!dL_ddLdoutput && param_gradients_mode == EGradientMode::Ignore) || padded_output_width() == 0 || num_elements == 0) {
			return;
		}
		grad_t * grad_grad_array;
		GPUMemoryArena::Allocation grad_array_tmp;
		if (!std::is_same<grad_t, T>::value) {
			grad_array_tmp = allocate_workspace(stream, m_n_params * sizeof(grad_t));
			grad_grad_array = (grad_t*)grad_array_tmp.data();
		} else {
			grad_grad_array = (grad_t*)this->gradients();
		}


        if (param_gradients_mode == EGradientMode::Overwrite) {
            CUDA_CHECK_THROW(cudaMemsetAsync(grad_grad_array, 0, m_n_params * sizeof(grad_t), stream));
        }

		static constexpr uint32_t N_THREADS = 512;
		const dim3 blocks_qff = { div_round_up(input.n(), N_THREADS), m_n_frequencies, 2};
		// const uint32_t B, 
		// const uint32_t F, 
		// const uint32_t R,
		// const uint32_t min_log2_freq, 
		// const uint32_t max_log2_freq, 
		// const uint32_t P,
		// MatrixView<const T> grad_output,
		// MatrixView<const float> points,      // Bx3
		// MatrixView<float> grad_grad_points,  // Bx3
		// const T * __restrict__ features,       // Fx2xRsxC
		// GRAD_T* grad2_features,
		// GRAD_T* grad_grad_outputs,

		kernel_qff_backward_input_backward<grad_t, T, N_POS_DIMS, C><<<blocks_qff, N_THREADS, 0, stream>>>(
			input.n(), // B
			m_n_frequencies, // F
			m_n_quants, // Q
			m_log2_min_freq, // I
			m_log2_max_freq, // X
			m_n_to_pad, // P
			
			dL_doutput.view(),
			input.view(),
			dL_ddLdinput.view(),
			use_inference_params ? this->inference_params() : this->params(),
			grad_grad_array,
			dL_ddLdoutput->view()
        );
	}

	uint32_t input_width() const override {
		return N_POS_DIMS;
	}

	uint32_t padded_output_width() const override {
		return m_n_output_dims + m_n_to_pad;
	}

	uint32_t output_width() const override {
		return padded_output_width();
	}

	uint32_t required_input_alignment() const override {
		return 1;
	}

	void set_padded_output_width(uint32_t padded_output_width) override {
		CHECK_THROW(padded_output_width >= m_n_output_dims);
		m_n_to_pad = padded_output_width - m_n_output_dims;
	}
    void set_params_impl(T* params, T* inference_params, T* gradients) override { }

	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
		// Initialize the hashgrid from the GPU, because the number of parameters can be quite large.
		generate_random_uniform<float>(rnd, n_params(), params_full_precision, -1e-4f * scale, 1e-4f * scale);
	}
    size_t n_params() const override {
		return m_n_params;
	}

	uint32_t n_pos_dims() const override {
		return N_POS_DIMS;
	}

	uint32_t n_features() const override {
		return C;
	}

	uint32_t required_output_alignment() const override {
		return 1;
	}

	MatrixLayout preferred_output_layout() const override {
		return AoS;
	}

	json hyperparams() const override {
		return {
			{"otype", "QFF"},
			{"n_frequencies", m_n_frequencies},
			{"log2_min_freq", m_log2_min_freq},
			{"log2_max_freq", m_log2_max_freq},
			{"n_quants", m_n_quants},
			{"n_features_per_level", C}
		};
	}

private:
	struct ForwardContext : public Context {
		GPUMatrix<float> dy_dx;
	};

	uint32_t m_n_frequencies;
	uint32_t m_n_quants;
	uint32_t m_log2_min_freq;
	uint32_t m_log2_max_freq;

	// derived sizes
	uint32_t m_n_params;
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;
};


template <typename T, uint32_t N_POS_DIMS>
QFF<T>* create_qff_encoding_1(const json& encoding) {

#define TCNN_QFF_PARAMS \
	encoding.value("log2_min_freq", 0u), \
	encoding.value("log2_max_freq", 6u), \
	encoding.value("n_quants", 64u), \
	encoding.value("n_frequencies", 6u), \

	const uint32_t n_feats = encoding.value("n_features", 4u);
	switch (n_feats) {
		case 1: return new QFFTemplated<T, N_POS_DIMS, 1>{ TCNN_QFF_PARAMS };
		case 2: return new QFFTemplated<T, N_POS_DIMS, 2>{ TCNN_QFF_PARAMS };
		case 4: return new QFFTemplated<T, N_POS_DIMS, 4>{ TCNN_QFF_PARAMS };
		case 8: return new QFFTemplated<T, N_POS_DIMS, 8>{ TCNN_QFF_PARAMS };
		default: throw std::runtime_error{"QFF: number of features must be 1, 2, 4 or 8"};
	}
#undef TCNN_QFF_PARAMS
}

template <typename T>
QFF<T>* create_qff_encoding(uint32_t n_dims_to_encode, const json& encoding) {
	switch (n_dims_to_encode) {
		// case 2: return create_qff_encoding_1<T, 2>(encoding);
		case 3: return create_qff_encoding_1<T, 3>(encoding);
		default: throw std::runtime_error{"QFF: number of input dims must be 2,3 or 4."};
	}
}

TCNN_NAMESPACE_END
