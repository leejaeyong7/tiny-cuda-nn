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

template <typename T>
__device__ T trilinear_interp(
    const T * __restrict__ features, 
    const uint32_t R, 
    const float sx, 
    const float sy, 
    const float sz){

    const float x = ((sx + 1) * 0.5) * (R - 1);
    const float y = ((sy + 1) * 0.5) * (R - 1);
    const float z = ((sz + 1) * 0.5) * (R - 1);

    const uint32_t x0 = min(max((uint32_t)floor(x), 0), R-1);
    const uint32_t y0 = min(max((uint32_t)floor(y), 0), R-1);
    const uint32_t z0 = min(max((uint32_t)floor(z), 0), R-1);
    const uint32_t x1 = max(min((uint32_t)ceil(x), R-1), 0);
    const uint32_t y1 = max(min((uint32_t)ceil(y), R-1), 0);
    const uint32_t z1 = max(min((uint32_t)ceil(z), R-1), 0);

    const float wx = x - x0;
    const float wy = y - y0;
    const float wz = z - z0;

    T result = 0;

    TCNN_PRAGMA_UNROLL
    for(int l = 0; l < 8; l++) {
        const T* tp = features + \
                      (l & 0x01 ? z1 : z0) * R*R + \
                      (l & 0x02 ? y1 : y0) * R + \
                      (l & 0x04 ? x1 : x0);
        result += *tp * (T)(
            (l & 0x04 ? wx : 1 - wx) *
            (l & 0x02 ? wy : 1 - wy) *
            (l & 0x01 ? wz : 1 - wz)
        );
    }

    return result;
}

template <typename T>
__device__ void grad_trilinear_interp(
    T * __restrict__ grad_features, 
    const uint32_t R, 
    const float sx, 
    const float sy, 
    const float sz, 
    const T grad_output
){
    const float x = ((sx + 1) * 0.5) * (R - 1);
    const float y = ((sy + 1) * 0.5) * (R - 1);
    const float z = ((sz + 1) * 0.5) * (R - 1);

    const uint32_t x0 = min(max((uint32_t)floor(x), 0), R-1);
    const uint32_t y0 = min(max((uint32_t)floor(y), 0), R-1);
    const uint32_t z0 = min(max((uint32_t)floor(z), 0), R-1);
    const uint32_t x1 = max(min((uint32_t)ceil(x), R-1), 0);
    const uint32_t y1 = max(min((uint32_t)ceil(y), R-1), 0);
    const uint32_t z1 = max(min((uint32_t)ceil(z), R-1), 0);

    const float wx = x - x0;
    const float wy = y - y0;
    const float wz = z - z0;

    TCNN_PRAGMA_UNROLL
    for(int l = 0; l < 8; l++) {
        T* tp = grad_features + \
                (l & 0x01 ? z1 : z0) * R*R + \
                (l & 0x02 ? y1 : y0) * R + \
                (l & 0x04 ? x1 : x0);
        atomicAdd(tp, grad_output * (T)(
            (l & 0x04 ? wx : 1 - wx) *
            (l & 0x02 ? wy : 1 - wy) *
            (l & 0x01 ? wz : 1 - wz)
        ));
    }
}


// __global__ void frequency_encoding(
// 	const uint32_t num_elements,
// 	const uint32_t n_frequencies,
// 	const uint32_t num_to_encode,
// 	const uint32_t num_to_pad,
// 	MatrixView<const float> data_in,
// 	MatrixView<T> data_out,
// 	float* __restrict__ dy_dx)
template <typename T>
__global__ void kernel_qff_forward(
    const uint32_t B, 
	const uint32_t F, 
    const uint32_t C, 
    const uint32_t R,
    const uint32_t min_log2_freq, 
    const uint32_t max_log2_freq, 
    const uint32_t P,
    MatrixView<const float> points,      // Bx3
    const T * __restrict__ features,     // Fx2xCxRxRxR
	MatrixView<T> outputs             	 // BxF2C
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b>= B) return;
    const uint32_t f = blockIdx.y;
    const uint32_t c2 = blockIdx.z;
    const uint32_t RRR = R*R*R;
    const uint32_t c = c2 / 2;
    const uint32_t s = c2 % 2;

    features += f*2*C*RRR + s*C*RRR + c*RRR;

	const float freq_base = (float) (f * (max_log2_freq - min_log2_freq)) / (float) F;
    const float freq = scalbnf(1.0, freq_base);

    // first compute sinusoidal coeffs
    const float px = points(0, b);
    const float py = points(1, b);
    const float pz = points(2, b);

    const T sx = (s == 0) ? __sinf(freq * px) : __cosf(freq * px);
    const T sy = (s == 0) ? __sinf(freq * py) : __cosf(freq * py);
    const T sz = (s == 0) ? __sinf(freq * pz) : __cosf(freq * pz);
    outputs(f*2*C + s*C + c, b) = trilinear_interp(features, R, sx, sy, sz);
}


template <typename T>
__global__ void kernel_qff_backward(
	const uint32_t B, 
	const uint32_t F, 
    const uint32_t C, 
    const uint32_t R,
    const uint32_t min_log2_freq, 
    const uint32_t max_log2_freq, 
    const uint32_t P,
	MatrixView<const T> grad_output,
    MatrixView<const float> points,      // Bx3
    // MatrixView<float> grad_features
    T * __restrict__ grad_features       // Fx2xCxRxRxR
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b>= B) return;
    const uint32_t f = blockIdx.y;
    const uint32_t c2 = blockIdx.z;
    const uint32_t RRR = R*R*R;
    const uint32_t c = c2 / 2;
    const uint32_t s = c2 % 2;

    // setup gradient offset
    grad_features += f*2*C*RRR + s*C*RRR + c*RRR;

    const T go = grad_output(f*2*C + s*C + c, b);
	const float freq_base = (float) (f * (max_log2_freq - min_log2_freq)) / (float) F;
    const float freq = scalbnf(1.0, freq_base);

    // first compute sinusoidal coeffs
    const float px = points(0, b);
    const float py = points(1, b);
    const float pz = points(2, b);

    const T sx = (s == 0) ? __sinf(freq * px) : __cosf(freq * px);
    const T sy = (s == 0) ? __sinf(freq * py) : __cosf(freq * py);
    const T sz = (s == 0) ? __sinf(freq * pz) : __cosf(freq * pz);
    grad_trilinear_interp(grad_features, R, sx, sy, sz, go);
}

template <typename T>
class QFF : public Encoding<T> {
public:
	QFF(uint32_t log2_min_freq,
		uint32_t log2_max_freq,
		uint32_t n_features,
		uint32_t n_quants,
		uint32_t n_frequencies, 
		uint32_t n_dims_to_encode)
	: m_log2_min_freq{log2_min_freq}, m_log2_max_freq{log2_max_freq}, 
	  m_n_features{n_features}, m_n_quants{n_quants}, m_n_frequencies{n_frequencies}, 
	  m_n_dims_to_encode{n_dims_to_encode} 
	{
		m_n_output_dims = m_n_frequencies * 2 * m_n_features;
        m_n_params = m_n_quants * m_n_quants * m_n_quants * 2 * m_n_frequencies * m_n_features;
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
			forward->dy_dx = GPUMatrix<float>{m_n_dims_to_encode * m_n_frequencies * 2, input.n(), stream};
		}
		static constexpr uint32_t N_THREADS = 512;
		const dim3 blocks_qff = { div_round_up(input.n(), N_THREADS), m_n_frequencies, 2 * m_n_features};
		kernel_qff_forward<T><<<blocks_qff, N_THREADS, 0, stream>>>(
			input.n(), // B
			m_n_frequencies, // F
			m_n_features, // C
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

        T* grid_gradient = this->gradients();

        if (param_gradients_mode == EGradientMode::Overwrite) {
            CUDA_CHECK_THROW(cudaMemsetAsync(grid_gradient, 0, m_n_params * sizeof(T), stream));
        }


		static constexpr uint32_t N_THREADS = 512;
		const dim3 blocks_qff = { div_round_up(input.n(), N_THREADS), m_n_frequencies, 2*m_n_features };
		kernel_qff_backward<T><<<blocks_qff, N_THREADS, 0, stream>>>(
			input.n(), // B
			m_n_frequencies, // F
			m_n_features, // C
			m_n_quants, // Q
			m_log2_min_freq, // I
			m_log2_max_freq, // X
			m_n_to_pad, // P
			dL_doutput.view(),
			input.view(),
            grid_gradient
			// dL_dinput->view()
        );
	}

	uint32_t input_width() const override {
		return m_n_dims_to_encode;
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
			{"n_features_per_level", m_n_features}
		};
	}

private:
	struct ForwardContext : public Context {
		GPUMatrix<float> dy_dx;
	};

	uint32_t m_n_frequencies;
	uint32_t m_n_features;
	uint32_t m_n_quants;
	uint32_t m_log2_min_freq;
	uint32_t m_log2_max_freq;
	uint32_t m_n_dims_to_encode;

	// derived sizes
	uint32_t m_n_params;
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;
};

TCNN_NAMESPACE_END
