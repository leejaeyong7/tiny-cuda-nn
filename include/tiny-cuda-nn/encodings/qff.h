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

    const float xc = ((sx + 1) * 0.5) * (R - 1);
    const float yc = ((sy + 1) * 0.5) * (R - 1);
    const float zc = ((sz + 1) * 0.5) * (R - 1);
    
    const uint32_t x0 = min(max((uint32_t)floor(xc), 0), R-1);
    const uint32_t y0 = min(max((uint32_t)floor(yc), 0), R-1);
    const uint32_t z0 = min(max((uint32_t)floor(zc), 0), R-1);
    const uint32_t x1 = max(min((uint32_t)ceil(xc), R-1), 0);
    const uint32_t y1 = max(min((uint32_t)ceil(yc), R-1), 0);
    const uint32_t z1 = max(min((uint32_t)ceil(zc), R-1), 0);

    const T wx0 = (T)(xc - (float) x0);
    const T wy0 = (T)(yc - (float) y0);
    const T wz0 = (T)(zc - (float) z0);
    const T wx1 = (T)1 - wx0;
    const T wy1 = (T)1 - wy0;
    const T wz1 = (T)1 - wz0;

    // fxyz
    const T f000 = features[z0 * R * R + y0 * R + x0];
    const T f001 = features[z1 * R * R + y0 * R + x0];
    const T f010 = features[z0 * R * R + y1 * R + x0];
    const T f011 = features[z1 * R * R + y1 * R + x0];
    const T f100 = features[z0 * R * R + y0 * R + x1];
    const T f101 = features[z1 * R * R + y0 * R + x1];
    const T f110 = features[z0 * R * R + y1 * R + x1];
    const T f111 = features[z1 * R * R + y1 * R + x1];

    const T f00 = f000 * wx1 + f100 * wx0;
    const T f01 = f001 * wx1 + f101 * wx0;
    const T f10 = f010 * wx1 + f110 * wx0;
    const T f11 = f011 * wx1 + f111 * wx0;

    const T f0 = f00 * wy1 + f10 * wy0;
    const T f1 = f01 * wy1 + f11 * wy0;

    const T f = f0 * wz1 + f1 * wz0;

    return f;
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

    const float xc = ((sx + 1) * 0.5) * (R - 1);
    const float yc = ((sy + 1) * 0.5) * (R - 1);
    const float zc = ((sz + 1) * 0.5) * (R - 1);
    
    const uint32_t x0 = min(max((uint32_t)floor(xc), 0), R-1);
    const uint32_t y0 = min(max((uint32_t)floor(yc), 0), R-1);
    const uint32_t z0 = min(max((uint32_t)floor(zc), 0), R-1);
    const uint32_t x1 = max(min((uint32_t)ceil(xc), R-1), 0);
    const uint32_t y1 = max(min((uint32_t)ceil(yc), R-1), 0);
    const uint32_t z1 = max(min((uint32_t)ceil(zc), R-1), 0);

    const T wx0 = (T)(xc - (float) x0);
    const T wy0 = (T)(yc - (float) y0);
    const T wz0 = (T)(zc - (float) z0);
    const T wx1 = (T)1 - wx0;
    const T wy1 = (T)1 - wy0;
    const T wz1 = (T)1 - wz0;

    // apply gradient
    atomicAdd(grad_features + z0 * R * R + y0 * R + x0, (grad_output * wx1 * wy1 * wz1));
    atomicAdd(grad_features + z1 * R * R + y0 * R + x0, (grad_output * wx1 * wy1 * wz0));
    atomicAdd(grad_features + z0 * R * R + y1 * R + x0, (grad_output * wx1 * wy0 * wz1));
    atomicAdd(grad_features + z1 * R * R + y1 * R + x0, (grad_output * wx1 * wy0 * wz0));
    atomicAdd(grad_features + z0 * R * R + y0 * R + x1, (grad_output * wx0 * wy1 * wz1));
    atomicAdd(grad_features + z1 * R * R + y0 * R + x1, (grad_output * wx0 * wy1 * wz0));
    atomicAdd(grad_features + z0 * R * R + y1 * R + x1, (grad_output * wx0 * wy0 * wz1));
    atomicAdd(grad_features + z1 * R * R + y1 * R + x1, (grad_output * wx0 * wy0 * wz0));
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
    const uint32_t bf = blockIdx.x * blockDim.x + threadIdx.x;
	if (bf>= B*F) return;
    const uint32_t b = bf / F;
    const uint32_t f = bf % F;
    const uint32_t RRR = R*R*R;

    features += f*2*C*RRR;
    // outputs += b * F * 2 * C + f * 2 * C;

	const float freq_base = (float) (f * (max_log2_freq - min_log2_freq)) / (float) F;
    const float freq = pow(2.0, freq_base);

    // first compute sinusoidal coeffs
    const float px = points(0, b);
    const float py = points(1, b);
    const float pz = points(2, b);

    const float sx = __sinf(freq * px);
    const float sy = __sinf(freq * py);
    const float sz = __sinf(freq * pz);
    const float cx = __cosf(freq * px);
    const float cy = __cosf(freq * py);
    const float cz = __cosf(freq * pz);

    for (uint32_t c = 0; c < C; c++){
        const T* fv = features + c * RRR;
        // Bx(F2C+P)

        outputs(f * 2 * C + 0 * C + c, b)= trilinear_interp(fv + 0 * C * RRR, R, sx, sy, sz);
        outputs(f * 2 * C + 1 * C + c, b)= trilinear_interp(fv + 1 * C * RRR, R, cx, cy, cz);
    }
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
    const uint32_t bf = blockIdx.x * blockDim.x + threadIdx.x;
	if (bf>= B*F) return;
    const uint32_t b = bf / F;
    const uint32_t f = bf % F;
    const uint32_t RRR = R*R*R;

    // setup gradient offset
    grad_features += f*2*C*R*R*R;

	const float freq_base = (float) (f * (max_log2_freq - min_log2_freq)) / (float) F;
    const float freq = pow(2.0, freq_base);

    // first compute sinusoidal coeffs
    const float px = points(0, b);
    const float py = points(1, b);
    const float pz = points(2, b);

    const float sx = __sinf(freq * px);
    const float sy = __sinf(freq * py);
    const float sz = __sinf(freq * pz);
    const float cx = __cosf(freq * px);
    const float cy = __cosf(freq * py);
    const float cz = __cosf(freq * pz);

    for (uint32_t c = 0; c < C; c++){
        T* gf = grad_features + c * R*R*R;

        // compute grad features
        grad_trilinear_interp(gf + 0 * C * RRR, R, sx, sy, sz, grad_output(f * 2 * C + c + 0 * C, b));
        grad_trilinear_interp(gf + 1 * C * RRR, R, cx, cy, cz, grad_output(f * 2 * C + c + 1 * C, b)); 
    }
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
		static constexpr uint32_t N_THREADS = 256;
		const dim3 blocks_qff = { div_round_up(m_n_frequencies * input.n(), N_THREADS), 1 , 1 };
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


		static constexpr uint32_t N_THREADS = 256;
		const dim3 blocks_qff = { div_round_up(m_n_frequencies * input.n(), N_THREADS), 1 , 1 };
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
