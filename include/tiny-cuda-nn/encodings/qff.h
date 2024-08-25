#pragma once

#include <tiny-cuda-nn/object.h>
#include <tiny-cuda-nn/interp.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>
namespace tcnn {

struct ForwardContext : public Context {
	GPUMatrix<float, RM> dy_dx;
};


#define M_TAU 6.28318530717958647692
#define M_PI 3.14159265358979323846
#define M_HI 1.57079632679489661923

/**
 * @brief QFF1 encoding for ND inputs.
 * T: float or double
 * D: number of dimensions to encode (2 or 3)
 * C: number of features per frequency (1, 2, 4 or 8)
 * R: number of correlations per level (1, 2, 4 or 8) (Rank)
 */
template <typename T, uint32_t D, uint32_t C, uint32_t R>
class QFF : public Encoding<T> {
public:
	QFF(int32_t log2_min_freq,
		int32_t log2_max_freq,
		uint32_t n_quants,
		uint32_t n_frequencies)
	: m_log2_min_freq{log2_min_freq},
	  m_log2_max_freq{log2_max_freq},
	  m_n_quants{n_quants},
	  m_n_frequencies{n_frequencies}
	{
		m_n_output_dims = m_n_frequencies * 2 * C;
	}

	uint32_t input_width() const override {
		return D;
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

	uint32_t n_pos_dims() const {
		return D;
	}
	uint32_t n_features() const {
		return C;
	}

	uint32_t required_output_alignment() const override {
		return 1;
	}

	MatrixLayout preferred_output_layout() const override {
		return AoS;
	}

	uint32_t n_corrs() const {
		return R;
	};

	virtual std::string otype() const = 0;

	json hyperparams() const override {
		return {
			{"otype", otype()},
			{"n_frequencies", m_n_frequencies},
			{"log2_min_freq", m_log2_min_freq},
			{"log2_max_freq", m_log2_max_freq},
			{"n_quants", m_n_quants},
			{"n_features_per_level", C},
			{"rank", R}
		};
	}

protected:

	uint32_t m_n_frequencies;
	uint32_t m_n_quants;
	int32_t m_log2_min_freq;
	int32_t m_log2_max_freq;

	// derived sizes
	uint32_t m_n_params;
	uint32_t m_n_output_dims;
	uint32_t m_n_to_pad = 0;
};

}