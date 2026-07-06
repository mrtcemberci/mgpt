#include "LayerNormLayer.h"

LayerNormLayer::LayerNormLayer(int channels, float eps)
    : channels(channels),
      eps(eps),
      scale({channels}, 1.0f), // Gamma initialized to 1.0
      shift({channels}, 0.0f) { // Beta initialized to 0.0
}

// FORWARD PASS: Normalize across channel dimension C, apply gamma scale and beta shift
Tensor LayerNormLayer::forward(const Tensor& input) {
    if (input.shape.empty() || input.shape.back() != channels) {
        std::cerr << "LayerNormLayer::forward: input channel dimension mismatch!" << std::endl;
        exit(-1);
    }
    cached_input = input;

    int total_tokens = (int)(input.size() / channels);

    std::vector<int> mean_var_shape = input.shape;
    mean_var_shape.back() = 1;
    cached_mean = Tensor(mean_var_shape, 0.0f);
    cached_var = Tensor(mean_var_shape, 0.0f);
    cached_x_hat = Tensor(input.shape, 0.0f);
    Tensor output(input.shape, 0.0f);

    for (int i = 0; i < total_tokens; ++i) {
        const float* x_ptr = input.data.data() + i * channels;
        float* x_hat_ptr = cached_x_hat.data.data() + i * channels;
        float* out_ptr = output.data.data() + i * channels;

        // Compute token channel mean
        float sum = 0.0f;
        for (int j = 0; j < channels; ++j) {
            sum += x_ptr[j];
        }
        float mean = sum / (float)channels;
        cached_mean.data[i] = mean;

        // Compute token channel variance
        float var_sum = 0.0f;
        for (int j = 0; j < channels; ++j) {
            float diff = x_ptr[j] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / (float)channels;
        cached_var.data[i] = var;

        // Compute normalized x_hat and scaled/shifted output y
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < channels; ++j) {
            float x_hat = (x_ptr[j] - mean) * inv_std;
            x_hat_ptr[j] = x_hat;
            out_ptr[j] = scale.data[j] * x_hat + shift.data[j];
        }
    }

    return output;
}

// BACKWARD PASS: Compute dGamma, dBeta across all tokens, return dX
Tensor LayerNormLayer::backward(const Tensor& dout) {
    if (dout.shape != cached_input.shape) {
        std::cerr << "LayerNormLayer::backward: dout shape mismatch!" << std::endl;
        exit(-1);
    }

    int total_tokens = (int)(dout.size() / channels);
    Tensor dX(dout.shape, 0.0f);

    for (int i = 0; i < total_tokens; ++i) {
        const float* dout_ptr = dout.data.data() + i * channels;
        const float* x_hat_ptr = cached_x_hat.data.data() + i * channels;
        float* dx_ptr = dX.data.data() + i * channels;

        float var = cached_var.data[i];
        float inv_std = 1.0f / std::sqrt(var + eps);

        float sum_dx_hat = 0.0f;
        float sum_dx_hat_x_hat = 0.0f;

        for (int j = 0; j < channels; ++j) {
            float dy = dout_ptr[j];
            
            // Accumulate gradients for learnable parameters scale (gamma) and shift (beta)
            shift.grad[j] += dy;
            scale.grad[j] += dy * x_hat_ptr[j];

            // Local d_x_hat
            float dx_hat = dy * scale.data[j];
            sum_dx_hat += dx_hat;
            sum_dx_hat_x_hat += dx_hat * x_hat_ptr[j];
        }

        float mean_sum_dx_hat = sum_dx_hat / (float)channels;
        float mean_sum_dx_hat_x_hat = sum_dx_hat_x_hat / (float)channels;

        for (int j = 0; j < channels; ++j) {
            float dx_hat = dout_ptr[j] * scale.data[j];
            dx_ptr[j] = inv_std * (dx_hat - mean_sum_dx_hat - x_hat_ptr[j] * mean_sum_dx_hat_x_hat);
        }
    }

    return dX;
}

std::vector<Tensor*> LayerNormLayer::get_parameters() {
    return { &scale, &shift };
}
