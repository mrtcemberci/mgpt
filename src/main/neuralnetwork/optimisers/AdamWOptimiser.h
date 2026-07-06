#ifndef ADAMWOPTIMISER_H
#define ADAMWOPTIMISER_H
#include <cmath>
#include <unordered_map>

#include "Optimiser.h"
#include "Tensor.h"


class AdamWOptimizer : public Optimiser {
private:
    float lr, beta1, beta2, eps, weight_decay;
    int t; // Step counter
    std::unordered_map<Tensor*, std::vector<float>> m_cache;
    std::unordered_map<Tensor*, std::vector<float>> v_cache;
public:
    AdamWOptimizer(float learning_rate = 3e-4f, float b1 = 0.9f, float b2 = 0.999f,
                   float epsilon = 1e-8f, float decay = 0.01f)
        : lr(learning_rate), beta1(b1), beta2(b2), eps(epsilon), weight_decay(decay), t(0) {}

    void step(std::vector<Tensor*>& parameters) override {
        t++;
        for (Tensor* param : parameters) {
            // Initialize caches if first time seeing this parameter
            if (!m_cache.contains(param)) {
                m_cache[param] = std::vector<float>(param->data.size(), 0.0f);
                v_cache[param] = std::vector<float>(param->data.size(), 0.0f);
            }
            std::vector<float>& m = m_cache[param];
            std::vector<float>& v = v_cache[param];

            for (size_t i = 0; i < param->data.size(); ++i) {
                float g = param->grad[i];
                // Weight decay update
                param->data[i] -= lr * weight_decay * param->data[i];

                // Update biased first and second moment estimates
                m[i] = beta1 * m[i] + (1.0f - beta1) * g;
                v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

                // Bias corrections
                float m_hat = m[i] / (1.0f - std::pow(beta1, t));
                float v_hat = v[i] / (1.0f - std::pow(beta2, t));

                // Apply parameter update
                param->data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }
};



#endif //ADAMWOPTIMISER_H
