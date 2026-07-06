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
            param->adamw_step(m, v, lr, beta1, beta2, eps, weight_decay, t);
        }
    }
};



#endif //ADAMWOPTIMISER_H
