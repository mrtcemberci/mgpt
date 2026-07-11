#ifndef ADAMWOPTIMISER_H
#define ADAMWOPTIMISER_H
#include <cmath>
#include <fstream>
#include <unordered_map>

#include "Optimiser.h"
#include "Tensor.h"
#include "../utils/cuda_ops.h"


class AdamWOptimizer : public Optimiser {
private:
    float lr, beta1, beta2, eps, weight_decay;
    int t; // Step counter
    std::unordered_map<Tensor*, Tensor> m_cache;
    std::unordered_map<Tensor*, Tensor> v_cache;
public:
    AdamWOptimizer(float learning_rate = 3e-4f, float b1 = 0.9f, float b2 = 0.999f,
                   float epsilon = 1e-8f, float decay = 0.01f)
        : lr(learning_rate), beta1(b1), beta2(b2), eps(epsilon), weight_decay(decay), t(0) {}

    void step(std::vector<Tensor*>& parameters) override {
        t++;
        for (Tensor* param : parameters) {
            // Initialize caches if first time seeing this parameter
            if (!m_cache.contains(param)) {
                m_cache[param] = Tensor(param->shape, 0.0f, param->device);
                v_cache[param] = Tensor(param->shape, 0.0f, param->device);
            }
            Tensor& m = m_cache[param];
            Tensor& v = v_cache[param];
            param->adamw_step(m, v, lr, beta1, beta2, eps, weight_decay, t);
        }
    }

    void set_lr(float new_lr) override { lr = new_lr; }
    float get_lr() const override { return lr; }

    void save_state(const std::string& filepath, const std::vector<Tensor*>& parameters) {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open()) return;
        out.write((char*)&t, sizeof(int));
        for (Tensor* param : parameters) {
            if (!m_cache.contains(param)) {
                m_cache[param] = Tensor(param->shape, 0.0f, param->device);
                v_cache[param] = Tensor(param->shape, 0.0f, param->device);
            }
            Tensor& m = m_cache[param];
            Tensor& v = v_cache[param];
            if (param->device == Device::CUDA) {
                std::vector<float> h_m(m.size()), h_v(v.size());
                cuda_ops::copy_device_to_host(h_m.data(), m.get_data_ptr(), m.size());
                cuda_ops::copy_device_to_host(h_v.data(), v.get_data_ptr(), v.size());
                out.write((char*)h_m.data(), m.size() * sizeof(float));
                out.write((char*)h_v.data(), v.size() * sizeof(float));
            } else {
                out.write((char*)m.data.data(), m.size() * sizeof(float));
                out.write((char*)v.data.data(), v.size() * sizeof(float));
            }
        }
        out.close();
        std::cout << "Successfully saved AdamW optimizer state (step " << t << ") to " << filepath << "!\n";
    }

    void load_state(const std::string& filepath, const std::vector<Tensor*>& parameters) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open()) return;
        in.read((char*)&t, sizeof(int));
        for (Tensor* param : parameters) {
            if (!m_cache.contains(param)) {
                m_cache[param] = Tensor(param->shape, 0.0f, param->device);
                v_cache[param] = Tensor(param->shape, 0.0f, param->device);
            }
            Tensor& m = m_cache[param];
            Tensor& v = v_cache[param];
            if (param->device == Device::CUDA) {
                std::vector<float> h_m(m.size()), h_v(v.size());
                in.read((char*)h_m.data(), m.size() * sizeof(float));
                in.read((char*)h_v.data(), v.size() * sizeof(float));
                cuda_ops::copy_host_to_device(m.get_data_ptr(), h_m.data(), m.size());
                cuda_ops::copy_host_to_device(v.get_data_ptr(), h_v.data(), v.size());
            } else {
                in.read((char*)m.data.data(), m.size() * sizeof(float));
                in.read((char*)v.data.data(), v.size() * sizeof(float));
            }
        }
        in.close();
        std::cout << "Successfully loaded AdamW optimizer state (step " << t << ") from " << filepath << "!\n";
    }
};



#endif //ADAMWOPTIMISER_H
