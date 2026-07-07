#ifndef SGDOPTIMISER_H
#define SGDOPTIMISER_H
#include "Optimiser.h"
#include "Tensor.h"


class SGDOptimizer : public Optimiser {
private:
    float lr;
public:
    explicit SGDOptimizer(float learning_rate) : lr(learning_rate) {}

    void step(std::vector<Tensor*>& parameters) override {
        for (Tensor* param : parameters) {
            param->sgd_step(lr);
        }
    }

    void set_lr(float new_lr) override { lr = new_lr; }
    float get_lr() const override { return lr; }
};

#endif //SGDOPTIMISER_H
