#ifndef OPTIMISER_H
#define OPTIMISER_H



#include <vector>
#include "Tensor.h"

class Optimiser {
public:
    virtual ~Optimiser() = default;

    // Execute one optimization step across all model parameters
    virtual void step(std::vector<Tensor*>& parameters) = 0;
    virtual void set_lr(float new_lr) = 0;
    virtual float get_lr() const = 0;
};


#endif //OPTIMISER_H
