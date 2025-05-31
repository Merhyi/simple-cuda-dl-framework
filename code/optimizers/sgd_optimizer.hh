#pragma once
#include "optimizer_base.hh"



// In optimizer.h / sgd_optimizer.h

// (OptimizerBase definition as before)

class SGDOptimizer : public OptimizerBase {
public:
    // Constructor takes learning rate, etc.
    SGDOptimizer(float learning_rate, float momentum = 0.0f, float weight_decay = 0.0f);

    // Takes a list of Matrix pointers that are learnable parameters
    void set_parameter_group(const ParameterGroup& group);

    void step() override;

private:
    float lr_;
    float momentum_coeff_;
    float weight_decay_coeff_;

    ParameterGroup params_group_to_optimize_;
    std::map<const Matrix*, Matrix> momentum_buffers_;

    // CUDA kernel for SGD update (as before)
    static void launch_sgd_update_kernel(
        float* param_data, const float* grad_data, float* momentum_data,
        int num_elements, float lr, float momentum_coeff, float weight_decay_coeff);
};