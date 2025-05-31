// adam_optimizer.hh
#pragma once

#include "optimizer_base.hh" // Includes optimizer_utils.hh, which in turn includes Matrix and ParameterGroup
#include <map>               // For std::map to store Adam's state buffers (m and v)


class AdamOptimizer : public OptimizerBase {
public:

    AdamOptimizer(float learning_rate = 0.001f,
                  float beta1 = 0.9f,
                  float beta2 = 0.999f,
                  float epsilon = 1e-8f,
                  float weight_decay = 0.0f);


    void set_parameter_group(const ParameterGroup& group);

    void step() override;


private:
    float lr_;                  // Learning rate
    float beta1_;               // Coefficient for first moment estimate
    float beta2_;               // Coefficient for second moment estimate
    float epsilon_;             // Small constant for numerical stability
    float weight_decay_coeff_;  // Weight decay (L2 regularization) coefficient

    ParameterGroup params_group_to_optimize_; // Stores optimizable parameters

    std::map<const Matrix*, Matrix> m_buffers_; // Stores E[g]
    std::map<const Matrix*, Matrix> v_buffers_; // Stores E[g^2]
    
    int t_; // Global timestep counter, incremented at each call to step(). Used for bias correction.


    static void launch_adam_update_kernel(
        float* param_data,
        const float* grad_data,
        float* m_data,
        float* v_data,
        int num_elements,
        float lr,
        float beta1,
        float beta2,
        float epsilon,
        float weight_decay_coeff,
        int current_t
    );
};