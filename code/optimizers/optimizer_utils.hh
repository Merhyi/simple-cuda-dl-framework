// In a new header, e.g., optimizer_utils.h
#pragma once
#include "..\nn_utils\matrix.hh" // Assuming your Matrix class definition
#include <vector>

struct OptimizableParameter {
    Matrix* parameter_matrix; // Pointer to W or b
    Matrix* gradient_matrix;  // Pointer to dW or db
    // The optimizer might create and manage its own state Matrix (e.g., momentum)
    // associated with this parameter.

    OptimizableParameter(Matrix* param, Matrix* grad)
        : parameter_matrix(param), gradient_matrix(grad) {
    }

    // Helper to get device pointers and size, assuming Matrix has these
    float* get_param_device_ptr() const { return parameter_matrix->data_device.get(); }
    float* get_grad_device_ptr() const { return gradient_matrix->data_device.get(); }
    int get_num_elements() const {
        // Assuming Matrix shape stores (cols, rows) or similar
        return parameter_matrix->shape.x * parameter_matrix->shape.y;
    }
};


// ParameterGroup: A collection of OptimizableParameters
class ParameterGroup {
public:
    std::vector<OptimizableParameter> parameters;

    ParameterGroup() = default;

    // Add a single parameter-gradient pair
    void add_parameter(Matrix* param_matrix, Matrix* grad_matrix) {
        parameters.emplace_back(param_matrix, grad_matrix);
    }

    // Add an already constructed OptimizableParameter
    void add_parameter(const OptimizableParameter& optim_param) {
        parameters.push_back(optim_param);
    }

    // Add all parameters from another group (e.g., merging parameters from different model parts)
    void add_parameters_from_group(const ParameterGroup& other_group) {
        parameters.insert(parameters.end(), other_group.parameters.begin(), other_group.parameters.end());
    }

    // Iterators to allow range-based for loops
    auto begin() { return parameters.begin(); }
    auto end() { return parameters.end(); }
    auto begin() const { return parameters.cbegin(); }
    auto end() const { return parameters.cend(); }
    auto cbegin() const { return parameters.cbegin(); }
    auto cend() const { return parameters.cend(); }

    bool empty() const {
        return parameters.empty();
    }

    size_t size() const {
        return parameters.size();
    }

    void clear() {
        parameters.clear();
    }
};
