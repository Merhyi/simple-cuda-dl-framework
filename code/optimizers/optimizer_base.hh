// In optimizer.h (or a new sgd_optimizer.h)
#pragma once
#include "optimizer_utils.hh" // For OptimizableParameter
#include <vector>
#include <map> // For state like momentum buffers

// Forward declaration if Matrix is complex
// class Matrix;

class OptimizerBase {
public:
    virtual ~OptimizerBase() = default;
    virtual void step() = 0;
    virtual void zero_param_gradients() { /* Default: let training loop handle it */ }
    // Optimizer does not zero gradients directly on layer's dW/db.
    // The layer or training loop calls layer.zero_gradients().
};