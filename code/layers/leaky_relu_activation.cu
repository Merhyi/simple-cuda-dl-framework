// leaky_relu_activation.cu
#include "leaky_relu_activation.hh"
#include "../nn_utils/nn_exception.hh" // For NNException (adjust path if needed)
#include <cuda_runtime.h> // For CUDA specific types and functions
#include <cmath>          // For fmaxf (though direct comparison is often used)

// CUDA kernel for Leaky ReLU forward pass
__global__ void leakyReluActivationForward(const float* Z_device, float* A_device,
    int num_elements, float alpha) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_elements) {
        float z_val = Z_device[index];
        if (z_val > 0) {
            A_device[index] = z_val;
        }
        else {
            A_device[index] = alpha * z_val;
        }
        // Alternative using fmaxf (less direct for Leaky ReLU, but works if alpha is small positive)
        // A_device[index] = fmaxf(z_val, alpha * z_val); // This is not standard LeakyReLU if z_val > 0
        // Correct way for LeakyReLU is an if/else or ternary:
        // A_device[index] = (z_val > 0) ? z_val : (alpha * z_val);
    }
}

// CUDA kernel for Leaky ReLU backward pass
__global__ void leakyReluActivationBackprop(const float* Z_device, const float* dA_device,
    float* dZ_device, int num_elements, float alpha) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_elements) {
        if (Z_device[index] > 0) {
            dZ_device[index] = dA_device[index]; // Derivative is 1, so dL/dZ = dL/dA * 1
        }
        else {
            dZ_device[index] = alpha * dA_device[index]; // Derivative is alpha, so dL/dZ = dL/dA * alpha
        }
    }
}

// --- LeakyReLUActivation Class Implementation ---

LeakyReLUActivation::LeakyReLUActivation(std::string name, float alpha_val)
    : alpha(alpha_val) {
    this->name = name; // Assuming NNLayer base class has a 'name' member or similar
    if (alpha <= 0) { // Alpha should be a small positive constant
        // Optionally throw an exception or log a warning if alpha is not positive
        // For simplicity, we'll allow it but it might not behave as typical LeakyReLU
        // NNException::throwIfDeviceErrorsOccurred("Alpha for LeakyReLU must be positive.");
        // Or set a default if invalid: this->alpha = 0.01f;
    }
}

LeakyReLUActivation::~LeakyReLUActivation() {
    // Memory for A, Z, dZ (if they own it) should be handled by Matrix destructor
    // or their shared_ptr if they use that.
    // If NNLayer has virtual destructor, this is fine.
}

Matrix& LeakyReLUActivation::forward(Matrix& Z_input) {
    // Cache the input Z for use in backpropagation
    // Depending on Matrix implementation, this might be a copy or a reference/shared_ptr assignment
    // For safety and typical use, Z should store a copy or manage its own memory based on Z_input.
    // If Z_input can go out of scope or change before backprop, a deep copy is needed.
    // Assuming Z_input is a temporary result from a previous layer and we need its values for backprop.
    this->Z = Z_input; // This might need to be a deep copy if Z_input is not persistent

    // Allocate memory for the output A if not already allocated or if shape changed
    A.allocateMemoryIfNotAllocated(Z_input.shape);

    int num_elements = Z_input.shape.x * Z_input.shape.y;
    if (num_elements == 0) return A; // Nothing to process

    dim3 block_size(256);
    dim3 num_of_blocks((num_elements + block_size.x - 1) / block_size.x);

    leakyReluActivationForward << <num_of_blocks, block_size >> > (
        Z_input.data_device.get(), // Use Z_input directly for forward pass
        A.data_device.get(),
        num_elements,
        this->alpha);

    NNException::throwIfDeviceErrorsOccurred("LeakyReLUActivation: Cannot perform forward propagation.");
    // cudaDeviceSynchronize(); // Optional: for debugging or immediate error checking

    return A;
}

Matrix& LeakyReLUActivation::backprop(Matrix& dA_input) {
    // learning_rate is typically not used by activation layer's backprop itself,
    // as activation layers usually don't have learnable parameters updated by SGD-like rules.

    // Allocate memory for dZ if not already allocated or if shape changed
    // dZ should have the same shape as Z (and A, and dA_input)
    dZ.allocateMemoryIfNotAllocated(Z.shape);

    int num_elements = Z.shape.x * Z.shape.y;
    if (num_elements == 0) return dZ; // Nothing to process



    dim3 block_size(256);
    dim3 num_of_blocks((num_elements + block_size.x - 1) / block_size.x);

    leakyReluActivationBackprop << <num_of_blocks, block_size >> > (
        Z.data_device.get(),       // Use the cached Z from the forward pass
        dA_input.data_device.get(),
        dZ.data_device.get(),
        num_elements,
        this->alpha);

    NNException::throwIfDeviceErrorsOccurred("LeakyReLUActivation: Cannot perform back propagation.");
    // cudaDeviceSynchronize(); // Optional: for debugging or immediate error checking

    return dZ;
}

void LeakyReLUActivation::get_learnable_parameters(ParameterGroup& group)
{
}