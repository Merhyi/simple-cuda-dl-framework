// tanh_activation.cu
#include "tanh_activation.hh"
#include "../nn_utils/nn_exception.hh" // For NNException (or your error handling)
#include <cmath>                       // For tanhf on host (not directly used in kernel but good include)

// It's good practice to include cuda_runtime.h for __global__, blockIdx, etc.
#include <cuda_runtime.h>


// --- CUDA Kernels ---

/**
 * @brief CUDA kernel for Tanh forward activation.
 *        A[index] = tanhf(Z[index])
 * @param Z_device Pointer to the input matrix Z data on the device.
 * @param A_device Pointer to the output matrix A data on the device.
 * @param num_elements Total number of elements in the matrices.
 */
__global__ void tanhActivationForwardKernel(const float* Z_device, float* A_device,
    int num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_elements) {
        A_device[index] = tanhf(Z_device[index]); // tanhf is the float version of tanh
    }
}

/**
 * @brief CUDA kernel for Tanh backward propagation.
 *        dZ[index] = dA[index] * (1 - A_val^2), where A_val is Tanh(Z[index])
 *                    from the forward pass.
 * @param A_forward_device Pointer to the activation output A from the forward pass (device).
 *                         This is Tanh(Z).
 * @param dA_input_device Pointer to the incoming gradient dL/dA (device).
 * @param dZ_output_device Pointer to store the outgoing gradient dL/dZ (device).
 * @param num_elements Total number of elements in the matrices.
 */
__global__ void tanhActivationBackpropKernel(const float* A_forward_device,
    const float* dA_input_device,
    float* dZ_output_device,
    int num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_elements) {
        float tanh_Z_val = A_forward_device[index]; // This is A = Tanh(Z)
        float derivative_tanh_Z = 1.0f - (tanh_Z_val * tanh_Z_val);
        dZ_output_device[index] = dA_input_device[index] * derivative_tanh_Z;
    }
}


// --- TanhActivation Class Method Implementations ---

TanhActivation::TanhActivation(std::string name) {
    this->name = name;
    // A and dZ will be allocated on demand in forward/backprop
}

TanhActivation::~TanhActivation() {
    // Matrix members A and dZ should handle their own memory deallocation
    // in their destructors if they manage raw pointers.
    // If they are stack-allocated or use smart pointers managing device memory,
    // this destructor can be empty.
}

Matrix& TanhActivation::forward(Matrix& Z_input) {
    // Store a reference or copy Z if needed by the backprop logic that uses original Z.
    // For Tanh, the derivative is 1 - Tanh(Z)^2, so we only need Tanh(Z), which is A.
    // The ReLU example caches Z because its derivative (0 or 1) depends directly on Z's sign.
    // this->Z_cache = Z_input; // Uncomment and declare Z_cache if Z_input itself is needed later

    // Ensure output matrix A is allocated with the same shape as input Z_input
    A.allocateMemoryIfNotAllocated(Z_input.shape);

    int num_elements = Z_input.shape.y * Z_input.shape.x;
    if (num_elements == 0) {
        // Handle empty input if necessary, or let downstream error handling catch it
        return A;
    }

    dim3 block_size(256); // Common block size
    dim3 num_of_blocks((num_elements + block_size.x - 1) / block_size.x);

    tanhActivationForwardKernel << <num_of_blocks, block_size >> > (
        Z_input.data_device.get(), // Assuming Matrix has data_device.get() for device pointer
        A.data_device.get(),
        num_elements
        );

    // Error checking (using your NNException or a similar mechanism)
    NNException::throwIfDeviceErrorsOccurred("TanhActivation: Error during forward propagation CUDA kernel launch.");
    // cudaDeviceSynchronize(); // Optional: for more robust error checking immediately, but can slow down.
                              // throwIfDeviceErrorsOccurred likely calls cudaGetLastError.

    return A;
}

Matrix& TanhActivation::backprop(Matrix& dA_input) {
    // Ensure dZ matrix is allocated with the same shape as A (or original Z_input)
    // A.shape should be valid and set from the forward pass.
    dZ.allocateMemoryIfNotAllocated(A.shape);

    int num_elements = A.shape.y * A.shape.x;
    if (num_elements == 0) {
        return dZ; // Handle empty input
    }

    dim3 block_size(256);
    dim3 num_of_blocks((num_elements + block_size.x - 1) / block_size.x);

    // For backprop, we need A (which is Tanh(Z) from the forward pass)
    // and dA_input (the incoming gradient dL/dA).
    tanhActivationBackpropKernel << <num_of_blocks, block_size >> > (
        A.data_device.get(),         // Pass A (Tanh(Z) from forward)
        dA_input.data_device.get(),  // Pass dL/dA
        dZ.data_device.get(),        // Output dL/dZ
        num_elements
        );

    NNException::throwIfDeviceErrorsOccurred("TanhActivation: Error during backpropagation CUDA kernel launch.");
    // cudaDeviceSynchronize(); // Optional

    return dZ;
}

void TanhActivation::get_learnable_parameters(ParameterGroup& group) {
    // Tanh activation has no learnable parameters, so this function is empty.
    (void)group; // Suppress unused parameter warning if 'group' is not used.
}