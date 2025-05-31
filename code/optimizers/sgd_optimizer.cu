// In sgd_optimizer.cpp (or sgd_optimizer.cu if kernel is defined here)
#include "sgd_optimizer.hh"

#include <iostream>
#include <cuda_runtime.h> // For CUDA calls if not in matrix.h
// Include your NNException if used for error checking CUDA calls

// sgd_update_kernel implementation (as defined before)
__global__ void sgd_update_kernel(
    float* params_data,
    const float* grads_data,
    float* momentum_buffer,
    int num_elements,
    float learning_rate,
    float momentum_coeff,
    float weight_decay_coeff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float grad_val = grads_data[idx];
        //if (idx == 0 && blockIdx.x == 0 && blockIdx.y == 0) { // Ensure only one thread prints
        //    printf("Kernel: grads_data pointer: %p\n", grads_data);
        //    if (num_elements > 0) {
        //        printf("Kernel: grads_data[0] = %f\n", grads_data[0]); // Print first element
        //    }
        //}
        //printf("Grad Val: %f\n", grad_val);
        //printf("Before params_data[%d]: %f\n", idx, params_data[idx]);

        // Apply weight decay (L2 regularization) before momentum
        if (weight_decay_coeff > 0.0f && params_data != nullptr) { // Check params_data for safety
            grad_val += weight_decay_coeff * params_data[idx];
        }

        if (momentum_coeff > 0.0f && momentum_buffer != nullptr) {
            momentum_buffer[idx] = momentum_coeff * momentum_buffer[idx] + grad_val;
            grad_val = momentum_buffer[idx]; // Use momentum-updated gradient
        }
        if (params_data != nullptr) { // Final safety check
            params_data[idx] -= learning_rate * grad_val;
        }
        /*printf("After params_data[%d]: %f\n", idx, params_data[idx]);*/
    }
}

void SGDOptimizer::launch_sgd_update_kernel(
    float* param_data, const float* grad_data, float* momentum_data,
    int num_elements, float lr, float momentum_coeff, float weight_decay_coeff)
{
    if (num_elements <= 0) return; // Nothing to update
    const int threads_per_block = 256; // Or a configurable/adaptive value
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    sgd_update_kernel << <blocks_per_grid, threads_per_block >> > (
        param_data, grad_data, momentum_data,
        num_elements, lr, momentum_coeff, weight_decay_coeff
        );
    // cudaDeviceSynchronize(); // Usually not here, but after all optimizer steps or kernel launches in a batch
    // NNException::throwIfDeviceErrorsOccurred("SGD kernel launch failed."); // Check after sync
}


SGDOptimizer::SGDOptimizer(float learning_rate, float momentum, float weight_decay)
    : lr_(learning_rate), momentum_coeff_(momentum), weight_decay_coeff_(weight_decay) {
    if (lr_ < 0.0f) {
        throw std::invalid_argument("Learning rate cannot be negative.");
    }
    if (momentum_coeff_ < 0.0f || momentum_coeff_ >= 1.0f) { // Momentum typically [0, 1)
        // Allow momentum == 0 (no momentum)
        if (momentum_coeff_ != 0.0f)
            throw std::invalid_argument("Momentum coefficient should be in [0.0, 1.0).");
    }
    if (weight_decay_coeff_ < 0.0f) {
        throw std::invalid_argument("Weight decay coefficient cannot be negative.");
    }
}






void SGDOptimizer::set_parameter_group(const ParameterGroup& group) {
    params_group_to_optimize_ = group; // Makes a copy of the group
    momentum_buffers_.clear();      // Clear any state from previous groups

    if (momentum_coeff_ > 0.0f) {
        for (const auto& optim_param : params_group_to_optimize_) {

            // Create a momentum buffer Matrix with the same shape as the parameter
            // The Matrix::AsGradientFor is a good utility here if it just allocates and zeros
            // Or a dedicated constructor for state buffers.
            Matrix momentum_buffer_matrix(optim_param.parameter_matrix->shape);
            momentum_buffer_matrix.allocateMemory(); // Constructor should handle this
            momentum_buffer_matrix.fill_zeros_on_device(); // Ensure it's zeroed out

            // Store it, keyed by the parameter_matrix's pointer for uniqueness
            momentum_buffers_[optim_param.parameter_matrix] = std::move(momentum_buffer_matrix);
            
        }
    }
}



// 你的 printMatrix 函数，它接收一个 Matrix 对象引用
static void printMatrixCUDA(const Matrix& mat_to_print) {
    // ... rest of the logic using mat_to_print.data_device.get() and cudaMemcpy ...
    int count = mat_to_print.shape.x * mat_to_print.shape.y;

    std::vector<float> h_data(count);
    printf("Pointer: %p\n", mat_to_print.data_device.get());
    cudaError_t err = cudaMemcpy(h_data.data(), mat_to_print.data_device.get(), count * sizeof(float), cudaMemcpyDeviceToHost);
    for (auto data : h_data)
    {
        std::cout << data << "\t";
    }
}



void SGDOptimizer::step() {
    if (params_group_to_optimize_.empty()) {
        // Or log a warning: std::cerr << "Optimizer step called with no parameters." << std::endl;
        return;
    }

    for (const auto& optim_param : params_group_to_optimize_) {

        float* param_ptr = optim_param.get_param_device_ptr();
        float* grad_ptr = optim_param.get_grad_device_ptr();
        int num_elements = optim_param.get_num_elements();

        if (num_elements == 0) { // Skip if parameter has no elements
            continue;
        }
        if (param_ptr == nullptr || grad_ptr == nullptr) {
            // Should have been caught by isValid or earlier checks, but defensive
            std::cerr << "Warning: Skipping parameter update due to null data/grad pointer." << std::endl;
            continue;
        }


        float* momentum_ptr = nullptr;
        if (momentum_coeff_ > 0.0f) {
            auto it = momentum_buffers_.find(optim_param.parameter_matrix);
            if (it != momentum_buffers_.end()) {
                momentum_ptr = it->second.data_device.get(); // Get data pointer of the momentum Matrix
            }
            else {
                // This would be an internal error if momentum is enabled but buffer wasn't created
                throw std::runtime_error("Momentum buffer not found for a parameter during optimizer step. Parameter was likely not added correctly or momentum state was cleared.");
            }
        }

        
        
        /*std::cout << std::endl << "Before Update: num_elements " << num_elements << std::endl;
        printMatrixCUDA(*optim_param.gradient_matrix);
        std::cout << "Optimizer: Address of gradient_matrix in OptimizableParameter: "
            << static_cast<void*>(optim_param.get_grad_device_ptr()) << std::endl;*/

        
        // Launch the CUDA kernel for this specific parameter
        launch_sgd_update_kernel(param_ptr, grad_ptr, momentum_ptr,
            num_elements, lr_, momentum_coeff_, weight_decay_coeff_);

        /*cudaDeviceSynchronize();
        std::cout << "After Update: " << std::endl;
        printMatrix(*optim_param.parameter_matrix);
        printMatrix(*optim_param.gradient_matrix);*/

        
    }

    cudaDeviceSynchronize();

    // It's generally better to synchronize outside the optimizer's step,
    // e.g., once per training batch after all kernels (forward, backward, optimizer) are launched,
    // to allow maximum parallelism. But for simplicity or debugging, it can be here.
    // cudaDeviceSynchronize();
    // NNException::throwIfDeviceErrorsOccurred("Error occurred during optimizer step kernel execution.");
}