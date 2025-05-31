// adam_optimizer.cu
#include "adam_optimizer.hh" // Also includes optimizer_base.hh, optimizer_utils.hh -> matrix.hh
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>      // For std::pow on host
#include <cstdio>     // For fprintf, stderr
#include <stdexcept>  // For std::runtime_error (optional, for more robust error handling)
#include <iostream>

// Basic CUDA error checking macro
#ifndef cudaSafeCall
#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError_t err, const char* file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA API failed at %s:%i with error: %s (%s)\n",
            file, line, cudaGetErrorString(err), cudaGetErrorName(err));
        // Consider throwing an exception for more robust error handling
        // throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
        // For now, just print and continue, or exit if critical
        // exit(EXIT_FAILURE); 
    }
}
#endif


// --- CUDA Kernel for Adam Update ---
__global__ void adam_update_kernel_impl(
    float* __restrict__ params_dev,      // Parameter tensor (W or b)
    const float* __restrict__ grads_dev, // Gradient tensor (dW or db)
    float* __restrict__ m_dev,           // 1st moment vector (m) on device
    float* __restrict__ v_dev,           // 2nd moment vector (v) on device
    int num_elements,                    // Number of elements in the tensor
    float lr,                            // Learning rate
    float beta1,                         // Exponential decay rate for 1st moment
    float beta2,                         // Exponential decay rate for 2nd moment
    float epsilon,                       // Small constant for numerical stability
    float weight_decay_coeff,            // Weight decay coefficient
    float beta1_t_power,                 // beta1^t (precomputed bias correction term for m)
    float beta2_t_power                  // beta2^t (precomputed bias correction term for v)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        float param_val = params_dev[idx];
        float grad_val = grads_dev[idx];

        // Apply weight decay (L2 regularization style, added to gradient)
        // For true AdamW, weight decay is applied differently (directly to param before Adam step)
        if (weight_decay_coeff > 0.0f) {
            grad_val += weight_decay_coeff * param_val;
        }

        // Update biased first moment estimate
        float m_prev = m_dev[idx];
        float m_t = beta1 * m_prev + (1.0f - beta1) * grad_val;
        m_dev[idx] = m_t;

        // Update biased second raw moment estimate
        float v_prev = v_dev[idx];
        float v_t = beta2 * v_prev + (1.0f - beta2) * grad_val * grad_val;
        v_dev[idx] = v_t;

        // Compute bias-corrected first moment estimate
        // Ensure (1.0f - beta1_t_power) is not zero.
        // Since t >= 1 and beta1 < 1, beta1_t_power < 1, so 1 - beta1_t_power > 0,
        // unless beta1_t_power underflows to a value so close to 1 that it's numerically 1,
        // or if beta1_t_power is exactly 1 (only if t=0 or beta1=1, which are ruled out).
        // A very small denominator is handled by epsilon later.
        float m_hat_t = m_t / (1.0f - beta1_t_power);

        // Compute bias-corrected second raw moment estimate
        float v_hat_t = v_t / (1.0f - beta2_t_power);

        // Update parameters
        params_dev[idx] = param_val - lr * m_hat_t / (sqrtf(v_hat_t) + epsilon);
    }
}

// --- AdamOptimizer Method Implementations ---

AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float epsilon, float weight_decay)
    : lr_(learning_rate),
    beta1_(beta1),
    beta2_(beta2),
    epsilon_(epsilon),
    weight_decay_coeff_(weight_decay),
    t_(0) // Timestep counter starts at 0, will be incremented to 1 before first use
{
    // Basic validation (can be more extensive)
    if (lr_ <= 0.0f) {
        fprintf(stderr, "AdamOptimizer Warning: Learning rate (%.4e) should be > 0.\n", lr_);
    }
    if (!(beta1_ >= 0.0f && beta1_ < 1.0f)) {
        fprintf(stderr, "AdamOptimizer Warning: beta1 (%.4f) should be in [0, 1).\n", beta1_);
    }
    if (!(beta2_ >= 0.0f && beta2_ < 1.0f)) {
        fprintf(stderr, "AdamOptimizer Warning: beta2 (%.4f) should be in [0, 1).\n", beta2_);
    }
    if (epsilon_ <= 0.0f) {
        fprintf(stderr, "AdamOptimizer Warning: epsilon (%.4e) should be > 0.\n", epsilon_);
    }
    if (weight_decay_coeff_ < 0.0f) {
        fprintf(stderr, "AdamOptimizer Warning: weight_decay_coeff (%.4e) should be >= 0.\n", weight_decay_coeff_);
    }
}

void AdamOptimizer::set_parameter_group(const ParameterGroup& group) {
    params_group_to_optimize_ = group; // Copy the group

    // Clear existing state buffers and reset timestep
    m_buffers_.clear();
    v_buffers_.clear();
    t_ = 0;


    for (const auto& optim_param : params_group_to_optimize_) {
        Matrix* param_matrix_ptr = optim_param.parameter_matrix;
        if (!param_matrix_ptr) {
            fprintf(stderr, "AdamOptimizer Error: Encountered a null parameter matrix in set_parameter_group.\n");
            // Potentially throw an exception or handle error more gracefully
            // For now, skip this parameter
            continue;
        }

        // Assuming Matrix shape stores (cols, rows) or (width, height)
        // And constructor takes (rows, cols, on_device)
        int rows = param_matrix_ptr->shape.y;
        int cols = param_matrix_ptr->shape.x;
        

        if (rows <= 0 || cols <= 0) {
            fprintf(stderr, "AdamOptimizer Warning: Parameter matrix has invalid dimensions (%d x %d). Skipping state buffer creation.\n", rows, cols);
            continue;
        }

        // Create m buffer (first moment)
        // Use emplace to construct Matrix in-place in the map.
        // The key is const Matrix* (pointer to the parameter matrix being optimized).
        // The value is a Matrix object (the m_buffer itself).
        auto m_it_pair = m_buffers_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(param_matrix_ptr),       // Arguments for key construction
            std::forward_as_tuple(rows, cols) // Arguments for Matrix value construction
        );

        if (m_it_pair.second) { // True if insertion took place
            // m_it_pair.first is an iterator to the inserted element (std::map::value_type*)
            // m_it_pair.first->second is the Matrix object (the m buffer)
            // Assuming Matrix has a method like `fill_device(float value)` to zero-initialize
            m_it_pair.first->second.allocateMemory();
            m_it_pair.first->second.fill_zeros_on_device();
        }
        else {
            // This case should ideally not happen if clear() was called and keys are unique.
            fprintf(stderr, "AdamOptimizer Error: Failed to emplace m_buffer for a parameter (key collision?).\n");
        }

        // Create v buffer (second moment)
        auto v_it_pair = v_buffers_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(param_matrix_ptr),       // Arguments for key construction
            std::forward_as_tuple(rows, cols) // Arguments for Matrix value construction
        );

        if (v_it_pair.second) {
            v_it_pair.first->second.allocateMemory();
            v_it_pair.first->second.fill_zeros_on_device();
        }
        else {
            fprintf(stderr, "AdamOptimizer Error: Failed to emplace v_buffer for a parameter (key collision?).\n");
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











void AdamOptimizer::step() {
    if (params_group_to_optimize_.empty()) {
        return; // Nothing to optimize
    }

    t_++; // Increment timestep (1-indexed for first step)

    for (const auto& optim_param : params_group_to_optimize_) {
        Matrix* param_matrix_ptr = optim_param.parameter_matrix;
        Matrix* grad_matrix_ptr = optim_param.gradient_matrix;

        // Find the m and v buffers for the current parameter
        auto m_iter = m_buffers_.find(param_matrix_ptr);
        auto v_iter = v_buffers_.find(param_matrix_ptr);

        Matrix& m_buffer = m_iter->second; // Reference to the m_buffer Matrix object
        Matrix& v_buffer = v_iter->second; // Reference to the v_buffer Matrix object

        float* param_data_dev = optim_param.get_param_device_ptr();
        const float* grad_data_dev = optim_param.get_grad_device_ptr();
        float* m_data_dev = m_buffer.data_device.get(); // Assuming Matrix has data_device.get()
        float* v_data_dev = v_buffer.data_device.get(); // Assuming Matrix has data_device.get()
        int num_elements = optim_param.get_num_elements();

        /*std::cout << std::endl << "Before Update: num_elements " << num_elements << std::endl;
        std::cout << std::endl << "Parameter Matrix: " << std::endl;
        printMatrixCUDA(*optim_param.parameter_matrix);
        std::cout << std::endl << "Gradient Matrix: " << std::endl;
        printMatrixCUDA(*optim_param.gradient_matrix);
        std::cout << std::endl << "Optimizer: Address of gradient_matrix in OptimizableParameter: "
            << static_cast<void*>(optim_param.get_grad_device_ptr()) << std::endl;*/

        // Call the static host wrapper to launch the CUDA kernel
        AdamOptimizer::launch_adam_update_kernel(
            param_data_dev,
            grad_data_dev,
            m_data_dev,
            v_data_dev,
            num_elements,
            lr_,
            beta1_,
            beta2_,
            epsilon_,
            weight_decay_coeff_,
            t_ // Pass the current timestep t (1-indexed)
        );

        /*std::cout << std::endl << "After Update: " << std::endl;
        std::cout << std::endl << "Parameter Matrix: " << std::endl;
        printMatrixCUDA(*optim_param.parameter_matrix);
        std::cout << std::endl << "Gradient Matrix: " << std::endl;
        printMatrixCUDA(*optim_param.gradient_matrix);*/




    }
    // After all kernels for this step are launched, you might check for errors once.
    cudaDeviceSynchronize(); // Synchronize to catch any async kernel errors
}

// Static host wrapper to launch the CUDA kernel
void AdamOptimizer::launch_adam_update_kernel(
    float* param_data_dev,
    const float* grad_data_dev,
    float* m_data_dev,
    float* v_data_dev,
    int num_elements,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay_coeff,
    int current_t // current_t is the 1-indexed timestep
) {
    if (num_elements == 0) {
        return;
    }

    // Precompute powers of beta1 and beta2 for bias correction on the host
    // std::pow can be sensitive for large 'current_t' or extreme beta values.
    // Ensure beta1 and beta2 are < 1.0 to prevent beta_t_power from becoming 1 or >1.
    float beta1_t_power = (beta1 < 1.0f) ? std::pow(beta1, static_cast<float>(current_t)) : 0.0f;
    float beta2_t_power = (beta2 < 1.0f) ? std::pow(beta2, static_cast<float>(current_t)) : 0.0f;

    // If beta1 or beta2 is 1.0 (should be caught by constructor warnings), pow(1.0, t) = 1.0.
    // This would make 1.0 - beta_t_power = 0, leading to division by zero in the kernel.
    // The constructor warns, but defensive calculation here is good.
    // If current_t is very large, beta_t_power will approach 0.
    if (1.0f - beta1_t_power == 0.0f && beta1 < 1.0f) { // Should only happen if beta1_t_power underflows to 1, very unlikely
        // This case is highly unlikely if beta1 < 1.0.
        // If beta1 is indeed 1.0, it's problematic for Adam.
        // fprintf(stderr, "Warning: (1.0 - beta1^t) is zero. beta1=%f, t=%d\n", beta1, current_t);
    }
    if (1.0f - beta2_t_power == 0.0f && beta2 < 1.0f) {
        // fprintf(stderr, "Warning: (1.0 - beta2^t) is zero. beta2=%f, t=%d\n", beta2, current_t);
    }


    // Configure kernel launch parameters
    const int threads_per_block = 256; // Common choice, can be tuned
    // Ensure num_blocks is at least 1, even if num_elements < threads_per_block
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    adam_update_kernel_impl << <num_blocks, threads_per_block >> > (
        param_data_dev,
        grad_data_dev,
        m_data_dev,
        v_data_dev,
        num_elements,
        lr,
        beta1,
        beta2,
        epsilon,
        weight_decay_coeff,
        beta1_t_power,
        beta2_t_power
        );

    // Check for kernel launch errors (asynchronous, use cudaGetLastError)
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed in AdamOptimizer::launch_adam_update_kernel: %s\n",
            cudaGetErrorString(launch_err));
    }
    // Note: A cudaDeviceSynchronize() here would make errors synchronous and easier to debug,
    // but it impacts performance. It's better to synchronize at a higher level if needed,
    // or after all optimizer steps in a training iteration.
    // The cudaDeviceSynchronize() in step() will catch these kernel errors.
}