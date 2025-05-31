
// main_test.cpp
#include ".\optimizers\optimizer_base.hh"
#include ".\optimizers\adam_optimizer.hh" // Your AdamOptimizer header
#include <iostream>
#include <vector>
#include <cmath>    // For std::sqrt, std::pow, std::fabs
#include <iomanip>  // For std::fixed, std::setprecision
#include <numeric>  // For std::iota
#include <algorithm> // For std::generate







int main() {
    const size_t num_params = 10; // Number of parameters to optimize
    const float learning_rate = 0.01f;
    const int batch_size = 4;
    const int num_steps = 5; // Number of optimization steps to simulate

    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;

    std::cout << std::fixed << std::setprecision(7);

    // --- Initialize Host Data ---
    std::vector<float> h_params(num_params);
    std::vector<float> h_grad_sums(num_params);
    // For CPU verification
    std::vector<float> h_params_cpu_ref(num_params);

    // Fill with some initial values
    std::generate(h_params.begin(), h_params.end(), [n = 0.0f]() mutable { return n += 0.1f; });

    // --- Allocate GPU Memory ---
    float* d_params, * d_grad_sums;
    cudaMalloc(&d_params, num_params * sizeof(float));
    cudaMalloc(&d_grad_sums, num_params * sizeof(float));

    // --- Copy initial params to GPU ---
    cudaMemcpy(d_params, h_params.data(), num_params * sizeof(float), cudaMemcpyHostToDevice);

    // --- Create Optimizer ---
    // We use a raw pointer here for OptimizerBase to show polymorphism.
    // In a real app, std::unique_ptr for NeuralNetwork to own optimizer is better.
    OptimizerBase* optimizer = new AdamOptimizer(num_params, beta1, beta2, epsilon);
    // ^ Pass total number of parameters this optimizer instance will manage.

    std::cout << "Initial Parameters (Host): ";
    for (float p : h_params) std::cout << p << " ";
    std::cout << std::endl;

    // --- Simulation Loop ---
    for (int step = 0; step < num_steps; ++step) {
        std::cout << "\n--- Step " << step + 1 << " ---" << std::endl;

        // Simulate some gradients for this step
        for (size_t i = 0; i < num_params; ++i) {
            // Simple varying gradients: e.g., (param_value * 0.2 + 0.05 * step) * batch_size
            // grad_sums are SUM over batch, so multiply by batch_size
            h_grad_sums[i] = (h_params_cpu_ref[i] * 0.2f + 0.05f * (step + 1)) * static_cast<float>(batch_size);
        }
        cudaMemcpy(d_grad_sums, h_grad_sums.data(), num_params * sizeof(float), cudaMemcpyHostToDevice);

        std::cout << "  Grad Sums (Host): ";
        for (float g : h_grad_sums) std::cout << g / batch_size << "(avg) "; // Print average gradient
        std::cout << std::endl;

        // Perform optimizer step on GPU
        optimizer->step(d_params, d_grad_sums, num_params, learning_rate, batch_size);
        // For Adam, the internal 't' in optimizer object increments.

        // Copy updated params from GPU back to Host for printing/verification
        std::vector<float> h_params_from_gpu(num_params);
        cudaMemcpy(h_params_from_gpu.data(), d_params, num_params * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize(); // Ensure all GPU work is done


        std::cout << "  Params (GPU after step): ";
        for (float p : h_params_from_gpu) std::cout << p << " ";
        std::cout << std::endl;

    }

    // --- Cleanup ---
    delete optimizer; // Delete through base class pointer
    cudaFree(d_params);
    cudaFree(d_grad_sums);

    std::cout << "\nTest finished." << std::endl;
    return 0;
}