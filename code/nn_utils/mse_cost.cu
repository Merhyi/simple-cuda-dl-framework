
#include "mse_cost.hh"
#include "nn_exception.hh"

#include <math.h>
#include <iostream>
#include <assert.h>


//__global__ void meanSquareErrorCost(
//    const float* predictions_dev,
//    const float* target_dev,
//    int N,
//    float* d_output_sum_sq_error)
//{
//    // Calculate the global thread index for a 1D grid of 1D blocks
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//    // Boundary check: ensure the thread is within the valid range of elements
//    if (idx < N) {
//        // Calculate the difference between prediction and target
//        float diff = predictions_dev[idx] - target_dev[idx];
//
//        // Calculate the squared error
//        float squared_error = diff * diff;
//
//        // Atomically add the local squared error to the global sum.
//        // This is crucial to prevent race conditions when multiple threads
//        // try to update the same memory location (d_output_sum_sq_error) concurrently.
//        atomicAdd(d_output_sum_sq_error, squared_error);
//    }
//}


__global__ void meanSquareErrorCostKernel(
    const float* __restrict__ y_pred,
    const float* __restrict__ y_true,
    int N_total_elements,
    float* __restrict__ d_block_partial_sums // Output for per-block sum of squared errors
) {
    extern __shared__ float s_cache[]; // Shared memory for in-block reduction

    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_local = threadIdx.x;

    if (tid_global < N_total_elements) {
        float pred_val = y_pred[tid_global];
        float true_val = y_true[tid_global];
        float diff = pred_val - true_val;
        s_cache[tid_local] = diff * diff;
    }
    else {
        s_cache[tid_local] = 0.0f; // Threads outside range contribute 0 to sum
    }

    __syncthreads();

    // In-block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid_local < s) {
            s_cache[tid_local] += s_cache[tid_local + s];
        }
        __syncthreads();
    }

    // First thread in each block writes the block's partial sum
    if (tid_local == 0) {
        d_block_partial_sums[blockIdx.x] = s_cache[0];
    }
}






//__global__ void dMeanSquareErrorCost(
//    const float* predictions_dev,
//    const float* target_dev,
//    float* dY_dev, // Output: derivative w.r.t. predictions
//    int N_total_elements)
//{
//    // Calculate the global thread index for a 1D grid of 1D blocks
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//    // Boundary check: ensure the thread is within the valid range of elements
//    if (idx < N_total_elements) {
//        // Get the prediction and target for the current element
//        float prediction = predictions_dev[idx];
//        float target = target_dev[idx];
//
//        // Calculate the difference
//        float diff = prediction - target;
//
//        // Calculate the derivative: (2/N) * diff
//        // Ensure N_total_elements is not zero to avoid division by zero.
//        // The host code should ideally prevent N=0 cases.
//        if (N_total_elements > 0) {
//            dY_dev[idx] = (2.0f / static_cast<float>(N_total_elements)) * diff;
//        }
//        else {
//            // Handle N=0 case if necessary, e.g., set derivative to 0
//            // or rely on host to not call with N=0.
//            // If N_total_elements can be 0 and this is not an error,
//            // this branch is important.
//            dY_dev[idx] = 0.0f;
//        }
//    }
//}

__global__ void dMeanSquareErrorCostKernel(
    const float* __restrict__ y_pred,
    const float* __restrict__ y_true,
    float* __restrict__ d_cost_d_pred,      // Output for derivative
    int N_total_elements
) {
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x; // Global thread ID

    if (tid_global < N_total_elements) {
        float pred_val = y_pred[tid_global];
        float true_val = y_true[tid_global];
        float diff = pred_val - true_val;

        d_cost_d_pred[tid_global] = (2.0f / static_cast<float>(N_total_elements)) * diff;
    }
}





//float MSECost::cost(Matrix predictions, Matrix target) {
//    assert(predictions.shape.x == target.shape.x);
//    // 假设 predictions.shape.x 是元素的总数 N
//    int N = predictions.shape.x;
//
//    if (N == 0) { // 处理空输入的情况
//        return 0.0f;
//    }
//
//    float* d_cost_sum_sq_error; // 使用 d_ 前缀表示设备指针更清晰
//    // 使用 cudaMallocManaged 使得宿主和设备都能直接访问该指针，简化了内存操作
//    // 也可以使用 cudaMalloc 后跟 cudaMemcpy 来将结果拷回宿主
//    cudaError_t err = cudaMallocManaged(&d_cost_sum_sq_error, sizeof(float));
//
//
//    *d_cost_sum_sq_error = 0.0f; // 初始化GPU上的累加器为0
//
//    dim3 block_size(256); // 一个常用的块大小
//    // 计算需要的块数，确保覆盖所有N个元素
//    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x);
//
//    // 调用CUDA kernel
//    // meanSquareErrorCost<<<num_of_blocks, block_size>>>(...); // kernel定义如上
//    meanSquareErrorCost << <num_of_blocks, block_size >> > (
//        predictions.data_device.get(), // 假设 .get() 返回原始设备指针 float*
//        target.data_device.get(),
//        N,
//        d_cost_sum_sq_error
//        );
//
//    // 检查kernel启动是否有立即错误 (可选，cudaDeviceSynchronize会捕获异步错误)
//    // NNException::throwIfDeviceErrorsOccurred("Error during meanSquareErrorCost kernel launch.");
//
//
//    // 同步设备，确保kernel执行完毕并且结果已写入d_cost_sum_sq_error
//    cudaDeviceSynchronize();
//    // 检查同步过程中或kernel执行期间是否有错误
//    NNException::throwIfDeviceErrorsOccurred("Cannot compute sum of squared errors (post-kernel sync).");
//
//    float h_sum_squared_error = *d_cost_sum_sq_error; // 从GPU（或managed memory）读取总和
//
//    cudaFree(d_cost_sum_sq_error); // 释放分配的GPU内存
//
//    // 计算均方误差 (MSE)
//    return h_sum_squared_error / static_cast<float>(N);
//}


float MSECost::cost(
    Matrix predictions, Matrix target
    /*const float* y_pred_device,
    const float* y_true_device,
    int batch_size,
    int num_features*/
) {


    int size = predictions.shape.x;
    if (size == 0) {
        return 0.0f;
    }

    // Kernel launch configuration
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    if (num_blocks == 0 && size > 0) num_blocks = 1;


    // Allocate device memory for per-block partial sums
    float* d_block_partial_sums = nullptr;
    cudaMalloc(&d_block_partial_sums, num_blocks * sizeof(float));

    size_t shared_mem_size = threads_per_block * sizeof(float);

    // Launch kernel to get sum of squared errors per block
    meanSquareErrorCostKernel << <num_blocks, threads_per_block, shared_mem_size >> > (
        predictions.data_device.get(),
        target.data_device.get(),
        size,
        d_block_partial_sums
        );

    cudaDeviceSynchronize(); // Ensure kernel completion before copying sums

    // Copy per-block partial sums from device to host
    float* h_block_partial_sums = new float[num_blocks];
    cudaMemcpy(h_block_partial_sums, d_block_partial_sums,
        num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum the partial sums on the host
    double total_sum_of_squared_errors = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        total_sum_of_squared_errors += h_block_partial_sums[i];
    }

    // Final MSE cost
    float final_mse_cost = static_cast<float>(total_sum_of_squared_errors / static_cast<double>(size));

    // Clean up
    delete[] h_block_partial_sums;
    cudaFree(d_block_partial_sums);

    return final_mse_cost;
}




//Matrix MSECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
//	assert(predictions.shape.x == target.shape.x);
//
//	dim3 block_size(256);
//	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
//	dMeanSquareErrorCost << <num_of_blocks, block_size >> > (predictions.data_device.get(),
//		target.data_device.get(),
//		dY.data_device.get(),
//		predictions.shape.x);
//	NNException::throwIfDeviceErrorsOccurred("Cannot compute derivative for binary cross entropy.");
//
//	return dY;
//}

Matrix MSECost::dCost(
    Matrix predictions, Matrix target, Matrix dY
    //const float* y_pred_device,
    //const float* y_true_device,
    //float* d_cost_d_pred_device, // Output: derivative
    //int batch_size,
    //int num_features
) {

    int size = predictions.shape.x;

    // Kernel launch configuration
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    dMeanSquareErrorCostKernel << <num_blocks, threads_per_block >> > (
        predictions.data_device.get(),
        target.data_device.get(),
        dY.data_device.get(),
        size
        );
    
    return dY;
}

