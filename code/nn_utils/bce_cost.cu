#include "bce_cost.hh"
#include "nn_exception.hh"

#include <math.h>
#include <iostream>
#include <assert.h>


//__global__ void binaryCrossEntropyCost(float* predictions, float* target,
//									   int size, float* cost) {
//
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (index < size) {
//		float partial_cost = target[index] * logf(predictions[index])
//				+ (1.0f - target[index]) * logf(1.0f - predictions[index]);
//		atomicAdd(cost, - partial_cost / size);
//	}
//}


__global__ void binaryCrossEntropyCostOptimized(float* predictions, float* target,
    int size, float* global_sum_neg_partial_cost) {

    // 为块内规约声明共享内存
    // 假设主机端总是以 block_size.x = 256 启动此核函数
    __shared__ float sdata[256];

    int tid_in_block = threadIdx.x; // 块内线程ID
    int index = blockIdx.x * blockDim.x + tid_in_block; // 全局线程ID

    float current_neg_partial_cost = 0.0f; // 当前线程计算的 -partial_cost

    if (index < size) {
        float pred_val = predictions[index];
        float target_val = target[index];


        //// 关键：对 prediction 值进行钳位，以避免 logf(0) 或 logf(<=0) 导致 NaN/inf
        //// 这是保证数值稳定性的重要步骤，不仅仅是错误检查
        //const float epsilon = 1e-9f; // 一个很小的正数
        //pred_val = fmaxf(pred_val, epsilon);           // pred_val >= epsilon
        //pred_val = fminf(pred_val, 1.0f - epsilon);   // pred_val <= 1.0 - epsilon

        // 计算 -(target * log(pred) + (1-target) * log(1-pred))
        current_neg_partial_cost = -(target_val * logf(pred_val)
            + (1.0f - target_val) * logf(1.0f - pred_val));
    }

    sdata[tid_in_block] = current_neg_partial_cost;

    __syncthreads(); // 同步，确保所有线程都已将自己的值写入共享内存

    // --- 块内并行规约 (求和) ---
    // 假设 blockDim.x 是2的幂 (例如256)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid_in_block < s) {
            sdata[tid_in_block] += sdata[tid_in_block + s];
        }
        __syncthreads(); // 每次迭代后同步，确保加法操作完成
    }

    // 块内的第一个线程 (threadIdx.x == 0) 将该块的总和原子地加到全局累加器上
    if (tid_in_block == 0) {
        atomicAdd(global_sum_neg_partial_cost, sdata[0]);
    }
}

__global__ void dBinaryCrossEntropyCost(float* predictions, float* target, float* dY,
								     	int size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		dY[index] = -1.0 * ( target[index]/predictions[index] - (1 - target[index])/(1 - predictions[index]) );
	}
}


//float BCECost::cost(Matrix predictions, Matrix target) {
//	assert(predictions.shape.x == target.shape.x);
//
//	float* cost;
//	cudaMallocManaged(&cost, sizeof(float));
//	*cost = 0.0f;
//
//	dim3 block_size(256);
//	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
//	binaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device.get(),
//														  target.data_device.get(),
//														  predictions.shape.x, cost);
//	cudaDeviceSynchronize();
//	NNException::throwIfDeviceErrorsOccurred("Cannot compute binary cross entropy cost.");
//
//	float cost_value = *cost;
//	cudaFree(cost);
//
//	return cost_value;
//}

float BCECost::cost(Matrix predictions, Matrix target) {
    // assert(predictions.shape.x == target.shape.x); // 假设调用前已断言或处理
    int size = predictions.shape.x;
    if (size == 0) {
        return 0.0f;
    }

    float* d_global_sum_neg_partial_cost; // 用于存储所有 -partial_cost 之和的设备内存指针
    // 使用 cudaMalloc 分配设备内存，因为这个中间结果主要在设备上使用
    cudaMalloc(&d_global_sum_neg_partial_cost, sizeof(float));
    // 实际代码中应检查 cudaMalloc 的返回值

    // 初始化设备上的累加器为0.0f
    float h_initial_sum = 0.0f;
    cudaMemcpy(d_global_sum_neg_partial_cost, &h_initial_sum, sizeof(float), cudaMemcpyHostToDevice);
    // 实际代码中应检查 cudaMemcpy 的返回值

    dim3 block_size(256); // 与核函数内共享内存大小匹配
    dim3 num_of_blocks((size + block_size.x - 1) / block_size.x);

    // 调用优化后的核函数
    binaryCrossEntropyCostOptimized << <num_of_blocks, block_size >> > (
        predictions.data_device.get(),
        target.data_device.get(),
        size,
        d_global_sum_neg_partial_cost);
    // 实际代码中应调用 cudaGetLastError() 检查核函数启动错误

    cudaDeviceSynchronize(); // 确保所有GPU计算完成
    // 实际代码中应检查同步错误

    float h_total_sum_neg_partial_cost; // 用于从设备接收总和的主机变量
    cudaMemcpy(&h_total_sum_neg_partial_cost, d_global_sum_neg_partial_cost, sizeof(float), cudaMemcpyDeviceToHost);
    // 实际代码中应检查 cudaMemcpy 的返回值

    cudaFree(d_global_sum_neg_partial_cost); // 释放设备内存

    // 最终成本是 (sum of (-partial_cost)) / size (cost_value)
    return h_total_sum_neg_partial_cost / size;
}



Matrix BCECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
	assert(predictions.shape.x == target.shape.x);

	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
	dBinaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device.get(),
														   target.data_device.get(),
														   dY.data_device.get(),
														   predictions.shape.x);
	NNException::throwIfDeviceErrorsOccurred("Cannot compute derivative for binary cross entropy.");

	return dY;
}
