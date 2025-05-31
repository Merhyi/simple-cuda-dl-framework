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

    // Ϊ���ڹ�Լ���������ڴ�
    // ���������������� block_size.x = 256 �����˺˺���
    __shared__ float sdata[256];

    int tid_in_block = threadIdx.x; // �����߳�ID
    int index = blockIdx.x * blockDim.x + tid_in_block; // ȫ���߳�ID

    float current_neg_partial_cost = 0.0f; // ��ǰ�̼߳���� -partial_cost

    if (index < size) {
        float pred_val = predictions[index];
        float target_val = target[index];


        //// �ؼ����� prediction ֵ����ǯλ���Ա��� logf(0) �� logf(<=0) ���� NaN/inf
        //// ���Ǳ�֤��ֵ�ȶ��Ե���Ҫ���裬�������Ǵ�����
        //const float epsilon = 1e-9f; // һ����С������
        //pred_val = fmaxf(pred_val, epsilon);           // pred_val >= epsilon
        //pred_val = fminf(pred_val, 1.0f - epsilon);   // pred_val <= 1.0 - epsilon

        // ���� -(target * log(pred) + (1-target) * log(1-pred))
        current_neg_partial_cost = -(target_val * logf(pred_val)
            + (1.0f - target_val) * logf(1.0f - pred_val));
    }

    sdata[tid_in_block] = current_neg_partial_cost;

    __syncthreads(); // ͬ����ȷ�������̶߳��ѽ��Լ���ֵд�빲���ڴ�

    // --- ���ڲ��й�Լ (���) ---
    // ���� blockDim.x ��2���� (����256)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid_in_block < s) {
            sdata[tid_in_block] += sdata[tid_in_block + s];
        }
        __syncthreads(); // ÿ�ε�����ͬ����ȷ���ӷ��������
    }

    // ���ڵĵ�һ���߳� (threadIdx.x == 0) ���ÿ���ܺ�ԭ�ӵؼӵ�ȫ���ۼ�����
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
    // assert(predictions.shape.x == target.shape.x); // �������ǰ�Ѷ��Ի���
    int size = predictions.shape.x;
    if (size == 0) {
        return 0.0f;
    }

    float* d_global_sum_neg_partial_cost; // ���ڴ洢���� -partial_cost ֮�͵��豸�ڴ�ָ��
    // ʹ�� cudaMalloc �����豸�ڴ棬��Ϊ����м�����Ҫ���豸��ʹ��
    cudaMalloc(&d_global_sum_neg_partial_cost, sizeof(float));
    // ʵ�ʴ�����Ӧ��� cudaMalloc �ķ���ֵ

    // ��ʼ���豸�ϵ��ۼ���Ϊ0.0f
    float h_initial_sum = 0.0f;
    cudaMemcpy(d_global_sum_neg_partial_cost, &h_initial_sum, sizeof(float), cudaMemcpyHostToDevice);
    // ʵ�ʴ�����Ӧ��� cudaMemcpy �ķ���ֵ

    dim3 block_size(256); // ��˺����ڹ����ڴ��Сƥ��
    dim3 num_of_blocks((size + block_size.x - 1) / block_size.x);

    // �����Ż���ĺ˺���
    binaryCrossEntropyCostOptimized << <num_of_blocks, block_size >> > (
        predictions.data_device.get(),
        target.data_device.get(),
        size,
        d_global_sum_neg_partial_cost);
    // ʵ�ʴ�����Ӧ���� cudaGetLastError() ���˺�����������

    cudaDeviceSynchronize(); // ȷ������GPU�������
    // ʵ�ʴ�����Ӧ���ͬ������

    float h_total_sum_neg_partial_cost; // ���ڴ��豸�����ܺ͵���������
    cudaMemcpy(&h_total_sum_neg_partial_cost, d_global_sum_neg_partial_cost, sizeof(float), cudaMemcpyDeviceToHost);
    // ʵ�ʴ�����Ӧ��� cudaMemcpy �ķ���ֵ

    cudaFree(d_global_sum_neg_partial_cost); // �ͷ��豸�ڴ�

    // ���ճɱ��� (sum of (-partial_cost)) / size (cost_value)
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
