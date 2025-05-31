#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>
#include <string>

#include "linear_layer.hh"
#include "../nn_utils/nn_exception.hh"





#define TILE_DIM 12




void LinearLayer::zero_gradients() {
	// Use CUDA memset or a simple kernel to fill dW and db device memory with zeros
	cudaMemset(dW.data_device.get(), 0, dW.shape.x * dW.shape.y * sizeof(float));
	NNException::throwIfDeviceErrorsOccurred("Failed to zero dW");
	cudaMemset(db.data_device.get(), 0, db.shape.x * db.shape.y * sizeof(float));
	NNException::throwIfDeviceErrorsOccurred("Failed to zero db");
	
}


void LinearLayer::initializeWeightsRandomly() {
	std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution(0.0, 1.0);

	for (int x = 0; x < W.shape.x; x++) {
		for (int y = 0; y < W.shape.y; y++) {
			W[y * W.shape.x + x] = normal_distribution(generator) * weights_init_threshold;
		}
	}

	W.copyHostToDevice();
}

void LinearLayer::initializeBiasWithZeros() {
	for (int x = 0; x < b.shape.x; x++) {
		b[x] = 0;
	}

	b.copyHostToDevice();
}


LinearLayer::LinearLayer(std::string name, Shape W_shape) :
	W(W_shape), b(W_shape.y, 1), dW(W_shape), db(W_shape.y, 1)
{
	this->name = name;
	b.allocateMemory();
	W.allocateMemory();
	initializeBiasWithZeros();
	initializeWeightsRandomly();
	dW.allocateMemory();
	db.allocateMemory();
	zero_gradients();

}

LinearLayer::~LinearLayer()
{
}


// 你的 printMatrix 函数，它接收一个 Matrix 对象引用
static void printMatrixCUDA(const Matrix& mat_to_print) {
	// ... rest of the logic using mat_to_print.data_device.get() and cudaMemcpy ...
	int count = mat_to_print.shape.x * mat_to_print.shape.y;

	std::vector<float> h_data(count);
	cudaError_t err = cudaMemcpy(h_data.data(), mat_to_print.data_device.get(), count * sizeof(float), cudaMemcpyDeviceToHost);
	for (auto data : h_data)
	{
		std::cout << data << "\t";
	}
}


// In LinearLayer.cpp
void LinearLayer::get_learnable_parameters(ParameterGroup& group) {
	// Add weight parameter and its gradient
	group.add_parameter(&W, &dW);

	//std::cout << "Optimizer: Address of parameter_matrix W in Linear Layer: "
	//	<< static_cast<void*>(W.data_device.get()) << std::endl;
	std::cout << "Optimizer: Address of parameter_matrix dW in Linear Layer: "
		<< static_cast<void*>(dW.data_device.get()) << std::endl;
	
	// Add bias parameter and its gradient
	group.add_parameter(&b, &db);
	
}


// Kernel to compute dW (gradient of weights)
__global__ void linearLayerComputeWeightGradients(
	const float* dZ,    // Upstream gradient (dLoss/dZ)
	const float* A,     // Activation from forward pass (input to this layer)
	float* dW,          // Output: Gradient for weights (dLoss/dW)
	int dZ_x_dim, int dZ_y_dim, // dZ is batch_size x out_features
	int A_x_dim, int A_y_dim,   // A is batch_size x in_features
	int W_rows, int W_cols)     // W is in_features x out_features
	// dW has same shape as W
{
	int col = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to W_cols (out_features)
	int row = blockIdx.y * blockDim.y + threadIdx.y; // Corresponds to W_rows (in_features)

	// dW_ij = sum_k (A_ki * dZ_kj) / batch_size --- if A is (batch, in), dZ is (batch, out)
	// Here, A is (A_x_dim = batch_size, A_y_dim = in_features)
	// dZ is (dZ_x_dim = batch_size, dZ_y_dim = out_features)
	// W (and dW) is (W_rows = A_y_dim = in_features, W_cols = dZ_y_dim = out_features)

	//float dW_value = 0.0f;

	//if (row < W_rows && col < W_cols) {
	//	
	//	// A is treated as A_transpose for matrix mult: A^T * dZ
	//	// A_ik (A[k * A_y_dim + row] if A was in_features x batch_size)
	//	// dZ_kj (dZ[k * dZ_y_dim + col] if dZ was batch_size x out_features)
	//	// Iterate over batch_size (dZ_x_dim or A_x_dim)
	//	for (int k = 0; k < dZ_x_dim; k++) { // k is batch sample index
	//		// A[k * A_y_dim + row] is A_transpose[row][k] (value of k-th sample for in_feature `row`)
	//		// dZ[k * dZ_y_dim + col] is dZ[k][col] (value of k-th sample for out_feature `col`)
	//		dW_value += A[k * A_y_dim + row] * dZ[k * dZ_y_dim + col];
	//	}
	//	// Average over batch size
	//	dW[row * W_cols + col] = dW_value / static_cast<float>(dZ_x_dim);
	//}

	int W_x_dim = A_y_dim;
	int W_y_dim = dZ_y_dim;

	float dW_value = 0.0f;

	if (row < W_y_dim && col < W_x_dim) {
		for (int i = 0; i < dZ_x_dim; i++) {
			dW_value += dZ[row * dZ_x_dim + i] * A[col * A_x_dim + i];
		}
		dW[row * W_x_dim + col] = (dW_value / A_x_dim);
	}
}











// Kernel to compute db (gradient of biases)
__global__ void linearLayerComputeBiasGradients(
	const float* dZ,    // Upstream gradient (dLoss/dZ)
	float* db,          // Output: Gradient for biases (dLoss/db)
	int dZ_x_dim, int dZ_y_dim, // dZ is batch_size x out_features
	int b_size)         // b is out_features x 1, db has same shape
{
	// Each thread computes sum for one bias element.
	// b_size should be dZ_y_dim (number of output features)
	//int bias_idx = blockIdx.x * blockDim.x + threadIdx.x; // Index for the bias/output feature
	//float db_value = 0.0f;
	//if (bias_idx < b_size) {
	//	
	//	// Sum dZ over the batch dimension for this output feature
	//	for (int i = 0; i < dZ_x_dim; i++) { // i is batch sample index
	//		db_value += dZ[i * dZ_y_dim + bias_idx];
	//	}
	//	// Average over batch size
	//	// Using atomicAdd because multiple blocks might try to sum into db if not careful,
	//	// but if each thread handles one bias_idx uniquely, direct assignment is fine.
	//	// For simplicity assuming 1D grid for biases here.
	//	db[bias_idx] = db_value / static_cast<float>(dZ_x_dim);
	//}

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dZ_x_dim * dZ_y_dim) {
		int dZ_x = index % dZ_x_dim;
		int dZ_y = index / dZ_x_dim;
		atomicAdd(&db[dZ_y], dZ[dZ_y * dZ_x_dim + dZ_x] / dZ_x_dim);
	}
}




__global__ void linearLayerForward( float* W, float* A, float* Z, float* b,
									int W_x_dim, int W_y_dim,
									int A_x_dim, int A_y_dim) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int Z_x_dim = A_x_dim;
	int Z_y_dim = W_y_dim;

	float Z_value = 0;

	if (row < Z_y_dim && col < Z_x_dim) {
		for (int i = 0; i < W_x_dim; i++) {
			Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
		}
		Z[row * Z_x_dim + col] = Z_value + b[row];
	}
}






__global__ void linearLayerBackprop(float* W, float* dZ, float *dA,
									int W_x_dim, int W_y_dim,
									int dZ_x_dim, int dZ_y_dim) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// W is treated as transposed
	int dA_x_dim = dZ_x_dim;
	int dA_y_dim = W_x_dim;

	float dA_value = 0.0f;

	if (row < dA_y_dim && col < dA_x_dim) {
		for (int i = 0; i < W_y_dim; i++) {
			dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
		}
		dA[row * dA_x_dim + col] = dA_value;
	}
}







// Wrapper for the kernel to compute dW
void LinearLayer::computeWeightGradients(Matrix& dZ_upstream) {
	dim3 block_size(8, 8); // Or other appropriate size
	// dW has shape W.shape (in_features x out_features)
	// A.shape.y is in_features, dZ_upstream.shape.y is out_features
	dim3 num_of_blocks((W.shape.x + block_size.x - 1) / block_size.x,
		(W.shape.y + block_size.y - 1) / block_size.y);

	linearLayerComputeWeightGradients << <num_of_blocks, block_size >> > (
		dZ_upstream.data_device.get(),
		A.data_device.get(),      // A is stored from forward pass
		dW.data_device.get(),     // Output dW
		dZ_upstream.shape.x, dZ_upstream.shape.y, // batch_size, out_features
		A.shape.x, A.shape.y,                     // batch_size, in_features
		W.shape.y, W.shape.x                      // W_rows (in_features), W_cols (out_features)
		// My W is (cols, rows) so W.shape.x = cols, W.shape.y = rows
		// If W is (rows=in_features, cols=out_features):
		// W_rows = W.shape.y, W_cols = W.shape.x
		);
}

// Wrapper for the kernel to compute db
void LinearLayer::computeBiasGradients(Matrix& dZ_upstream) {
	// b.shape.x is the number of bias elements (== out_features)
	dim3 block_size(256);
	dim3 num_of_blocks((b.shape.x + block_size.x - 1) / block_size.x);

	linearLayerComputeBiasGradients << <num_of_blocks, block_size >> > (
		dZ_upstream.data_device.get(),
		db.data_device.get(),      // Output db
		dZ_upstream.shape.x, dZ_upstream.shape.y, // batch_size, out_features
		b.shape.x                  // Number of biases (out_features)
		);
}








// this backprop only calculates and stores the gradients.
// Renamed from backprop, learning_rate removed
Matrix& LinearLayer::backprop(Matrix& dZ) {
	// 1. Compute dLoss/dA (gradient to pass to the layer below)
	// This needs A.shape for dA allocation

	zero_gradients();

	/*std::cout << std::endl << "Before: " << std::endl;
	std::cout << "dW: " << std::endl;
	printMatrixCUDA(dW);
	std::cout << "db: " << std::endl;
	printMatrixCUDA(db);*/

	dA.allocateMemoryIfNotAllocated(A.shape);
	computeAndStoreBackpropError(dZ); // Uses W and dZ_upstream to calculate dA
	NNException::throwIfDeviceErrorsOccurred("Cannot compute backprop error (dA).");

	// 2. Compute dLoss/dW (gradient for weights)
	// dW is already allocated with W.shape
	computeWeightGradients(dZ);
	NNException::throwIfDeviceErrorsOccurred("Cannot compute weight gradients (dW).");

	// 3. Compute dLoss/db (gradient for biases)
	// db is already allocated with b.shape
	computeBiasGradients(dZ);
	NNException::throwIfDeviceErrorsOccurred("Cannot compute bias gradients (db).");

	/*std::cout << std::endl <<  "After: " << std::endl;
	std::cout << "dW: " <<  std::endl;
	printMatrixCUDA(dW);
	std::cout << "db: " << std::endl;
	printMatrixCUDA(db);*/
	
	return dA; // Return dLoss/dA for the previous layer
}


Matrix& LinearLayer::forward(Matrix& A) {
	assert(W.shape.x == A.shape.y);

	this->A = A;
	Shape Z_shape(A.shape.x, W.shape.y);
	Z.allocateMemoryIfNotAllocated(Z_shape);

	computeAndStoreLayerOutput(A);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform linear layer forward propagation.");

	return Z;
}







void LinearLayer::computeAndStoreLayerOutput(Matrix& A) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(Z.shape.x + block_size.x - 1) / block_size.x,
						(Z.shape.y + block_size.y - 1) / block_size.y);
	linearLayerForward<<<num_of_blocks, block_size>>>( W.data_device.get(),
													   A.data_device.get(),
													   Z.data_device.get(),
													   b.data_device.get(),
													   W.shape.x, W.shape.y,
													   A.shape.x, A.shape.y);
}


void LinearLayer::computeAndStoreBackpropError(Matrix& dZ) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(A.shape.x + block_size.x - 1) / block_size.x,
						(A.shape.y + block_size.y - 1) / block_size.y);
	linearLayerBackprop<<<num_of_blocks, block_size>>>( W.data_device.get(),
														dZ.data_device.get(),
														dA.data_device.get(),
														W.shape.x, W.shape.y,
														dZ.shape.x, dZ.shape.y);
}



int LinearLayer::getXDim() const {
	return W.shape.x;
}

int LinearLayer::getYDim() const {
	return W.shape.y;
}

Matrix LinearLayer::getWeightsMatrix() const {
	return W;
}

Matrix LinearLayer::getBiasVector() const {
	return b;
}
