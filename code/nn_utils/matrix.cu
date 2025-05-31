#include "matrix.hh"
#include "nn_exception.hh"
#include <time.h> // For time() to seed the random number generator by default
#include <curand_kernel.h> // For cuRAND device functions
#include <vector>

Matrix::Matrix(size_t x_dim, size_t y_dim) :
	shape(x_dim, y_dim), data_device(nullptr), data_host(nullptr),
	device_allocated(false), host_allocated(false)
{
}

Matrix::Matrix(Shape shape) :
	Matrix(shape.x, shape.y)
{ }

void Matrix::allocateCudaMemory() {
	if (!device_allocated) {
		float* device_memory = nullptr;
		cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
		NNException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
		data_device = std::shared_ptr<float>(device_memory,
											 [&](float* ptr){ cudaFree(ptr); });
		device_allocated = true;
	}
}

void Matrix::allocateHostMemory() {
	if (!host_allocated) {
		data_host = std::shared_ptr<float>(new float[shape.x * shape.y],
										   [&](float* ptr){ delete[] ptr; });
		host_allocated = true;
	}
}

void Matrix::allocateMemory() {
	allocateCudaMemory();
	allocateHostMemory();
}

void Matrix::allocateMemoryIfNotAllocated(Shape shape) {
	if (!device_allocated && !host_allocated) {
		this->shape = shape;
		allocateMemory();
	}
}

void Matrix::copyHostToDevice() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_device.get(), data_host.get(), shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
	}
	else {
		throw NNException("Cannot copy host data to not allocated memory on device.");
	}
}

void Matrix::copyDeviceToHost() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_host.get(), data_device.get(), shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy device data to host.");
	}
	else {
		throw NNException("Cannot copy device data to not allocated memory on host.");
	}
}

float& Matrix::operator[](const int index) {
	return data_host.get()[index];
}

const float& Matrix::operator[](const int index) const {
	return data_host.get()[index];
}

int Matrix::paramCount()
{
	return shape.x * shape.y;
}




void Matrix::fill_zeros_on_device() {


	size_t num_elements = static_cast<size_t>(shape.x) * shape.y;
	if (num_elements == 0) {
		return; // Nothing to zero if no elements
	}
	size_t num_bytes = num_elements * sizeof(float);
	cudaMemset(data_device.get(), 0, num_bytes);
}


__global__ void fill_ones_kernel(float* data, size_t num_elements) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_elements) {
		data[idx] = 1.0f;
	}
}

void Matrix::fill_ones_on_device() {
	size_t num_elements = static_cast<size_t>(shape.x) * static_cast<size_t>(shape.y);

	if (num_elements == 0) {
		return;
	}

	int threads_per_block = 256;
	int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

	fill_ones_kernel << <blocks_per_grid, threads_per_block >> > (data_device.get(), num_elements);

}




__global__ void random_init_kernel(float* data, size_t num_elements, unsigned long long seed, float min_val, float max_val) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_elements) {
		curandState_t state;
		curand_init(seed, idx, 0, &state);

		// Generate a uniform random float in [0.0, 1.0)
		float r = curand_uniform(&state);

		// Scale and shift to the desired range [min_val, max_val)
		data[idx] = min_val + r * (max_val - min_val);
	}
}



void Matrix::random_init_on_device(float min_val, float max_val, unsigned long long seed) {
	// Calculate the total number of elements in the matrix
	size_t num_elements = static_cast<size_t>(shape.x) * static_cast<size_t>(shape.y);

	// If no elements, nothing to do
	if (num_elements == 0) {
		return;
	}

	if (seed == 0) {
		seed = static_cast<unsigned long long>(time(0));
	}

	// Define CUDA kernel launch configuration
	int threads_per_block = 256; // Common choice
	int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

	// Launch the CUDA kernel
	// data_device.get() should return a float* to the device memory
	random_init_kernel << <blocks_per_grid, threads_per_block >> > (
		data_device.get(),
		num_elements,
		seed,
		min_val,
		max_val
		);
}


void Matrix::fillHostData(const std::vector<float>& vec_data) {

	std::copy(vec_data.begin(), vec_data.end(), data_host.get());
}