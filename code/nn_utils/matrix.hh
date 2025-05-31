#pragma once

#include "shape.hh"

#include <memory>
#include <vector>

class Matrix {
private:
	bool device_allocated;
	bool host_allocated;

	void allocateCudaMemory();
	void allocateHostMemory();

public:
	Shape shape;

	std::shared_ptr<float> data_device;
	std::shared_ptr<float> data_host;

	Matrix(size_t x_dim = 1, size_t y_dim = 1);
	Matrix(Shape shape);


	int paramCount();

	void allocateMemory();
	void allocateMemoryIfNotAllocated(Shape shape);

	void copyHostToDevice();
	void copyDeviceToHost();
	void fill_zeros_on_device();
	void fill_ones_on_device();
	void random_init_on_device(float min_val = 0.0f, float max_val = 1.0f, unsigned long long seed = 0);

	float& operator[](const int index);
	const float& operator[](const int index) const;
	void fillHostData(const std::vector<float>& vec_data);
};
