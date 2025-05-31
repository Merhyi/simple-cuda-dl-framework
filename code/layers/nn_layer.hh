#pragma once

#include <iostream>
#include <vector>

#include "../nn_utils/matrix.hh"
#include "..\optimizers\optimizer_base.hh"


class NNLayer {
protected:
	std::string name;

public:
	virtual ~NNLayer() = 0;

	virtual Matrix& forward(Matrix& A) = 0;
	virtual Matrix& backprop(Matrix& dZ) = 0;

	std::string getName() { return this->name; };

	// Method to collect learnable parameters from this layer
	virtual void get_learnable_parameters(ParameterGroup& group) = 0;
	

};

inline NNLayer::~NNLayer() {}
