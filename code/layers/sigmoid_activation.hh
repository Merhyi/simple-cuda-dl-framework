#pragma once

#include "nn_layer.hh"


class SigmoidActivation : public NNLayer {
private:
	Matrix A;

	Matrix Z;
	Matrix dZ;

public:
	SigmoidActivation(std::string name);
	~SigmoidActivation();

	void get_learnable_parameters(ParameterGroup& group) override;
	Matrix& forward(Matrix& Z);
	Matrix& backprop(Matrix& dA);
};
