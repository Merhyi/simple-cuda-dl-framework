#pragma once

#include <vector>
#include "layers/nn_layer.hh"
// #include "nn_utils/bce_cost.hh"

#include "nn_utils/cost_function_base.hh"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	CostFunction* cost_func_ptr;

	Matrix Y;
	Matrix dY;

	float learning_rate; // but now discarded


public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();

	Matrix forward(Matrix X);
	void backprop(Matrix predictions, Matrix target);

	void addLayer(NNLayer *layer);
	void setCostFunction(CostFunction* cost_func_ptr);
	std::vector<NNLayer*> getLayers() const;

	ParameterGroup NeuralNetwork::get_all_optimizable_parameters() const;

};
