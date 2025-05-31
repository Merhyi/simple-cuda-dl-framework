#include "neural_network.hh"
#include "nn_utils/nn_exception.hh"
#include "optimizers/optimizer_utils.hh"



NeuralNetwork::NeuralNetwork(float learning_rate) :
	learning_rate(learning_rate)
{ }

NeuralNetwork::~NeuralNetwork() {
	for (auto layer : layers) {
		delete layer;
	}
}

void NeuralNetwork::setCostFunction(CostFunction* cost_func_ptr)
{
	this->cost_func_ptr = cost_func_ptr;
}

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

Matrix NeuralNetwork::forward(Matrix X) {
	Matrix Z = X;

	for (auto layer : layers) {
		Z = layer->forward(Z);
	}

	Y = Z;
	return Y;
}

void NeuralNetwork::backprop(Matrix predictions, Matrix target) {
	dY.allocateMemoryIfNotAllocated(predictions.shape);
	Matrix error = cost_func_ptr->dCost(predictions, target, dY);

	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		error = (*it)->backprop(error);
	}

	cudaDeviceSynchronize();
}

std::vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}

ParameterGroup NeuralNetwork::get_all_optimizable_parameters() const {
	ParameterGroup model_params_group;
	for (auto& layer : layers) {
		layer->get_learnable_parameters(model_params_group);
	}
	return model_params_group;
}