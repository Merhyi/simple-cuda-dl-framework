#include <iostream>
#include <time.h>

#include <cuda_runtime.h>
#include <cuComplex.h> // CUDA complex number types and functions

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "layers/tanh_activation.hh"
#include "layers/leaky_relu_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"
#include "nn_utils/mse_cost.hh"

#include "./datasets/coordinates_dataset.hh"
#include "datasets/regression_dataset.hh"

#include "optimizers/optimizer_utils.hh"
#include "optimizers/sgd_optimizer.hh"
#include "optimizers/adam_optimizer.hh"

#include <iomanip>


float computeAccuracy(const Matrix& predictions, const Matrix& targets);

int main() {

	cudaEvent_t start, stop;
	float milliseconds = 0;
	double elapsed = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	srand( time(NULL) );
	size_t num_batches = 10;
	size_t samples_per_batch = 20;
	float noise_level = 0.2f;

	std::cout << "Generating 2D Linear Regression Dataset..." << std::endl;
	size_t features_2d = 2;
	LinearRegressionDataset dataset(samples_per_batch, num_batches, features_2d, noise_level, 2.5f, 1.0f); 
	std::cout << "2D Dataset generation complete. Num Batches: " << dataset.getNumOfBatches() << std::endl;

	
	MSECost mse_cost;
	const float learning_rate = 0.005f;
	const float momentum = 0.0f;
	const float weight_decay = 0.0f;



	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", Shape(2, 10)));
	nn.addLayer(new TanhActivation("tanh_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(10, 1)));
	nn.setCostFunction(&mse_cost);

	ParameterGroup model_parameters = nn.get_all_optimizable_parameters();
	SGDOptimizer optimizer(learning_rate, momentum, weight_decay);
	// AdamOptimizer optimizer(learning_rate);
	optimizer.set_parameter_group(model_parameters); // Optimizer now knows which params to update

	// network training
	Matrix Y;
	cudaEventRecord(start, 0);
	for (int epoch = 0; epoch < 2001; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
			Y = nn.forward(dataset.getBatches().at(batch));
			nn.backprop(Y, dataset.getTargets().at(batch));
			cost += mse_cost.cost(Y, dataset.getTargets().at(batch));

			
			optimizer.step();
			
		}

		if (epoch % 100 == 0) {
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / dataset.getNumOfBatches()
						<< std::endl;
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Total Training Time: "
		<< std::fixed << std::setprecision(3) << milliseconds << " ms" << std::endl;

	// compute accuracy
	Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
	Y.copyDeviceToHost();

	float test_cost = mse_cost.cost(Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));

	std::cout 	<< "Test Cost: " << test_cost << std::endl;

	return 0;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++) {
		float prediction = predictions[i] > 0.5 ? 1 : 0;
		if (prediction == targets[i]) {
			correct_predictions++;
		}
	}

	return static_cast<float>(correct_predictions) / m;
}
