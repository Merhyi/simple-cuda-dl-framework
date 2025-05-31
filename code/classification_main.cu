#include ".\datasets\binary_dataset.hh" // Adjust path as needed
#include <iostream>
#include <algorithm> // For std::min
#include "neural_network.hh"
#include "layers/linear_layer.hh"

#include "nn_utils/bce_cost.hh"
#include "layers/sigmoid_activation.hh"
#include "layers/tanh_activation.hh"
#include "optimizers/adam_optimizer.hh"
#include "layers/leaky_relu_activation.hh"

#include <cuda_runtime.h>
#include <cuComplex.h> // CUDA complex number types and functions
#include <iomanip>


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



int main()
{
	cudaEvent_t start, stop;
	float milliseconds = 0;
	double elapsed = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	size_t num_batches = 5;
	size_t samples_per_batch = 50;
	float noise = 0.15f;

	const float learning_rate = 0.01f;

	std::cout << "Generating Linearly Separable SimpleBinaryDataset..." << std::endl;
	SimpleBinaryDataset dataset(samples_per_batch, num_batches, noise, false);
	std::cout << "Linear dataset generation complete." << std::endl;
	std::cout << "Number of Batches: " << dataset.getNumOfBatches() << std::endl;




	BCECost bce_cost;
	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", Shape(2, 4)));
	nn.addLayer(new TanhActivation("tanh_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(4, 1)));
	nn.addLayer(new SigmoidActivation("sigmoid_output"));
	nn.setCostFunction(&bce_cost);

	ParameterGroup model_parameters = nn.get_all_optimizable_parameters();
	AdamOptimizer optimizer(learning_rate);
	// AdamOptimizer optimizer(learning_rate);
	optimizer.set_parameter_group(model_parameters); // Optimizer now knows which params to update

	Matrix Y;
	cudaEventRecord(start, 0);
	for (int epoch = 0; epoch < 1001; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
			Y = nn.forward(dataset.getBatches().at(batch));
			nn.backprop(Y, dataset.getTargets().at(batch));
			cost += bce_cost.cost(Y, dataset.getTargets().at(batch));


			optimizer.step();

		}

		if (epoch % 100 == 0) {
			std::cout << "Epoch: " << epoch
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

	float accuracy = computeAccuracy(
		Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
	std::cout << "Accuracy: " << accuracy << std::endl;

	return 0;
}

