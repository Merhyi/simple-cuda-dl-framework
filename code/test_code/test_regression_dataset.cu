#include "datasets/regression_dataset.hh"
#include "./nn_utils/matrix.hh" // For Matrix type
#include <iostream>
#include <algorithm> // For std::min
#include <vector>    // For std::vector in print function

void print_lr_first_n_samples(const Matrix& features, const Matrix& targets, size_t n, size_t batch_idx, size_t num_input_features) {
    std::cout << "\n--- Batch " << batch_idx << " (first " << n << " samples) ---" << std::endl;

    // Make host copies for printing (safer if original might be device-only or get modified)
    Matrix features_host(features.shape);
    if (features.data_host && features.shape.x * features.shape.y > 0) { // Check if host data pointer is valid
        features_host.allocateMemory();
        std::memcpy(features_host.data_host.get(), features.data_host.get(), features.shape.x * features.shape.y * sizeof(float));
    }
    else if (features.shape.x * features.shape.y > 0) {
        std::cerr << "Warning: features.data_host is null or empty for printing batch " << batch_idx << std::endl;
        // Optionally, try copyDeviceToHost if appropriate, but dataset populates host first
    }


    Matrix targets_host(targets.shape);
    if (targets.data_host && targets.shape.x * targets.shape.y > 0) {
        targets_host.allocateMemory();
        std::memcpy(targets_host.data_host.get(), targets.data_host.get(), targets.shape.x * targets.shape.y * sizeof(float));
    }
    else if (targets.shape.x * targets.shape.y > 0) {
        std::cerr << "Warning: targets.data_host is null or empty for printing batch " << batch_idx << std::endl;
    }


    size_t num_to_print = 0;
    if (features_host.data_host && targets_host.data_host) { // Only print if host data is valid
        num_to_print = std::min(n, features_host.shape.y); // features_host.shape.y is num_samples_in_batch
    }


    for (size_t r = 0; r < num_to_print; ++r) {
        std::cout << "Sample " << r << ": ";
        std::vector<float> x_vals;
        for (size_t f = 0; f < num_input_features; ++f) {
            // Accessing column f, row r from features_host
            // Index = f * num_rows_in_batch + r
            float val = features_host[f * features_host.shape.y + r];
            x_vals.push_back(val);
            std::cout << "x" << f + 1 << "=" << val << (f == num_input_features - 1 ? "" : ", ");
        }
        // Accessing row r, col 0 from targets_host
        std::cout << ", y_target=" << targets_host[r] << std::endl;
    }
}


int main() {
    size_t num_batches = 3;
    size_t samples_per_batch = 20;
    float noise_level = 0.2f;

    std::cout << "Generating 1D Linear Regression Dataset..." << std::endl;
    size_t features_1d = 1;
    LinearRegressionDataset dataset_1d(samples_per_batch, num_batches, features_1d, noise_level, 3.0f, -1.5f); // y = 3x - 1.5 + noise
    std::cout << "1D Dataset generation complete. Num Batches: " << dataset_1d.getNumOfBatches() << std::endl;

    if (dataset_1d.getNumOfBatches() > 0) {
        print_lr_first_n_samples(dataset_1d.getBatches()[0], dataset_1d.getTargets()[0], 5, 0, features_1d);
    }
    dataset_1d.saveBatchesToCSV("linear_regression_1d_data");


    std::cout << "\nGenerating 2D Linear Regression Dataset..." << std::endl;
    size_t features_2d = 2;
    // For 2D, true_w_param (2.5f here) will be used as w1, and w2 will be derived (e.g., 2.5f * 0.75f)
    LinearRegressionDataset dataset_2d(samples_per_batch, num_batches, features_2d, noise_level, 2.5f, 1.0f);
    std::cout << "2D Dataset generation complete. Num Batches: " << dataset_2d.getNumOfBatches() << std::endl;

    if (dataset_2d.getNumOfBatches() > 0) {
        print_lr_first_n_samples(dataset_2d.getBatches()[0], dataset_2d.getTargets()[0], 5, 0, features_2d);
    }
    dataset_2d.saveBatchesToCSV("linear_regression_2d_data");

    return 0;
}