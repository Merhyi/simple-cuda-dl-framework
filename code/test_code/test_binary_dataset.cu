#include "./datasets/binary_dataset.hh" // Adjust path as needed
#include "./nn_utils/matrix.hh"  // For Matrix type info if needed directly (not really here)
#include <iostream>
#include <algorithm> // For std::min

void print_first_n_samples(const Matrix& features, const Matrix& targets, size_t n, size_t batch_idx) {
    std::cout << "\n--- Batch " << batch_idx << " (first " << n << " samples) ---" << std::endl;
    // Create temporary copies to ensure host data is up-to-date for printing
    // This depends on how your Matrix class handles device/host synchronization
    Matrix features_host(features.shape);
    features_host.allocateMemory(); // Ensure memory is allocated
    std::memcpy(features_host.data_host.get(), features.data_host.get(), features.shape.x * features.shape.y * sizeof(float)); // Assuming direct host access is fine after copyDeviceToHost in dataset
    // OR, if features is device-only after dataset creation:
    // features_host = features; // if Matrix has copy constructor
    // features_host.copyDeviceToHost();

    Matrix targets_host(targets.shape);
    targets_host.allocateMemory();
    std::memcpy(targets_host.data_host.get(), targets.data_host.get(), targets.shape.x * targets.shape.y * sizeof(float));
    // targets_host = targets;
    // targets_host.copyDeviceToHost();


    size_t num_to_print = std::min(n, features_host.shape.y);
    for (size_t r = 0; r < num_to_print; ++r) {
        std::cout << "Sample " << r
            << ": x1=" << features_host[r]                           // Col 0, Row r
            << ", x2=" << features_host[features_host.shape.y + r]  // Col 1, Row r
                << ", label=" << targets_host[r]                         // Col 0, Row r
                << std::endl;
    }
}


int main() {
    size_t num_batches = 5;
    size_t samples_per_batch = 50;
    float noise = 0.1f;

    std::cout << "Generating Linearly Separable SimpleBinaryDataset..." << std::endl;
    SimpleBinaryDataset linear_dataset(samples_per_batch, num_batches, noise, true);
    std::cout << "Linear dataset generation complete." << std::endl;
    std::cout << "Number of Batches: " << linear_dataset.getNumOfBatches() << std::endl;

    if (linear_dataset.getNumOfBatches() > 0) {
        print_first_n_samples(linear_dataset.getBatches()[0], linear_dataset.getTargets()[0], 5, 0);
    }
    linear_dataset.saveBatchesToCSV("simple_linear_data");


    std::cout << "\nGenerating Non-Linearly Separable SimpleBinaryDataset..." << std::endl;
    SimpleBinaryDataset nonlinear_dataset(samples_per_batch, num_batches, noise, false);
    std::cout << "Non-linear dataset generation complete." << std::endl;
    std::cout << "Number of Batches: " << nonlinear_dataset.getNumOfBatches() << std::endl;

    if (nonlinear_dataset.getNumOfBatches() > 0) {
        print_first_n_samples(nonlinear_dataset.getBatches()[0], nonlinear_dataset.getTargets()[0], 5, 0);
    }
    nonlinear_dataset.saveBatchesToCSV("simple_nonlinear_data");


    return 0;
}