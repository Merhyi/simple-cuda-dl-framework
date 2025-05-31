#include ".\datasets\planar_dataset.hh" // Adjust path as needed
#include <iostream>
#include <algorithm> // For std::min

int main() {
    size_t num_batches = 1;
    size_t samples_per_batch = 400;
    float noise_level = 0.15f;

    std::cout << "Generating PlanarDataset..." << std::endl;
    PlanarDataset dataset(samples_per_batch, num_batches, noise_level);
    std::cout << "Dataset generation complete." << std::endl;

    std::cout << "\nDataset Details:" << std::endl;
    std::cout << "Number of Batches: " << dataset.getNumOfBatches() << std::endl;
    std::cout << "Batch Size (nominal): " << samples_per_batch << std::endl;

    const auto& feature_batches = dataset.getBatches();
    const auto& target_batches = dataset.getTargets();

    if (!feature_batches.empty()) {
        std::cout << "\n--- First Feature Batch (first 5 samples from host data) ---" << std::endl;
        const Matrix& first_fb = feature_batches[0];
        // Assuming first_fb.shape.x_dim is the number of rows in this batch
        for (size_t r = 0; r < std::min((size_t)5, first_fb.shape.x); ++r) {
            // Accessing using column-major logic
            std::cout << "Sample " << r
                << ": x=" << first_fb[r]  // Col 0, Row r
                << ", y=" << first_fb[first_fb.shape.x + r] // Col 1, Row r
                    << std::endl;
        }

        std::cout << "\n--- First Target Batch (first 5 samples from host data) ---" << std::endl;
        const Matrix& first_tb = target_batches[0];
        for (size_t r = 0; r < std::min((size_t)5, first_tb.shape.x); ++r) {
            std::cout << "Sample " << r << ": label=" << first_tb[r] << std::endl;
        }
    }

    std::cout << "\nSaving batches to CSV..." << std::endl;
    dataset.saveBatchesToCSV("planar_data_from_new_matrix");
    std::cout << "CSV saving complete." << std::endl;

    return 0;
}