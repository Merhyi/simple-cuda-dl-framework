#pragma once

#include "../nn_utils/matrix.hh" // Or your actual matrix library path
#include "../nn_utils/shape.hh"  // Or your actual shape library path

#include <vector>
#include <string> // For saveBatchesToCSV (optional)

class SimpleBinaryDataset {
private:
    size_t batch_size_param_;
    size_t num_batches_param_;
    float noise_param_;       // Noise level for data points
    bool linearly_separable_; // Flag to control data separability

    // Match naming convention
    std::vector<Matrix> batches_; // Feature batches (x,y)
    std::vector<Matrix> targets_; // Target batches (labels)

public:
    // Constructor
    // batch_size: The number of samples per batch
    // num_batches: The total number of batches to generate
    // noise: Standard deviation for Gaussian noise added to coordinates
    // linearly_separable: If true, generates linearly separable data, otherwise slightly non-linear
    SimpleBinaryDataset(size_t batch_size, size_t num_batches, float noise, bool linearly_separable = true);

    // Getters
    size_t getNumOfBatches() const;
    const std::vector<Matrix>& getBatches() const;
    const std::vector<Matrix>& getTargets() const;
    size_t getBatchSize() const; // Returns the configured batch size

    // Optional: Utility to save generated batches to CSV files
    void saveBatchesToCSV(const std::string& base_filename) const;
};