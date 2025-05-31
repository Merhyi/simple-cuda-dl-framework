#pragma once

#include "../nn_utils/matrix.hh" // Or your actual matrix library path
#include "../nn_utils/shape.hh"  // Or your actual shape library path

#include <vector>
#include <string> // For saveBatchesToCSV (optional)

class LinearRegressionDataset {
private:
    size_t batch_size_param_;
    size_t num_batches_param_;
    float noise_param_;       // Noise level for target values
    float true_weight_;       // The 'w' in y = w*x + b
    float true_bias_;         // The 'b' in y = w*x + b
    size_t input_features_;   // Number of input features (e.g., 1 for y=wx+b, 2 for y=w1x1+w2x2+b)


    // Match naming convention from CoordinatesDataset.cu
    std::vector<Matrix> batches_; // Feature batches (x values)
    std::vector<Matrix> targets_; // Target batches (y values)

public:
    // Constructor
    // batch_size: The number of samples per batch
    // num_batches: The total number of batches to generate
    // input_features: Number of input features for X. If 1, y = w*x_val + b. If >1, more complex.
    // noise: Standard deviation for Gaussian noise added to target y
    // true_w: The 'true' slope (if input_features=1) or a representative weight.
    // true_b: The 'true' intercept.
    LinearRegressionDataset(size_t batch_size, size_t num_batches,
        size_t input_features = 1, // Default to 1D input feature
        float noise = 0.1f,
        float true_w = 2.0f, float true_b = 0.5f);

    // Getters
    size_t getNumOfBatches() const;
    const std::vector<Matrix>& getBatches() const; // Returns X (features)
    const std::vector<Matrix>& getTargets() const; // Returns y (targets)
    size_t getBatchSize() const;

    // Optional: Utility to save generated batches to CSV files
    void saveBatchesToCSV(const std::string& base_filename) const;
};