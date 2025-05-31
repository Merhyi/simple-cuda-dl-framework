#include "regression_dataset.hh"

#include <random>    // For std::random_device, std::mt19937, std::normal_distribution, std::uniform_real_distribution
#include <cmath>
#include <fstream>   // For saveBatchesToCSV
#include <iomanip>   // For std::fixed, std::setprecision
#include <iostream>  // For error messages or debug output
#include <vector>    // For true_weights_ when input_features > 1

LinearRegressionDataset::LinearRegressionDataset(size_t batch_size, size_t num_batches,
    size_t input_features, float noise,
    float true_w_param, float true_b_param)
    : batch_size_param_(batch_size),
    num_batches_param_(num_batches),
    input_features_(input_features),
    noise_param_(noise),
    true_weight_(true_w_param), // Store for reference, mainly used if input_features_ == 1
    true_bias_(true_b_param)
{
    if (batch_size_param_ == 0 || num_batches_param_ == 0 || input_features_ == 0) {
        std::cerr << "Error: LinearRegressionDataset batch_size, num_batches, or input_features cannot be zero." << std::endl;
        return;
    }

    batches_.reserve(num_batches_param_);
    targets_.reserve(num_batches_param_);

    std::random_device rd;
    std::mt19937 gen(rd());
    // For generating X features, e.g., in range [-5, 5]
    std::uniform_real_distribution<float> feature_dist(-5.0f, 5.0f);
    std::normal_distribution<float> noise_dist(0.0f, noise_param_);

    // For multi-feature input, we need multiple true weights
    std::vector<float> true_weights_vec;
    if (input_features_ > 1) {
        true_weights_vec.reserve(input_features_);
        // Let's generate some example weights if more than one feature
        // e.g., w1=true_w_param, w2=true_w_param/2, w3=true_w_param/4, etc. or random
        float current_w = true_w_param;
        for (size_t f = 0; f < input_features_; ++f) {
            true_weights_vec.push_back(current_w);
            current_w *= 0.75f; // Just an example to vary weights
        }
    }


    for (size_t i = 0; i < num_batches_param_; ++i) {
        // Create Matrix objects for the current batch
        // Features: batch_size_param_ rows, input_features_ columns
        batches_.emplace_back(Shape(batch_size_param_, input_features_));
        // Targets: batch_size_param_ rows, 1 column (for scalar y)
        targets_.emplace_back(Shape(batch_size_param_, 1));

        batches_[i].allocateMemory();
        targets_[i].allocateMemory();

        for (size_t k = 0; k < batch_size_param_; ++k) { // k is the row index (sample index)
            float y_true_noiseless = true_bias_;

            for (size_t f = 0; f < input_features_; ++f) { // f is the feature index (column index)
                float x_val = feature_dist(gen);

                // Store x_val in the features matrix (batches_[i])
                // Using column-major like 1D indexing for Matrix:
                // Index for (row k, col f) = f * num_rows + k
                // num_rows for this matrix is batch_size_param_
                batches_[i][f * batch_size_param_ + k] = x_val;

                if (input_features_ == 1) {
                    y_true_noiseless += true_weight_ * x_val;
                }
                else {
                    y_true_noiseless += true_weights_vec[f] * x_val;
                }
            }

            float y_final = y_true_noiseless + noise_dist(gen);

            // Store y_final in the targets matrix (targets_[i])
            // Targets matrix has only 1 column, so col index is 0
            // Index for (row k, col 0) = 0 * num_rows + k = k
            targets_[i][k] = y_final;
        }

        batches_[i].copyHostToDevice();
        targets_[i].copyHostToDevice();
    }
}

// Getters
size_t LinearRegressionDataset::getNumOfBatches() const {
    return num_batches_param_;
}

const std::vector<Matrix>& LinearRegressionDataset::getBatches() const {
    return batches_;
}

const std::vector<Matrix>& LinearRegressionDataset::getTargets() const {
    return targets_;
}

size_t LinearRegressionDataset::getBatchSize() const {
    return batch_size_param_;
}


void LinearRegressionDataset::saveBatchesToCSV(const std::string& base_filename) const {
    for (size_t i = 0; i < batches_.size(); ++i) {
        std::string filename = base_filename + "_batch_" + std::to_string(i) + ".csv";
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            continue;
        }

        // Write header
        for (size_t f = 0; f < input_features_; ++f) {
            outfile << "x" << f + 1 << ",";
        }
        outfile << "y_target\n";
        outfile << std::fixed << std::setprecision(6);

        const Matrix& current_features = batches_[i];
        const Matrix& current_targets = targets_[i];

        for (size_t r = 0; r < current_features.shape.y; ++r) { // r is row index
            for (size_t f = 0; f < input_features_; ++f) {
                // Accessing using column-major logic for 1D operator[]
                // current_features[col_idx * num_rows + row_idx]
                outfile << current_features[f * current_features.shape.y + r] << ",";
            }
            outfile << current_targets[r] << "\n"; // Targets: col 0, row r
        }
        outfile.close();
        std::cout << "Batch " << i << " saved to " << filename << std::endl;
    }
}