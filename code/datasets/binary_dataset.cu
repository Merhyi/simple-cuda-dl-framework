#include "binary_dataset.hh" // Assuming SimpleBinaryDataset.hh is in the same directory

#include <random>    // For std::random_device, std::mt19937, std::normal_distribution
#include <cmath>     // For std::sqrt, std::abs
#include <fstream>   // For saveBatchesToCSV
#include <iomanip>   // For std::fixed, std::setprecision
#include <iostream>  // For error messages or debug output

SimpleBinaryDataset::SimpleBinaryDataset(size_t batch_size, size_t num_batches, float noise, bool linearly_separable)
    : batch_size_param_(batch_size),
    num_batches_param_(num_batches),
    noise_param_(noise),
    linearly_separable_(linearly_separable)
{
    if (batch_size_param_ == 0 || num_batches_param_ == 0) {
        std::cerr << "Error: SimpleBinaryDataset batch_size or num_batches cannot be zero." << std::endl;
        // Consider throwing an exception
        return;
    }

    batches_.reserve(num_batches_param_);
    targets_.reserve(num_batches_param_);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform_dist(-2.0f, 2.0f); // Base distribution for points
    std::normal_distribution<float> noise_dist(0.0f, noise_param_);

    for (size_t i = 0; i < num_batches_param_; ++i) {
        // Create Matrix objects for the current batch
        // Features: batch_size_param_ rows, 2 columns (for x1, x2)
        batches_.emplace_back(Shape(batch_size_param_, 2));
        // Targets: batch_size_param_ rows, 1 column (for label 0 or 1)
        targets_.emplace_back(Shape(batch_size_param_, 1));

        // Allocate memory using the Matrix class's method
        batches_[i].allocateMemory();
        targets_[i].allocateMemory();

        for (size_t k = 0; k < batch_size_param_; ++k) { // k is the row index within the batch
            float x1_base = uniform_dist(gen);
            float x2_base = uniform_dist(gen);
            int label_int;

            if (linearly_separable_) {
                // Simple linear separation: x1 > x2 + offset
                float offset = 0.2f;
                if (x1_base > x2_base + offset) {
                    label_int = 1;
                }
                else if (x1_base < x2_base - offset) { // Add a margin
                    label_int = 0;
                }
                else { // Points near the boundary, assign randomly or to one class
                    label_int = (uniform_dist(gen) > 0) ? 1 : 0; // Less distinct near boundary
                }
            }
            else {
                // Simple non-linear separation (e.g., based on distance from origin or a quadratic boundary)
                // Example: Points inside a circle vs outside, or a XOR-like pattern
                // For this example, let's do a simple XOR like pattern in one quadrant
                // Or based on quadrants:
                if ((x1_base > 0 && x2_base > 0) || (x1_base < -0.5f && x2_base < -0.5f)) {
                    label_int = 1;
                }
                else {
                    label_int = 0;
                }
            }

            float x1_final = x1_base + noise_dist(gen);
            float x2_final = x2_base + noise_dist(gen);

            // Store in the current batch's matrices using 1D operator[]
            // Assuming column-major access or a similar convention for Matrix:
            // For Matrix features (batch_size_param_ rows, 2 columns):
            //   x1_final (column 0, row k) -> index k
            //   x2_final (column 1, row k) -> index batch_size_param_ + k
            // This matches the convention from previous examples if shape.y is rows and shape.x is cols,
            // and LinearLayer expects (output_dim, input_dim) for W, meaning input vector is (input_dim, 1)
            // or a batch of inputs is (batch_size, input_dim).
            // If Matrix has shape.y = rows, shape.x = cols:
            batches_[i][k] = x1_final;                           // Row k, Col 0
            batches_[i][batch_size_param_ + k] = x2_final;       // Row k, Col 1

            // For Matrix targets (batch_size_param_ rows, 1 column):
            //   label (column 0, row k) -> index k
            targets_[i][k] = static_cast<float>(label_int);
        }

        // Copy data from host to device, as per the pattern
        batches_[i].copyHostToDevice();
        targets_[i].copyHostToDevice();
    }
}

// Getters
size_t SimpleBinaryDataset::getNumOfBatches() const {
    return num_batches_param_;
}

const std::vector<Matrix>& SimpleBinaryDataset::getBatches() const {
    return batches_;
}

const std::vector<Matrix>& SimpleBinaryDataset::getTargets() const {
    return targets_;
}

size_t SimpleBinaryDataset::getBatchSize() const {
    return batch_size_param_;
}


// Optional: Utility to save generated batches to CSV
void SimpleBinaryDataset::saveBatchesToCSV(const std::string& base_filename) const {
    for (size_t i = 0; i < batches_.size(); ++i) {
        std::string filename = base_filename + "_batch_" + std::to_string(i) + ".csv";
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            continue;
        }

        outfile << "x1,x2,label\n"; // Column names
        outfile << std::fixed << std::setprecision(6);

        const Matrix& current_features = batches_[i];
        const Matrix& current_targets = targets_[i];

        // Assuming Matrix::operator[] accesses host data for reading,
        // or that a copyDeviceToHost() was called if data is primarily on device.
        // For saving, we need host data.
        // If your Matrix class needs an explicit copy for this, you might do:
        // Matrix temp_features = current_features; temp_features.copyDeviceToHost();
        // Matrix temp_targets = current_targets; temp_targets.copyDeviceToHost();
        // Then use temp_features and temp_targets.
        // For simplicity, assuming operator[] is okay for host access for reading.

        for (size_t r = 0; r < current_features.shape.y; ++r) { // r is row index (sample index)
            // Accessing using column-major logic for 1D operator[]
            float x1 = current_features[r];                                   // Col 0, Row r
            float x2 = current_features[current_features.shape.y + r];        // Col 1, Row r
            float label = current_targets[r];                                 // Col 0, Row r (single column target)
            outfile << x1 << "," << x2 << "," << label << "\n";
        }
        outfile.close();
        std::cout << "Batch " << i << " saved to " << filename << std::endl;
    }
}