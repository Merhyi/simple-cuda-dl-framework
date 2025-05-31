// mnist_dataset.hh
#pragma once

#include "../nn_utils/matrix.hh" // Assuming this path is correct

#include <vector>
#include <string>
#include <cstdint> // For uint32_t, etc.

class MNISTDataset {
private:
    size_t batch_size;
    size_t number_of_batches;
    int num_actual_samples; // Total samples loaded and used

    std::vector<Matrix> batches; // Image data (batch_size x 784)
    std::vector<Matrix> targets; // Label data (batch_size x 1 for integer labels, or batch_size x 10 for one-hot)

    // Helper function to read MNIST IDX files
    uint32_t swap_endian(uint32_t val);
    std::vector<unsigned char> load_mnist_file(const std::string& path, uint32_t expected_magic, int& num_items, std::vector<int>& dims);

public:
    // Constructor now takes paths to MNIST files and optionally number of samples to load
    MNISTDataset(size_t batch_size,
        const std::string& image_file_path,
        const std::string& label_file_path,
        size_t samples_to_load = 60000, // MNIST training set has 60000 images
        bool normalize = true, // Normalize pixels to [0,1]
        bool one_hot_encode = false); // If true, targets are batch_size x 10

    int getNumOfBatches();
    std::vector<Matrix>& getBatches();
    std::vector<Matrix>& getTargets();
    int getNumActualSamples(); // Useful for verification
};