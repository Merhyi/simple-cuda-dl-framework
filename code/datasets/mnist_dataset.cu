// mnist_dataset.cu
#include "mnist_dataset.hh"
#include <fstream>
#include <iostream>
#include <stdexcept> // For std::runtime_error
#include <algorithm> // For std::min

// Helper to swap endianness (MNIST files are big-endian)
uint32_t MNISTDataset::swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

// Generic MNIST IDX file loader
std::vector<unsigned char> MNISTDataset::load_mnist_file(const std::string& path, uint32_t expected_magic, int& num_items, std::vector<int>& dims) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    uint32_t magic_number = 0;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = swap_endian(magic_number);

    if (magic_number != expected_magic) {
        throw std::runtime_error("Invalid magic number in file: " + path +
            " (got " + std::to_string(magic_number) +
            ", expected " + std::to_string(expected_magic) + ")");
    }

    uint32_t num_items_u32 = 0;
    file.read(reinterpret_cast<char*>(&num_items_u32), sizeof(num_items_u32));
    num_items = static_cast<int>(swap_endian(num_items_u32));

    dims.clear();
    // For images, there are 2 more dimensions (rows, cols)
    // For labels, this part is skipped (magic number indicates type)
    if (expected_magic == 0x00000803) { // Image file magic number
        for (int i = 0; i < 2; ++i) {
            uint32_t dim_size = 0;
            file.read(reinterpret_cast<char*>(&dim_size), sizeof(dim_size));
            dims.push_back(static_cast<int>(swap_endian(dim_size)));
        }
    }
    else if (expected_magic == 0x00000801) { // Label file magic number
        // Labels are 1D, no additional dimension headers beyond num_items
    }
    else {
        throw std::runtime_error("Unknown expected magic number for dimension reading.");
    }


    size_t data_size = num_items;
    for (int dim : dims) {
        data_size *= dim;
    }

    std::vector<unsigned char> data_buffer(data_size);
    file.read(reinterpret_cast<char*>(data_buffer.data()), data_size);

    if (file.gcount() != static_cast<std::streamsize>(data_size)) {
        throw std::runtime_error("Failed to read the full data from file: " + path);
    }

    return data_buffer;
}


MNISTDataset::MNISTDataset(size_t batch_size_param,
    const std::string& image_file_path,
    const std::string& label_file_path,
    size_t samples_to_load,
    bool normalize,
    bool one_hot_encode) :
    batch_size(batch_size_param)
{
    std::cout << "Loading MNIST dataset..." << std::endl;

    int num_images, num_labels;
    std::vector<int> image_dims; // Should be [rows, cols], e.g., [28, 28]
    std::vector<int> label_dims; // Should be empty for label file

    std::vector<unsigned char> image_data_raw = load_mnist_file(image_file_path, 0x00000803, num_images, image_dims);
    std::vector<unsigned char> label_data_raw = load_mnist_file(label_file_path, 0x00000801, num_labels, label_dims);

    if (num_images != num_labels) {
        throw std::runtime_error("Number of images and labels do not match!");
    }
    if (image_dims.size() != 2) {
        throw std::runtime_error("Image data should have 2 dimensions (rows, cols).");
    }
    const int image_rows = image_dims[0];
    const int image_cols = image_dims[1];
    const int image_features = image_rows * image_cols; // e.g., 28 * 28 = 784

    std::cout << "Successfully loaded " << num_images << " images ("
        << image_rows << "x" << image_cols << ") and " << num_labels << " labels." << std::endl;

    num_actual_samples = std::min(static_cast<size_t>(num_images), samples_to_load);
    number_of_batches = num_actual_samples / batch_size; // Integer division, drops last incomplete batch

    if (number_of_batches == 0 && num_actual_samples > 0) {
        std::cerr << "Warning: Not enough samples (" << num_actual_samples
            << ") for even one batch of size " << batch_size
            << ". Dataset will be empty." << std::endl;
    }
    else {
        std::cout << "Using " << num_actual_samples << " samples, forming "
            << number_of_batches << " batches of size " << batch_size << "." << std::endl;
    }


    for (size_t i = 0; i < number_of_batches; ++i) {
        // For MNIST, images are typically flattened: batch_size x (28*28)
        batches.push_back(Matrix(Shape(batch_size, image_features)));

        // Targets: batch_size x 1 (for integer labels) or batch_size x 10 (for one-hot)
        size_t target_features = one_hot_encode ? 10 : 1;
        targets.push_back(Matrix(Shape(batch_size, target_features)));

        batches.back().allocateMemory();
        targets.back().allocateMemory();

        for (size_t k = 0; k < batch_size; ++k) { // k is sample index within the batch
            size_t global_sample_idx = i * batch_size + k;

            // Fill image data (assuming Matrix uses column-major or has an intelligent operator[])
            // Based on CoordinatesDataset, Matrix is column-major like:
            // M(row, col) is at data[col * num_rows + row]
            // Here, row is sample index 'k', col is pixel feature index 'p'
            for (int p = 0; p < image_features; ++p) { // p is pixel feature index
                float pixel_value = static_cast<float>(image_data_raw[global_sample_idx * image_features + p]);
                if (normalize) {
                    pixel_value /= 255.0f;
                    // Optionally, center around zero: pixel_value -= 0.5f;
                }
                // batches.back() is Matrix(Shape(batch_size, image_features))
                // To set element (k, p) [sample k, pixel p]:
                batches.back()[p * batch_size + k] = pixel_value;
            }

            // Fill target data
            unsigned char label_val = label_data_raw[global_sample_idx];
            if (one_hot_encode) {
                // targets.back() is Matrix(Shape(batch_size, 10))
                // To set element (k, c) [sample k, class c]:
                // targets.back()[c * batch_size + k]
                for (size_t c = 0; c < 10; ++c) {
                    targets.back()[c * batch_size + k] = (c == label_val) ? 1.0f : 0.0f;
                }
            }
            else {
                // targets.back() is Matrix(Shape(batch_size, 1))
                // To set element (k, 0) [sample k, label value]:
                // targets.back()[0 * batch_size + k] which is targets.back()[k]
                targets.back()[k] = static_cast<float>(label_val);
            }
        }

        batches.back().copyHostToDevice();
        targets.back().copyHostToDevice();
    }
    std::cout << "MNIST dataset processing complete. " << batches.size() << " batches created." << std::endl;
}

int MNISTDataset::getNumOfBatches() {
    return static_cast<int>(number_of_batches);
}

std::vector<Matrix>& MNISTDataset::getBatches() {
    return batches;
}

std::vector<Matrix>& MNISTDataset::getTargets() {
    return targets;
}

int MNISTDataset::getNumActualSamples() {
    return num_actual_samples;
}