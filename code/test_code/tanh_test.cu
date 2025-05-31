#include "./layers/tanh_activation.hh"     // Our TanhActivation layer
#include "./nn_utils/matrix.hh"  // Our Matrix class
#include "./nn_utils/nn_exception.hh" // Error handling
#include <iostream>
#include <vector>
#include <cmath>     // For std::tanh on host
#include <iomanip>   // For std::fixed, std::setprecision
#include <numeric>   // For std::iota (optional)
#include <algorithm> // For std::transform

// Helper to compare float vectors with tolerance
bool compareFloatVectors(const std::vector<float>& v1, const std::vector<float>& v2, float tolerance = 1e-6f) {
    if (v1.size() != v2.size()) {
        std::cerr << "Vector sizes differ: " << v1.size() << " vs " << v2.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::fabs(v1[i] - v2[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": " << v1[i] << " vs " << v2[i]
                << " (diff: " << std::fabs(v1[i] - v2[i]) << ")" << std::endl;
                return false;
        }
    }
    return true;
}

void printVector(const std::string& name, const std::vector<float>& vec, int precision = 7) {
    std::cout << name << " (size " << vec.size() << "): [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(precision) << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
}


// Modify your .cu file to use num_elements() from shape
// In tanh_activation.cu:
// Change: int num_elements = Z_input.shape.y * Z_input.shape.x;
// To:     int num_elements = Z_input.shape.num_elements();
// And similarly for A.shape in backprop.

int main() {
    try {
        size_t rows = 2;
        size_t cols = 3;
        Shape test_shape(rows, cols); // Assuming Shape(rows, cols) -> shape.y=rows, shape.x=cols
        size_t num_elements_total = test_shape.x * test_shape.y;

        // --- Test Data ---
        std::vector<float> z_host_data = { -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 3.0f };
        std::vector<float> da_host_data(num_elements_total);
        // Simple dA for testing: 1, 0.1, 0.01, ...
        float da_val = 1.0f;
        for (size_t i = 0; i < num_elements_total; ++i) {
            da_host_data[i] = da_val;
            da_val *= 0.5f;
        }


        // --- Matrices ---
        Matrix Z_input(test_shape);
        Z_input.allocateMemory(); // Explicitly allocate after construction
        Z_input.fillHostData(z_host_data);
        Z_input.copyHostToDevice();

        Matrix dA_input(test_shape);
        dA_input.allocateMemory();
        dA_input.fillHostData(da_host_data);
        dA_input.copyHostToDevice();

        TanhActivation tanh_layer("test_tanh");

        // --- Test Forward Pass ---
        std::cout << "--- Testing Forward Pass ---" << std::endl;
        Matrix& A_output = tanh_layer.forward(Z_input);
        A_output.copyDeviceToHost(); // Copy result back

        std::vector<float> a_expected(num_elements_total);
        std::transform(z_host_data.begin(), z_host_data.end(), a_expected.begin(),
            [](float z_val) { return std::tanh(z_val); });

        std::vector<float> a_result(num_elements_total);
        if (A_output.data_host) { // Check if host data is valid
            std::copy(A_output.data_host.get(), A_output.data_host.get() + num_elements_total, a_result.begin());
        }
        else {
            std::cerr << "A_output.data_host is null after forward pass + copyDeviceToHost!" << std::endl;
            return 1;
        }


        printVector("Z_host_data", z_host_data);
        printVector("A_expected   ", a_expected);
        printVector("A_result_gpu ", a_result);

        if (compareFloatVectors(a_expected, a_result)) {
            std::cout << "Forward Pass: PASS" << std::endl;
        }
        else {
            std::cout << "Forward Pass: FAIL" << std::endl;
            // return 1; // Optionally exit on first failure
        }

        // --- Test Backward Pass ---
        // The forward pass has already populated tanh_layer.A which is needed for backprop
        std::cout << "\n--- Testing Backward Pass ---" << std::endl;
        Matrix& dZ_output = tanh_layer.backprop(dA_input);
        dZ_output.copyDeviceToHost();

        std::vector<float> dz_expected(num_elements_total);
        for (size_t i = 0; i < num_elements_total; ++i) {
            float a_val = a_result[i]; // This is Tanh(Z) from the forward pass result
            dz_expected[i] = da_host_data[i] * (1.0f - (a_val * a_val));
        }

        std::vector<float> dz_result(num_elements_total);
        if (dZ_output.data_host) {
            std::copy(dZ_output.data_host.get(), dZ_output.data_host.get() + num_elements_total, dz_result.begin());
        }
        else {
            std::cerr << "dZ_output.data_host is null after backprop pass + copyDeviceToHost!" << std::endl;
            return 1;
        }

        printVector("dA_host_data ", da_host_data);
        printVector("A_from_fwd   ", a_result); // (Tanh(Z) values)
        printVector("dZ_expected  ", dz_expected);
        printVector("dZ_result_gpu", dz_result);

        if (compareFloatVectors(dz_expected, dz_result)) {
            std::cout << "Backward Pass: PASS" << std::endl;
        }
        else {
            std::cout << "Backward Pass: FAIL" << std::endl;
        }

    }
    catch (const NNException& e) {
        std::cerr << "NNException caught: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard exception caught: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}