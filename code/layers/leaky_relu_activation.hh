// leaky_relu_activation.hh
#pragma once

#include "nn_layer.hh" // Assuming NNLayer is defined here or included by it

class LeakyReLUActivation : public NNLayer {
private:
    Matrix A;    // Output of the activation (LeakyReLU(Z))
    Matrix Z;    // Input to the activation (cached from forward pass)
    Matrix dZ;   // Gradient dL/dZ (output of backprop)
    float alpha; // Slope for negative inputs

public:
    // Constructor takes the layer name and the alpha value
    LeakyReLUActivation(std::string name, float alpha_val = 0.01f);
    ~LeakyReLUActivation();

    // Forward pass: A = LeakyReLU(Z)
    Matrix& forward(Matrix& Z_input);

    void get_learnable_parameters(ParameterGroup& group) override;

    // Backward pass: dL/dZ = dL/dA * LeakyReLU'(Z)
    // learning_rate is not used by activation layers' backprop directly for parameter updates
    Matrix& backprop(Matrix& dA_input); // learning_rate unused

    // Getter for alpha if needed outside
    float getAlpha() const { return alpha; }
    
};