// tanh_activation.hh
#pragma once

#include "nn_layer.hh" 

class TanhActivation : public NNLayer {
private:

    Matrix A;

    Matrix dZ; 

public:

    TanhActivation(std::string name);

    ~TanhActivation();

    Matrix& forward(Matrix& Z_input); // Renamed Z to Z_input for clarity


    Matrix& backprop(Matrix& dA_input); // Renamed dA to dA_input


    void get_learnable_parameters(ParameterGroup& group) override;
};