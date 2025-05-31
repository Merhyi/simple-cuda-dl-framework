#pragma once
#include "matrix.hh"
#include "cost_function_base.hh"

class MSECost : public CostFunction {
public:
	MSECost() = default; // Default constructor
	~MSECost() override = default; // Good practice to override virtual destructor

	float cost(Matrix predictions, Matrix target) override;
	Matrix dCost(Matrix predictions, Matrix target, Matrix dY) override;
};
