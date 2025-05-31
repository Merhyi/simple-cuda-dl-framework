#pragma once
#include "matrix.hh"
#include "cost_function_base.hh"

class BCECost : public CostFunction {
public:
	float cost(Matrix predictions, Matrix target);
	Matrix dCost(Matrix predictions, Matrix target, Matrix dY);
};
