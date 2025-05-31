#pragma once
#include "matrix.hh"

class CostFunction {
public:
    virtual ~CostFunction() = default;

    /**
     * @brief Computes the cost.
     * @param predictions Model's output.
     * @param target Ground truth.
     * @return The scalar cost value.
     */
    virtual float cost(const Matrix predictions, const Matrix target) = 0;

    /**
     * @brief Computes the derivative of the cost with respect to the predictions.
     * @param predictions Model's output.
     * @param target Ground truth.
     * @param dY Output matrix to store the derivatives (dL/dPredictions).
     *           Its memory should be pre-allocated and have the same shape as predictions.
     * @return The dY matrix (often for convenience in chaining, though dY is modified in-place).
     */
    virtual Matrix dCost(const Matrix predictions, const Matrix target, Matrix dY) = 0;
    // If dY is always created by dCost, the signature might be:
    // virtual Matrix dCost(const Matrix& predictions, const Matrix& target) = 0;
    // But your snippet suggests dY is passed in.
};