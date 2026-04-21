#pragma once
#include "tensor.h"
#include <cmath>

// ── RELU ──────────────────────────────────────────────────────────────────────
// forward:  f(x) = max(0, x)
// backward: f'(x) = 1 if x > 0 else 0

struct ReLU {
    Tensor last_input; // saved for backprop

    Tensor forward(const Tensor& input) {
        last_input = input;
        return input.apply([](float x) { return x > 0 ? x : 0.0f; });
    }

    Tensor backward(const Tensor& grad) {
        // gradient only flows where input was positive
        return grad * last_input.apply([](float x) {
            return x > 0 ? 1.0f : 0.0f;
        });
    }
};

// ── SIGMOID ───────────────────────────────────────────────────────────────────
// forward:  f(x) = 1 / (1 + e^-x)
// backward: f'(x) = f(x) * (1 - f(x))

struct Sigmoid {
    Tensor last_output;

    Tensor forward(const Tensor& input) {
        last_output = input.apply([](float x) {
            return 1.0f / (1.0f + std::exp(-x));
        });
        return last_output;
    }

    Tensor backward(const Tensor& grad) {
        // reuse saved output — no need to recompute sigmoid
        Tensor ones(last_output.rows, last_output.cols);
        ones.fill(1.0f);
        Tensor sig_deriv = last_output * (ones - last_output);
        return grad * sig_deriv;
    }
};

// ── TANH ──────────────────────────────────────────────────────────────────────
// forward:  f(x) = tanh(x)
// backward: f'(x) = 1 - tanh(x)^2

struct Tanh {
    Tensor last_output;

    Tensor forward(const Tensor& input) {
        last_output = input.apply([](float x) { return std::tanh(x); });
        return last_output;
    }

    Tensor backward(const Tensor& grad) {
        Tensor ones(last_output.rows, last_output.cols);
        ones.fill(1.0f);
        Tensor tanh_deriv = ones - (last_output * last_output);
        return grad * tanh_deriv;
    }
};

// ── SOFTMAX ───────────────────────────────────────────────────────────────────
// forward:  f(x_i) = e^x_i / sum(e^x_j)
// used in output layer for multi-class classification
// backward: handled together with CrossEntropy loss (simplifies to y_hat - y)

struct Softmax {
    Tensor last_output;

    Tensor forward(const Tensor& input) {
        Tensor result(input.rows, input.cols);
        for (int r = 0; r < input.rows; r++) {
            // subtract max for numerical stability
            float max_val = input.at(r, 0);
            for (int c = 1; c < input.cols; c++)
                max_val = std::max(max_val, input.at(r, c));

            float sum = 0.0f;
            for (int c = 0; c < input.cols; c++) {
                result.at(r, c) = std::exp(input.at(r, c) - max_val);
                sum += result.at(r, c);
            }
            for (int c = 0; c < input.cols; c++)
                result.at(r, c) /= sum;
        }
        last_output = result;
        return result;
    }

    Tensor backward(const Tensor& grad) {
        // when paired with CrossEntropy, gradient is simply (y_hat - y)
        // which is already computed in the loss — so we just pass grad through
        return grad;
    }
};