#pragma once
#include "tensor.h"
#include <cmath>

// ── MSE LOSS ──────────────────────────────────────────────────────────────────
// forward:  L = mean((y_hat - y)^2)
// backward: dL/dy_hat = 2 * (y_hat - y) / N

struct MSELoss {
    Tensor last_grad;

    float forward(const Tensor& y_hat, const Tensor& y) {
        Tensor diff = y_hat - y;
        last_grad = diff * (2.0f / diff.size());
        float loss = 0.0f;
        for (auto x : diff.data) loss += x * x;
        return loss / diff.size();
    }

    Tensor backward() {
        return last_grad;
    }
};

// ── CROSS ENTROPY LOSS ────────────────────────────────────────────────────────
// forward:  L = -sum(y * log(y_hat))
// backward: dL/dy_hat = y_hat - y  (when combined with Softmax)
// used for multi-class classification

struct CrossEntropyLoss {
    Tensor last_grad;

    float forward(const Tensor& y_hat, const Tensor& y) {
        float loss = 0.0f;
        last_grad = Tensor(y_hat.rows, y_hat.cols);

        for (int r = 0; r < y_hat.rows; r++) {
            for (int c = 0; c < y_hat.cols; c++) {
                // clamp to avoid log(0)
                float p = std::max(y_hat.at(r, c), 1e-7f);
                loss -= y.at(r, c) * std::log(p);
                // gradient: y_hat - y (Softmax + CrossEntropy combined)
                last_grad.at(r, c) = (y_hat.at(r, c) - y.at(r, c)) / y_hat.rows;
            }
        }
        return loss / y_hat.rows;
    }

    Tensor backward() {
        return last_grad;
    }
};