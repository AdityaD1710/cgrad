#pragma once
#include "tensor.h"
#include <cmath>

struct Optimizer {
    virtual void update(Tensor& weights, Tensor& biases,
                        const Tensor& dW,  const Tensor& dB) = 0;
    virtual ~Optimizer() = default;
};

struct SGD : Optimizer {
    float lr;
    SGD(float learning_rate = 0.01f) : lr(learning_rate) {}

    void update(Tensor& weights, Tensor& biases,
                const Tensor& dW,  const Tensor& dB) override {
        weights -= dW * lr;
        biases  -= dB * lr;
    }
};

struct Adam : Optimizer {
    float lr, beta1, beta2, epsilon;
    int   t = 0;
    Tensor mW, vW, mB, vB;
    bool initialized = false;

    Adam(float lr = 0.001f, float b1 = 0.9f,
         float b2 = 0.999f, float eps = 1e-8f)
        : lr(lr), beta1(b1), beta2(b2), epsilon(eps) {}

    void update(Tensor& weights, Tensor& biases,
                const Tensor& dW,  const Tensor& dB) override {
        if (!initialized) {
            mW = Tensor(weights.rows, weights.cols);
            vW = Tensor(weights.rows, weights.cols);
            mB = Tensor(biases.rows,  biases.cols);
            vB = Tensor(biases.rows,  biases.cols);
            initialized = true;
        }
        t++;
        updateParam(weights, dW, mW, vW);
        updateParam(biases,  dB, mB, vB);
    }

private:
    void updateParam(Tensor& param, const Tensor& grad,
                     Tensor& m,     Tensor& v) {
        float bc1 = 1.0f - std::pow(beta1, t);
        float bc2 = 1.0f - std::pow(beta2, t);
        for (int i = 0; i < param.size(); i++) {
            m.data[i] = beta1 * m.data[i] + (1 - beta1) * grad.data[i];
            v.data[i] = beta2 * v.data[i] + (1 - beta2) * grad.data[i] * grad.data[i];
            float m_hat = m.data[i] / bc1;
            float v_hat = v.data[i] / bc2;
            param.data[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
};