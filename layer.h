#pragma once
#include "tensor.h"
#include "activations.h"
#include "optimizer.h"
#include "regularization.h"
#include <memory>
#include <string>
// ── LAYER BASE ────────────────────────────────────────────────────────────────

struct Layer {
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad)  = 0;
    virtual void   updateWeights(Optimizer* opt)  {}
    virtual std::string name() const = 0;
    virtual ~Layer() = default;
};

// ── DENSE LAYER ───────────────────────────────────────────────────────────────
// y = x * W + b
// forward:  output = input @ weights + bias
// backward: dW = input.T @ grad_output
//           dB = sum(grad_output, axis=0)
//           dx = grad_output @ weights.T  ← passed to previous layer

struct Dense : Layer {
    Tensor weights;   // shape: (in_features x out_features)
    Tensor biases;    // shape: (1 x out_features)
    Tensor last_input;// saved for backward pass

    // optional regularizer — nullptr means no regularization
    L1Reg*         l1  = nullptr;
    L2Reg*         l2  = nullptr;
    ElasticNetReg* en  = nullptr;

    Dense(int in_features, int out_features) {
        weights = Tensor(in_features, out_features);
        biases  = Tensor(1, out_features);
        weights.xavier(in_features, out_features);
        biases.fill(0.0f);
    }

    void setL1(float lambda)                  { l1 = new L1Reg(lambda); }
    void setL2(float lambda)                  { l2 = new L2Reg(lambda); }
    void setElasticNet(float lam, float alpha) { en = new ElasticNetReg(lam, alpha); }

    Tensor forward(const Tensor& input) override {
        last_input = input;
        return input.matmul(weights).addBias(biases);
    }

    Tensor backward(const Tensor& grad) override {
        // gradient w.r.t weights: input.T @ grad
        Tensor dW = last_input.transpose().matmul(grad);

        // add regularization gradient if set
        if (l1) dW += l1->gradient(weights);
        if (l2) dW += l2->gradient(weights);
        if (en) dW += en->gradient(weights);

        // gradient w.r.t bias: sum across batch (rows)
        Tensor dB(1, biases.cols);
        for (int r = 0; r < grad.rows; r++)
            for (int c = 0; c < grad.cols; c++)
                dB.at(0, c) += grad.at(r, c);

        // save for weight update step
        last_dW = dW;
        last_dB = dB;

        // gradient w.r.t input: grad @ weights.T → passed to prev layer
        return grad.matmul(weights.transpose());
    }

    void updateWeights(Optimizer* opt) override {
        opt->update(weights, biases, last_dW, last_dB);
    }

    std::string name() const override { return "Dense"; }

    ~Dense() { delete l1; delete l2; delete en; }

private:
    Tensor last_dW, last_dB;
};

// ── ACTIVATION LAYERS (wrap activations as layers) ───────────────────────────

struct ReLULayer : Layer {
    ReLU activation;
    Tensor forward(const Tensor& input) override { return activation.forward(input); }
    Tensor backward(const Tensor& grad) override { return activation.backward(grad); }
    std::string name() const override { return "ReLU"; }
};

struct SigmoidLayer : Layer {
    Sigmoid activation;
    Tensor forward(const Tensor& input) override { return activation.forward(input); }
    Tensor backward(const Tensor& grad) override { return activation.backward(grad); }
    std::string name() const override { return "Sigmoid"; }
};

struct TanhLayer : Layer {
    Tanh activation;
    Tensor forward(const Tensor& input) override { return activation.forward(input); }
    Tensor backward(const Tensor& grad) override { return activation.backward(grad); }
    std::string name() const override { return "Tanh"; }
};

struct SoftmaxLayer : Layer {
    Softmax activation;
    Tensor forward(const Tensor& input) override { return activation.forward(input); }
    Tensor backward(const Tensor& grad) override { return activation.backward(grad); }
    std::string name() const override { return "Softmax"; }
};