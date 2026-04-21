#pragma once
#include "layer.h"
#include "loss.h"
#include <vector>
#include <iostream>
#include <iomanip>

struct Network {
    std::vector<Layer*> layers;
    Optimizer*          optimizer = nullptr;
    MSELoss*            mse_loss  = nullptr;
    CrossEntropyLoss*   ce_loss   = nullptr;
    bool                use_ce    = false;

    // ── build the network ─────────────────────────────────────────────────────

    void add(Layer* layer) {
        layers.push_back(layer);
    }

    void compile(Optimizer* opt, bool crossentropy = false) {
        optimizer = opt;
        use_ce    = crossentropy;
        if (use_ce) ce_loss  = new CrossEntropyLoss();
        else        mse_loss = new MSELoss();
    }

    void summary() {
        std::cout << "\n=== Network Summary ===\n";
        for (auto* l : layers)
            std::cout << "  " << l->name() << "\n";
        std::cout << "  Loss: " << (use_ce ? "CrossEntropy" : "MSE") << "\n";
        std::cout << "=======================\n\n";
    }

    // ── forward pass ──────────────────────────────────────────────────────────

    Tensor predict(const Tensor& input) {
        Tensor out = input;
        for (auto* layer : layers)
            out = layer->forward(out);
        return out;
    }

    // ── training step ─────────────────────────────────────────────────────────

    float trainStep(const Tensor& X, const Tensor& y) {
        // forward
        Tensor y_hat = predict(X);

        // compute loss
        float loss = use_ce ? ce_loss->forward(y_hat, y)
                            : mse_loss->forward(y_hat, y);

        // get initial gradient from loss
        Tensor grad = use_ce ? ce_loss->backward()
                             : mse_loss->backward();

        // backward — reverse through layers
        for (int i = (int)layers.size() - 1; i >= 0; i--)
            grad = layers[i]->backward(grad);

        // update weights in all layers
        for (auto* layer : layers)
            layer->updateWeights(optimizer);

        return loss;
    }

    // ── fit (full training loop) ──────────────────────────────────────────────

    void fit(const Tensor& X, const Tensor& y,
             int epochs, int print_every = 100) {
        std::cout << "Training...\n";
        for (int epoch = 1; epoch <= epochs; epoch++) {
            float loss = trainStep(X, y);
            if (epoch % print_every == 0 || epoch == 1)
                std::cout << "  Epoch " << std::setw(5) << epoch
                          << "  loss: " << std::fixed
                          << std::setprecision(6) << loss << "\n";
        }
        std::cout << "Done.\n\n";
    }

    ~Network() {
        for (auto* l : layers) delete l;
        delete optimizer;
        delete mse_loss;
        delete ce_loss;
    }
};