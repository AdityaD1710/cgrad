#pragma once
#include "tensor.h"
#include <cmath>

// ── L1 REGULARIZATION ─────────────────────────────────────────────────────────
// penalty: lambda * sum(|w|)
// gradient: lambda * sign(w)
// effect: drives weights to exactly zero → sparse models

struct L1Reg {
    float lambda;
    L1Reg(float l = 0.01f) : lambda(l) {}

    float penalty(const Tensor& w) {
        float p = 0.0f;
        for (auto x : w.data) p += std::abs(x);
        return lambda * p;
    }

    Tensor gradient(const Tensor& w) {
        return w.apply([this](float x) {
            return lambda * (x > 0 ? 1.0f : (x < 0 ? -1.0f : 0.0f));
        });
    }
};

// ── L2 REGULARIZATION ─────────────────────────────────────────────────────────
// penalty: (lambda/2) * sum(w^2)
// gradient: lambda * w
// effect: drives weights toward zero but never exactly → smooth models

struct L2Reg {
    float lambda;
    L2Reg(float l = 0.01f) : lambda(l) {}

    float penalty(const Tensor& w) {
        float p = 0.0f;
        for (auto x : w.data) p += x * x;
        return (lambda / 2.0f) * p;
    }

    Tensor gradient(const Tensor& w) {
        return w * lambda;
    }
};

// ── ELASTICNET REGULARIZATION ─────────────────────────────────────────────────
// penalty: alpha * L1 + (1 - alpha) * L2
// gradient: alpha * sign(w) + (1 - alpha) * lambda * w
// effect: balance between sparsity (L1) and smoothness (L2)
// alpha = 1.0 → pure L1, alpha = 0.0 → pure L2

struct ElasticNetReg {
    float lambda;
    float alpha; // mix ratio: 1.0 = pure L1, 0.0 = pure L2

    ElasticNetReg(float l = 0.01f, float a = 0.5f) : lambda(l), alpha(a) {}

    float penalty(const Tensor& w) {
        float l1 = 0.0f, l2 = 0.0f;
        for (auto x : w.data) {
            l1 += std::abs(x);
            l2 += x * x;
        }
        return lambda * (alpha * l1 + (1.0f - alpha) * 0.5f * l2);
    }

    Tensor gradient(const Tensor& w) {
        return w.apply([this](float x) {
            float l1_grad = alpha * (x > 0 ? 1.0f : (x < 0 ? -1.0f : 0.0f));
            float l2_grad = (1.0f - alpha) * x;
            return lambda * (l1_grad + l2_grad);
        });
    }
};