#pragma once
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <cassert>
#include <functional>
#include <iomanip>

class Tensor {
public:
    std::vector<float> data;
    int rows, cols;

    // ── constructors ──────────────────────────────────────────────────────────

    Tensor() : rows(0), cols(0) {}

    Tensor(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}

    Tensor(int r, int c, std::vector<float> d)
        : rows(r), cols(c), data(std::move(d)) {
        if ((int)data.size() != r * c)
            throw std::runtime_error("Tensor: data size mismatch");
    }

    // ── element access ────────────────────────────────────────────────────────

    float& at(int r, int c) {
        return data[r * cols + c];
    }

    const float& at(int r, int c) const {
        return data[r * cols + c];
    }

    // ── shape info ────────────────────────────────────────────────────────────

    int size() const { return rows * cols; }

    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << ":\n";
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++)
                std::cout << std::setw(8) << std::fixed
                          << std::setprecision(4) << at(r, c);
            std::cout << "\n";
        }
        std::cout << "shape: (" << rows << " x " << cols << ")\n\n";
    }

    // ── initialization ────────────────────────────────────────────────────────

    // fill every element with a value
    void fill(float val) {
        for (auto& x : data) x = val;
    }

    // Xavier initialization — keeps gradients stable during training
    // scale = sqrt(2 / (fan_in + fan_out))
    void xavier(int fan_in, int fan_out) {
        float scale = std::sqrt(2.0f / (fan_in + fan_out));
        for (auto& x : data)
            x = scale * (2.0f * ((float)rand() / RAND_MAX) - 1.0f);
    }

    // ── math operations ───────────────────────────────────────────────────────

    // matrix multiply: (this) x (other) → result
    // this is (M x K), other is (K x N), result is (M x N)
    Tensor matmul(const Tensor& other) const {
        if (cols != other.rows)
            throw std::runtime_error(
                "matmul: shape mismatch (" +
                std::to_string(rows) + "x" + std::to_string(cols) +
                ") x (" + std::to_string(other.rows) +
                "x" + std::to_string(other.cols) + ")");

        Tensor result(rows, other.cols);
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < cols; k++)
                for (int j = 0; j < other.cols; j++)
                    result.at(i, j) += at(i, k) * other.at(k, j);
        return result;
    }

    // transpose: (M x N) → (N x M)
    Tensor transpose() const {
        Tensor result(cols, rows);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.at(c, r) = at(r, c);
        return result;
    }

    // element-wise add
    Tensor operator+(const Tensor& other) const {
        assertSameShape(other, "+");
        Tensor result(rows, cols);
        for (int i = 0; i < size(); i++)
            result.data[i] = data[i] + other.data[i];
        return result;
    }

    // element-wise subtract
    Tensor operator-(const Tensor& other) const {
        assertSameShape(other, "-");
        Tensor result(rows, cols);
        for (int i = 0; i < size(); i++)
            result.data[i] = data[i] - other.data[i];
        return result;
    }

    // element-wise multiply (Hadamard product)
    Tensor operator*(const Tensor& other) const {
        assertSameShape(other, "*");
        Tensor result(rows, cols);
        for (int i = 0; i < size(); i++)
            result.data[i] = data[i] * other.data[i];
        return result;
    }

    // scalar multiply
    Tensor operator*(float scalar) const {
        Tensor result(rows, cols);
        for (int i = 0; i < size(); i++)
            result.data[i] = data[i] * scalar;
        return result;
    }

    // in-place subtract (used by optimizer: weights -= lr * grad)
    Tensor& operator-=(const Tensor& other) {
        assertSameShape(other, "-=");
        for (int i = 0; i < size(); i++)
            data[i] -= other.data[i];
        return *this;
    }

    // in-place add
    Tensor& operator+=(const Tensor& other) {
        assertSameShape(other, "+=");
        for (int i = 0; i < size(); i++)
            data[i] += other.data[i];
        return *this;
    }

    // apply any function element-wise: relu, sigmoid, etc.
    Tensor apply(std::function<float(float)> fn) const {
        Tensor result(rows, cols);
        for (int i = 0; i < size(); i++)
            result.data[i] = fn(data[i]);
        return result;
    }

    // sum of all elements (used in loss functions)
    float sum() const {
        float s = 0;
        for (auto x : data) s += x;
        return s;
    }

    // mean of all elements
    float mean() const {
        return sum() / size();
    }

    // add bias row-wise: tensor is (batch x cols), bias is (1 x cols)
    Tensor addBias(const Tensor& bias) const {
        if (bias.rows != 1 || bias.cols != cols)
            throw std::runtime_error("addBias: bias shape mismatch");
        Tensor result(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.at(r, c) = at(r, c) + bias.at(0, c);
        return result;
    }

private:
    void assertSameShape(const Tensor& other, const std::string& op) const {
        if (rows != other.rows || cols != other.cols)
            throw std::runtime_error(
                "Tensor " + op + ": shape mismatch (" +
                std::to_string(rows) + "x" + std::to_string(cols) +
                ") vs (" + std::to_string(other.rows) +
                "x" + std::to_string(other.cols) + ")");
    }
};