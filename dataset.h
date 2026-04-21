#pragma once
#include "tensor.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>

struct Dataset {
    Tensor X;  // shape: (num_samples x 784)
    Tensor y;  // shape: (num_samples x 10)  one-hot encoded
    int num_samples;

    Dataset(int n) : X(n, 784), y(n, 10), num_samples(n) {}
};

// ── load CSV ──────────────────────────────────────────────────────────────────
// each row: label, pixel0, pixel1, ..., pixel783

Dataset loadMNIST(const std::string& path, int max_samples = -1) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    std::string line;
    std::getline(file, line); // skip header row

    // count rows first if no limit given
    std::vector<std::pair<int, std::vector<float>>> rows;
    rows.reserve(max_samples > 0 ? max_samples : 60000);

    int count = 0;
    while (std::getline(file, line)) {
        if (max_samples > 0 && count >= max_samples) break;

        std::stringstream ss(line);
        std::string cell;
        std::vector<float> pixels;
        int label = -1;
        int col = 0;

        while (std::getline(ss, cell, ',')) {
            if (col == 0)
                label = std::stoi(cell);
            else
                pixels.push_back(std::stof(cell) / 255.0f); // normalize to [0,1]
            col++;
        }

        if (label >= 0 && (int)pixels.size() == 784)
            rows.push_back({label, std::move(pixels)});
        count++;

        if (count % 10000 == 0)
            std::cout << "  Loaded " << count << " samples...\n";
    }

    int n = (int)rows.size();
    Dataset ds(n);

    for (int i = 0; i < n; i++) {
        // copy pixels into X
        for (int j = 0; j < 784; j++)
            ds.X.at(i, j) = rows[i].second[j];

        // one-hot encode label into y
        ds.y.at(i, rows[i].first) = 1.0f;
    }

    std::cout << "  Done. Loaded " << n << " samples.\n\n";
    return ds;
}

// ── mini-batch sampler ────────────────────────────────────────────────────────
// returns a random batch of size batch_size from the dataset

std::pair<Tensor, Tensor> getBatch(const Dataset& ds,
                                    int batch_size,
                                    std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, ds.num_samples - 1);

    Tensor bX(batch_size, 784);
    Tensor by(batch_size, 10);

    for (int i = 0; i < batch_size; i++) {
        int idx = dist(rng);
        for (int j = 0; j < 784; j++)
            bX.at(i, j) = ds.X.at(idx, j);
        for (int j = 0; j < 10; j++)
            by.at(i, j) = ds.y.at(idx, j);
    }

    return {bX, by};
}

// ── accuracy evaluation ───────────────────────────────────────────────────────
// runs the full test set through the network in chunks, returns accuracy %

float evaluate(Network& net, const Dataset& ds, int chunk_size = 500) {
    int correct = 0;
    int total   = ds.num_samples;

    for (int start = 0; start < total; start += chunk_size) {
        int end  = std::min(start + chunk_size, total);
        int size = end - start;

        Tensor bX(size, 784);
        Tensor by(size, 10);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < 784; j++)
                bX.at(i, j) = ds.X.at(start + i, j);
            for (int j = 0; j < 10; j++)
                by.at(i, j) = ds.y.at(start + i, j);
        }

        Tensor preds = net.predict(bX);

        for (int i = 0; i < size; i++) {
            // argmax of prediction
            int pred_label = 0;
            float max_val  = preds.at(i, 0);
            for (int j = 1; j < 10; j++) {
                if (preds.at(i, j) > max_val) {
                    max_val    = preds.at(i, j);
                    pred_label = j;
                }
            }
            // argmax of ground truth
            int true_label = 0;
            for (int j = 1; j < 10; j++)
                if (by.at(i, j) > by.at(i, true_label))
                    true_label = j;

            if (pred_label == true_label) correct++;
        }
    }

    return 100.0f * correct / total;
}