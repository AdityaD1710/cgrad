#include "network.h"
#include "dataset.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>

int main() {
    srand(42);
    std::mt19937 rng(42);

    // ── load data ─────────────────────────────────────────────────────────────

    std::cout << "Loading training data...\n";
    Dataset train = loadMNIST("mnist_train.csv", 10000); // start with 10k for speed

    std::cout << "Loading test data...\n";
    Dataset test  = loadMNIST("mnist_test.csv",  2000);

    // ── build network ─────────────────────────────────────────────────────────
    // 784 inputs (28x28 pixels) → 128 → ReLU → 64 → ReLU → 10 → Softmax

    Network net;
    net.add(new Dense(784, 128));
    net.add(new ReLULayer());
    net.add(new Dense(128, 64));
    net.add(new ReLULayer());
    net.add(new Dense(64, 10));
    net.add(new SoftmaxLayer());
    net.compile(new Adam(0.001f), true); // true = use CrossEntropy
    net.summary();

    // ── training loop ─────────────────────────────────────────────────────────

    int    epochs     = 20;
    int    batch_size = 32;
    int    steps_per_epoch = 200;
    float  best_acc   = 0.0f;

    std::cout << "Training on " << train.num_samples
              << " samples, batch size " << batch_size << "\n\n";

    auto t_start = std::chrono::steady_clock::now();

    for (int epoch = 1; epoch <= epochs; epoch++) {
        float total_loss = 0.0f;

        for (int step = 0; step < steps_per_epoch; step++) {
            auto [bX, by] = getBatch(train, batch_size, rng);
            total_loss   += net.trainStep(bX, by);
        }

        float avg_loss = total_loss / steps_per_epoch;
        float acc      = evaluate(net, test);

        if (acc > best_acc) best_acc = acc;

        auto now     = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - t_start).count();

        std::cout << "Epoch " << std::setw(2) << epoch
                  << "  loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << "  test acc: " << std::setprecision(2) << acc << "%"
                  << "  best: " << best_acc << "%"
                  << "  (" << std::setprecision(1) << elapsed << "s)\n";
    }

    // ── final report ──────────────────────────────────────────────────────────

    std::cout << "\n=== Final Test Accuracy: " << best_acc << "% ===\n";
    std::cout << "\nSample predictions (first 10 test images):\n";

    Tensor sample_X(10, 784);
    Tensor sample_y(10, 10);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 784; j++) sample_X.at(i,j) = test.X.at(i,j);
        for (int j = 0; j < 10;  j++) sample_y.at(i,j) = test.y.at(i,j);
    }

    Tensor preds = net.predict(sample_X);
    for (int i = 0; i < 10; i++) {
        int pred = 0, actual = 0;
        for (int j = 1; j < 10; j++) {
            if (preds.at(i,j)    > preds.at(i,pred))    pred   = j;
            if (sample_y.at(i,j) > sample_y.at(i,actual)) actual = j;
        }
        std::cout << "  Image " << i << ": predicted=" << pred
                  << "  actual=" << actual
                  << (pred == actual ? "  ✓" : "  ✗") << "\n";
    }

    return 0;
}