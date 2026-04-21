#include "network.h"
#include <iostream>

int main() {
    srand(42);

    // ── XOR dataset ───────────────────────────────────────────────────────────
    // XOR truth table:
    // 0 XOR 0 = 0
    // 0 XOR 1 = 1
    // 1 XOR 0 = 1
    // 1 XOR 1 = 0
    // a linear model cannot learn this — needs hidden layer

    Tensor X(4, 2, {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    });

    Tensor y(4, 1, {
        0,
        1,
        1,
        0
    });

    // ── build network ─────────────────────────────────────────────────────────

    Network net;
    net.add(new Dense(2, 4));       // input → hidden
    net.add(new TanhLayer());       // activation
    net.add(new Dense(4, 1));       // hidden → output
    net.add(new SigmoidLayer());    // output between 0 and 1
    net.compile(new Adam(0.01f));
    net.summary();

    // ── train ─────────────────────────────────────────────────────────────────

    net.fit(X, y, 5000, 500);

    // ── evaluate ──────────────────────────────────────────────────────────────

    std::cout << "=== Predictions ===\n";
    Tensor preds = net.predict(X);
    std::string inputs[4] = {"0 XOR 0", "0 XOR 1", "1 XOR 0", "1 XOR 1"};
    int expected[4] = {0, 1, 1, 0};

    int correct = 0;
    for (int i = 0; i < 4; i++) {
        float p = preds.at(i, 0);
        int predicted = p > 0.5f ? 1 : 0;
        bool ok = predicted == expected[i];
        if (ok) correct++;
        std::cout << "  " << inputs[i] << "  =>  "
                  << std::fixed << std::setprecision(4) << p
                  << "  (predicted: " << predicted
                  << ", expected: " << expected[i]
                  << (ok ? " ✓" : " ✗") << ")\n";
    }
    std::cout << "\nAccuracy: " << correct << "/4\n";

    return 0;
}