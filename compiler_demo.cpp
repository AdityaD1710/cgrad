#include "network.h"
#include "dataset.h"
#include "compiler.h"
#include <iostream>

int main() {
    srand(42);
    std::mt19937 rng(42);

    // ── train a small feature extractor (layers 1-2) ──────────────────────────
    std::cout << "=== Training feature extractor ===\n\n";

    Network feature_net;
    feature_net.add(new Dense(784, 128));
    feature_net.add(new ReLULayer());
    feature_net.add(new Dense(128, 64));
    feature_net.add(new ReLULayer());
    feature_net.compile(new Adam(0.001f), true);

    // ── train a small classifier head (layers 3) ──────────────────────────────
    std::cout << "Training classifier head ===\n\n";

    Network classifier_net;
    classifier_net.add(new Dense(64, 10));
    classifier_net.add(new SoftmaxLayer());
    classifier_net.compile(new Adam(0.001f), true);

    // ── train them together as one combined network ───────────────────────────
    std::cout << "Loading data and training combined network...\n\n";

    Dataset train = loadMNIST("mnist_train.csv", 5000);
    Dataset test  = loadMNIST("mnist_test.csv",  1000);

    // build the full combined network for training
    Network full_net;
    full_net.add(new Dense(784, 128));
    full_net.add(new ReLULayer());
    full_net.add(new Dense(128, 64));
    full_net.add(new ReLULayer());
    full_net.add(new Dense(64, 10));
    full_net.add(new SoftmaxLayer());
    full_net.compile(new Adam(0.001f), true);

    for (int epoch = 1; epoch <= 10; epoch++) {
        float total_loss = 0.0f;
        for (int step = 0; step < 100; step++) {
            auto [bX, by] = getBatch(train, 32, rng);
            total_loss += full_net.trainStep(bX, by);
        }
        float acc = evaluate(full_net, test);
        std::cout << "Epoch " << epoch << "  loss: "
                  << total_loss / 100 << "  acc: " << acc << "%\n";
    }

    // ── step 1: compile to binary ─────────────────────────────────────────────
    std::cout << "\n=== Step 1: Compile to binary ===\n";
    ModelCompiler::compile(full_net, "model.bin");

    // ── step 2: emit IR ───────────────────────────────────────────────────────
    std::cout << "\n=== Step 2: Emit IR ===\n";
    ModelCompiler::emitIR(full_net, "model.json");

    // ── step 3: load from binary ──────────────────────────────────────────────
    std::cout << "\n=== Step 3: Load from binary ===\n";
    Network* loaded = ModelCompiler::load("model.bin");
    loaded->compile(new Adam(0.001f), true);

    float loaded_acc = evaluate(*loaded, test);
    std::cout << "Loaded model accuracy: " << loaded_acc
              << "% (should match original)\n";

    // ── step 4: split and link ────────────────────────────────────────────────
    // save first half and second half separately, then link them
    std::cout << "\n=== Step 4: Split + Link ===\n";

    Network part_a;
    part_a.add(new Dense(784, 128));
    part_a.add(new ReLULayer());
    part_a.add(new Dense(128, 64));
    part_a.add(new ReLULayer());

    Network part_b;
    part_b.add(new Dense(64, 10));
    part_b.add(new SoftmaxLayer());

    // copy trained weights from full_net into part_a and part_b
    auto copyWeights = [](Dense* dst, Dense* src) {
        dst->weights = src->weights;
        dst->biases  = src->biases;
    };

    copyWeights(dynamic_cast<Dense*>(part_a.layers[0]),
                dynamic_cast<Dense*>(full_net.layers[0]));
    copyWeights(dynamic_cast<Dense*>(part_a.layers[2]),
                dynamic_cast<Dense*>(full_net.layers[2]));
    copyWeights(dynamic_cast<Dense*>(part_b.layers[0]),
                dynamic_cast<Dense*>(full_net.layers[4]));

    ModelCompiler::compile(part_a, "part_a.bin");
    ModelCompiler::compile(part_b, "part_b.bin");

    Network* linked = ModelCompiler::link("part_a.bin", "part_b.bin");
    linked->compile(new Adam(0.001f), true);

    float linked_acc = evaluate(*linked, test);
    std::cout << "Linked model accuracy: " << linked_acc
              << "% (should match original)\n";

    // ── print the IR ──────────────────────────────────────────────────────────
    std::cout << "\n=== Model IR (model.json) ===\n";
    std::ifstream ir("model.json");
    std::cout << ir.rdbuf() << "\n";

    delete loaded;
    delete linked;
    return 0;
}