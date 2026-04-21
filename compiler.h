#pragma once
#include "network.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

// ─────────────────────────────────────────────────────────────────────────────
// ModelCompiler — serializes, links, and loads trained networks
//
// Mirrors real compiler/linker concepts:
//   compile()  → trained Network → binary .bin file  (like: source → object file)
//   emitIR()   → trained Network → human-readable .json graph (like: LLVM IR)
//   link()     → two .bin files  → one chained Network (like: linker combines objects)
//   load()     → .bin file       → live Network ready for inference (like: loader)
// ─────────────────────────────────────────────────────────────────────────────

class ModelCompiler {
public:

    // ── compile ───────────────────────────────────────────────────────────────
    // serializes a trained Network to a binary .bin file
    // format per Dense layer:
    //   [4 bytes: rows][4 bytes: cols][rows*cols*4 bytes: weight floats]
    //   [4 bytes: 1   ][4 bytes: cols][cols*4 bytes: bias floats]
    // non-Dense layers (activations) stored as a 1-byte type tag only

    static void compile(Network& net, const std::string& path) {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("compile: cannot open " + path);

        // write magic header so we can validate on load
        const char magic[8] = "CGRAD01";
        file.write(magic, 8);

        // write number of layers
        int num_layers = (int)net.layers.size();
        file.write((char*)&num_layers, sizeof(int));

        for (auto* layer : net.layers) {
            // write layer type tag as a length-prefixed string
            std::string tag = layer->name();
            int tag_len = (int)tag.size();
            file.write((char*)&tag_len, sizeof(int));
            file.write(tag.c_str(), tag_len);

            // if Dense, write weights and biases
            if (tag == "Dense") {
                Dense* d = dynamic_cast<Dense*>(layer);
                writeTensor(file, d->weights);
                writeTensor(file, d->biases);
            }
            // activation layers have no weights — tag is enough to reconstruct
        }

        file.close();
        std::cout << "Compiled model saved to: " << path << "\n";
        std::cout << "  Layers: " << num_layers << "\n";
        printSize(path);
    }

    // ── load ──────────────────────────────────────────────────────────────────
    // deserializes a .bin file back into a live Network
    // rebuilds layer objects from type tags, restores weights exactly

    static Network* load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("load: cannot open " + path);

        // validate magic header
        char magic[8];
        file.read(magic, 8);
        if (std::string(magic, 7) != "CGRAD01")
            throw std::runtime_error("load: invalid file format");

        int num_layers;
        file.read((char*)&num_layers, sizeof(int));

        Network* net = new Network();

        for (int i = 0; i < num_layers; i++) {
            // read type tag
            int tag_len;
            file.read((char*)&tag_len, sizeof(int));
            std::string tag(tag_len, ' ');
            file.read(&tag[0], tag_len);

            // reconstruct layer from tag — this is the symbol table lookup
            if (tag == "Dense") {
                Tensor w = readTensor(file);
                Tensor b = readTensor(file);
                Dense* d = new Dense(w.rows, w.cols);
                d->weights = w;
                d->biases  = b;
                net->add(d);
            }
            else if (tag == "ReLU")    net->add(new ReLULayer());
            else if (tag == "Sigmoid") net->add(new SigmoidLayer());
            else if (tag == "Tanh")    net->add(new TanhLayer());
            else if (tag == "Softmax") net->add(new SoftmaxLayer());
            else throw std::runtime_error("load: unknown layer type: " + tag);
        }

        file.close();
        std::cout << "Loaded model from: " << path << " (" << num_layers << " layers)\n";
        return net;
    }

    // ── link ──────────────────────────────────────────────────────────────────
    // combines two saved .bin models into one chained Network
    // output of model A feeds directly into model B
    // mirrors how a linker combines two object files into one executable

    static Network* link(const std::string& pathA,
                         const std::string& pathB) {
        std::cout << "Linking: " << pathA << " + " << pathB << "\n";

        Network* a = load(pathA);
        Network* b = load(pathB);

        // steal layers from both into a new combined network
        Network* linked = new Network();
        for (auto* l : a->layers) linked->layers.push_back(l);
        for (auto* l : b->layers) linked->layers.push_back(l);

        // clear original vectors so destructor doesn't double-free
        a->layers.clear();
        b->layers.clear();
        delete a;
        delete b;

        std::cout << "Linked model has " << linked->layers.size() << " layers total\n";
        return linked;
    }

    // ── emitIR ────────────────────────────────────────────────────────────────
    // dumps a human-readable JSON graph of the network
    // analogous to LLVM IR — sits between model definition and binary
    // useful for debugging, visualization, and framework interop

    static void emitIR(Network& net, const std::string& path) {
        std::ofstream file(path);
        if (!file.is_open())
            throw std::runtime_error("emitIR: cannot open " + path);

        file << "{\n";
        file << "  \"framework\": \"cgrad\",\n";
        file << "  \"version\": \"1.0\",\n";
        file << "  \"num_layers\": " << net.layers.size() << ",\n";
        file << "  \"layers\": [\n";

        for (int i = 0; i < (int)net.layers.size(); i++) {
            auto* layer = net.layers[i];
            std::string tag = layer->name();
            file << "    {\n";
            file << "      \"index\": " << i << ",\n";
            file << "      \"type\": \"" << tag << "\"";

            if (tag == "Dense") {
                Dense* d = dynamic_cast<Dense*>(layer);
                file << ",\n      \"in_features\": "  << d->weights.rows;
                file << ",\n      \"out_features\": " << d->weights.cols;
                file << ",\n      \"params\": "
                     << d->weights.size() + d->biases.size();

                // sample first 4 weights as a sanity check
                file << ",\n      \"weight_sample\": [";
                for (int j = 0; j < std::min(4, d->weights.size()); j++) {
                    file << std::fixed << std::setprecision(6)
                         << d->weights.data[j];
                    if (j < 3) file << ", ";
                }
                file << "]";
            }

            file << "\n    }";
            if (i < (int)net.layers.size() - 1) file << ",";
            file << "\n";
        }

        file << "  ]\n}\n";
        file.close();
        std::cout << "IR emitted to: " << path << "\n";
    }

private:
    static void writeTensor(std::ofstream& file, const Tensor& t) {
        file.write((char*)&t.rows, sizeof(int));
        file.write((char*)&t.cols, sizeof(int));
        file.write((char*)t.data.data(), t.size() * sizeof(float));
    }

    static Tensor readTensor(std::ifstream& file) {
        int rows, cols;
        file.read((char*)&rows, sizeof(int));
        file.read((char*)&cols, sizeof(int));
        Tensor t(rows, cols);
        file.read((char*)t.data.data(), t.size() * sizeof(float));
        return t;
    }

    static void printSize(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            float kb = file.tellg() / 1024.0f;
            std::cout << "  File size: " << std::fixed
                      << std::setprecision(1) << kb << " KB\n";
        }
    }
};