#include "graph_perceptron.h"

s21::GraphPerceptron::GraphPerceptron(size_t size) : size_(size) {
  InitLayers();
}

s21::GraphPerceptron::~GraphPerceptron() {
  for (auto& layer : layers_) {
    delete layer;
  }
}

void s21::GraphPerceptron::InitLayers() {
  layers_.push_back(new s21::Layer(kInputLayerSize));
  for (size_t i = 1; i < size_ + 1; ++i) {
    layers_.push_back(new s21::Layer(kHiddenLayerSize, layers_[i - 1]));
  }
  layers_.push_back(new s21::Layer(kOutputLayerSize, layers_[size_]));
}

void s21::GraphPerceptron::SetInputValues(std::vector<double>& input_values) {
  layers_[0]->SetValue(input_values);
}

void s21::GraphPerceptron::FillWeightsRandomly() {
  for (size_t i = 1; i < layers_.size(); ++i) {
    layers_[i]->FillWeightsRandomly();
  }
}

void s21::GraphPerceptron::SetWeights(
    std::vector<std::vector<std::vector<double>>> weights) {
  for (size_t i = 1; i < layers_.size(); ++i) {
    for (size_t j = 0; j < layers_[i]->GetSizeOfLayer(); ++j) {
      layers_[i]->GetLayer()[j].SetWeight(weights[i - 1][j]);
    }
  }
}

void s21::GraphPerceptron::FeedForward() {
  for (size_t i = 1; i < layers_.size(); ++i) {
    layers_[i]->FeedForward();
  }
}

void s21::GraphPerceptron::BackPropagation() {
  layers_[layers_.size() - 1]->CalculateOutputError(1);

  for (int i = layers_.size() - 2; i >= 0; --i) {
    layers_[i]->CalculateError();
  }

  for (size_t i = layers_.size() - 1; i > 0; --i) {
    layers_[i]->UpdateWeights();
  }
}
