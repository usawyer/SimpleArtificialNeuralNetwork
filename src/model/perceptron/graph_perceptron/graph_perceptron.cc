#include "graph_perceptron.h"

s21::GraphPerceptron::GraphPerceptron(size_t size) : size_(size) {
  InitLayers();
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

void s21::GraphPerceptron::FeefForward() {
  for (size_t i = 1; i < layers_.size(); ++i) {
    layers_[i]->FeefForward();
  }
}

void s21::GraphPerceptron::BackPropagation() {}

int main() {
  std::vector<std::vector<std::vector<double>>> weights{
      {{0.9, 0.3, 0.4}, {0.2, 0.8, 0.2}, {0.1, 0.5, 0.6}},
      {{0.3, 0.7, 0.5}, {0.6, 0.5, 0.2}, {0.8, 0.1, 0.9}}};
  std::vector<double> value{0.9, 0.1, 0.8};
  s21::GraphPerceptron test(1);

  test.SetInputValues(value);
  // test.FillWeightsRandomly();
  test.SetWeights(weights);
  test.FeefForward();

  // std::vector<s21::Layer*> layers = test.GetLayerInfo();

  // for (size_t i = 0; i < layers.size(); ++i) {
  //   std::cout << layers[i]->GetSizeOfLayer() << std::endl;
  //   std::vector<s21::Neuron> neurons = layers[i]->GetLayer();

  //   for (size_t j = 0; j < neurons.size(); ++j) {
  //     std::vector<double> weight = neurons[j].GetWeight();
  //     std::cout << "[" << j << "]  value = " << neurons[j].GetValue()
  //               << " raw value = " << neurons[j].GetRawValue() << std::endl;
  //     if (weight.size() > 0) {
  //       for (size_t k = 0; k < weight.size(); ++k) {
  //         std::cout << weight[k] << " ";
  //       }
  //       std::cout << std::endl;
  //     }
  //   }
  //   std::cout << std::endl;
  // }
}
