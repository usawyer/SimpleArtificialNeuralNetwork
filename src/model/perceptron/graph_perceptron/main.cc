#ifndef SRC_MODEL_MAIN_H_
#define SRC_MODEL_MAIN_H_

#include <iostream>

#include "graph_perceptron.h"

int main() {
  // std::vector<std::vector<std::vector<double>>> weights{
  //     {{0.9, 0.3, 0.4}, {0.2, 0.8, 0.2}, {0.1, 0.5, 0.6}},
  //     {{0.3, 0.7, 0.5}, {0.6, 0.5, 0.2}, {0.8, 0.1, 0.9}}};
  // std::vector<double> value{0.9, 0.1, 0.8};
  // s21::GraphPerceptron test(3);

  // test.SetInputValues(value);
  // test.FillWeightsRandomly();
  // test.SetWeights(weights);
  // test.FeedForward();
  // test.BackPropagation();

  // std::vector<s21::Layer*> layers = test.GetLayerInfo();

  // for (size_t i = 0; i < layers.size(); ++i) {
  //   // std::cout << layers[i]->GetSizeOfLayer() << std::endl;
  //   std::vector<s21::Neuron> neurons = layers[i]->GetLayer();

  //   for (size_t j = 0; j < neurons.size(); ++j) {
  //     std::vector<double> weight = neurons[j].GetWeight();
  //     std::cout << "[" << j << "] value = " << neurons[j].GetValue()
  //               << std::endl;
  //     // if (weight.size() > 0) {
  //     //   std::cout << "weight = ";
  //     //   for (size_t k = 0; k < weight.size(); ++k) {
  //     //     std::cout << weight[k] << " ";
  //     //   }
  //     //   std::cout << std::endl;
  //     // }
  //   }
  //   std::cout << std::endl;
  // }

  // std::cout << "Errors: " << std::endl;
  // for (size_t i = 0; i < layers.size(); ++i) {
  //   std::cout << "Layer " << i << std::endl;
  //   std::vector<s21::Neuron> neurons = layers[i]->GetLayer();
  //   for (size_t l = 0; l < neurons.size(); ++l) {
  //     std::cout << neurons[l].GetError() << std::endl;
  //   }
  //   std::cout << std::endl;
  // }

  s21::Data data;
  data.Parse(
      "/Users/bfile/Projects/SimpleArtificialNeuralNetwork/src/source/dataset/"
      "emnist-letters-train.csv");
  s21::GraphPerceptron test(data, 3);
  test.Train();
}

#endif  // SRC_MODEL_MAIN_H_