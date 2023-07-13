#include "neuron.h"

void s21::Neuron::SetWeightRandomly(size_t size) {
  weights_.resize(size);
  delta_weights_.resize(size);

  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_real_distribution<> choise(-0.5, 0.5);

  std::generate(weights_.begin(), weights_.end(),
                [&]() { return choise(generator); });
}

void s21::Neuron::SetWeight(std::vector<double>& weights) {
  weights_.resize(weights.size());
  delta_weights_.resize(weights.size());

  std::copy(weights.begin(), weights.end(), weights_.begin());
}

void s21::Neuron::CalculateValue(std::vector<double>& previous_layer_values) {
  double raw_value = 0.0;
  size_t size = weights_.size();
  for (size_t i = 0; i < size; ++i) {
    raw_value += previous_layer_values[i] * weights_[i];
  }
  value_ = 1.0 / (1.0 + std::exp(-raw_value));
}

void s21::Neuron::CalculateError(double sum) {
  error_ = value_ * (1.0 - value_) * sum;
}

void s21::Neuron::UpdateWeight(std::vector<double>& previous_layer_values) {
  size_t size = previous_layer_values.size();
  for (size_t i = 0; i < size; ++i) {
    double delta = kLearningRate * delta_weights_[i] +
                   (1.0 - kLearningRate) * kLearningRate * error_ *
                       previous_layer_values[i];
    weights_[i] -= delta;
    delta_weights_[i] = delta;
  }
}
