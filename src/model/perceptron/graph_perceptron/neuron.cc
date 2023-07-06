#include "neuron.h"

void s21::Neuron::CalculateValue(std::vector<double>& previous_layer_values) {
  double raw_value = 0.0;
  for (size_t i = 0; i < weights_.size(); ++i) {
    raw_value += previous_layer_values[i] * weights_[i];
  }
  value_ = 1.0 / (1.0 + std::exp(-raw_value));
}

void s21::Neuron::CalculateError(double sum) {
  error_ = value_ * (1 - value_) * sum;
}
