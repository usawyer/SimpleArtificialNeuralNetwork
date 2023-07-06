#include "neuron.h"

// double Sigmoid_dx(double x) {
//   double sigmoid = Sigmoid(x);
//   return sigmoid * (1.0 - sigmoid);
// }

// double Error(double actual, double expected) { return actual - expected; }

// double WeightDelta(double error, double x) { return error * Sigmoid_dx(x); }

// int main() {
//   std::cout << Sigmoid(0.79) << std::endl;
//   std::cout << Sigmoid_dx(0.79) << std::endl;
//   std::cout << WeightDelta(0.69, 0.79) << std::endl;
// }

void s21::Neuron::CalculateValue(std::vector<double>& previous_layer_values) {
  double raw_value = 0.0;
  for (size_t i = 0; i < weights_.size(); ++i) {
    raw_value += previous_layer_values[i] * weights_[i];
  }
  raw_value_ = raw_value;
  value_ = 1.0 / (1.0 + std::exp(-raw_value));
}

// void s21::Neuron::CalculateError(double value) {}