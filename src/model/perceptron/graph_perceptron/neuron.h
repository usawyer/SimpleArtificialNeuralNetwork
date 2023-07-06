#ifndef SRC_MODEL_NEURON_H_
#define SRC_MODEL_NEURON_H_

#include <cmath>
#include <vector>

namespace s21 {
class Neuron {
 public:
  void CalculateValue(std::vector<double>& previous_layer_values);
  void CalculateError(double sum);

  double GetWeightByIndex(size_t index) { return weights_[index]; }
  std::vector<double> GetWeight() { return weights_; }
  double GetValue() { return value_; }
  double GetError() { return error_; }

  void SetValue(double value) { value_ = value; }
  void SetError(double error) { error_ = error; }
  void SetWeight(std::vector<double> weights) { weights_ = weights; }

 private:
  double value_ = 0;
  double error_ = 0;

  std::vector<double> weights_;
  std::vector<double> delta_weights_;
};
}  // namespace s21

#endif  // SRC_MODEL_NEURON_H_
