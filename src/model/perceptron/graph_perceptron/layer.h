#ifndef SRC_MODEL_LAYER_H_
#define SRC_MODEL_LAYER_H_

#include "neuron.h"

namespace s21 {
class Layer {
 public:
  explicit Layer(size_t size);
  Layer(size_t size, Layer* prev);

  Layer(const Layer& other) = default;
  Layer(Layer&& other) = default;
  Layer& operator=(const Layer& other) = default;
  Layer& operator=(Layer&& other) = default;
  ~Layer() = default;

  void SetValue(std::vector<double>& values);
  void FillWeightsRandomly();
  void FeedForward();
  void CalculateOutputError(size_t expected);
  void CalculateError();
  void UpdateWeights();

  std::vector<Neuron>& GetLayer() { return layer_; }
  size_t GetSizeOfLayer() { return size_; }

 private:
  std::vector<Neuron> layer_;
  Layer* prev_layer_;
  Layer* next_layer_;

  size_t size_;

  double ErrorSum(size_t index);
  std::vector<double> GetPrevLayerValues();
};
}  // namespace s21

#endif  // SRC_MODEL_LAYER_H_
