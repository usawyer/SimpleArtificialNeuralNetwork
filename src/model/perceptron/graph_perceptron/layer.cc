#include "layer.h"

s21::Layer::Layer(size_t size) : layer_(size) {
  prev_layer_ = nullptr;
  next_layer_ = nullptr;

  size_ = size;
}

s21::Layer::Layer(size_t size, Layer* prev) : layer_(size) {
  prev_layer_ = prev;
  prev->next_layer_ = this;

  size_ = size;
}

void s21::Layer::SetValue(std::vector<double>& values) {
  for (size_t i = 0; i < layer_.size(); ++i) {
    layer_[i].SetValue(values[i]);
  }
}

void s21::Layer::FillWeightsRandomly() {
  for (size_t i = 0; i < layer_.size(); ++i) {
    layer_[i].SetWeightRandomly(prev_layer_->GetSizeOfLayer());
  }
}

void s21::Layer::FeedForward() {
  std::vector<double> previous_layer_values = GetPrevLayerValues();
  for (size_t i = 0; i < layer_.size(); ++i) {
    layer_[i].CalculateValue(previous_layer_values);
  }
}

void s21::Layer::CalculateOutputError(size_t expected) {
  for (size_t i = 0; i < layer_.size(); ++i) {
    double value = layer_[i].GetValue();
    layer_[i].SetError(-value * (1.0 - value) *
                       (static_cast<double>(i == expected) - value));
  }
}

void s21::Layer::CalculateError() {
  for (size_t i = 0; i < layer_.size(); ++i)
    layer_[i].CalculateError(ErrorSum(i));
}

void s21::Layer::UpdateWeights() {
  std::vector<double> previous_layer_values = GetPrevLayerValues();
  for (size_t i = 0; i < layer_.size(); ++i) {
    layer_[i].UpdateWeight(previous_layer_values);
  }
}

double s21::Layer::ErrorSum(size_t index) {
  double sum = 0.0;
  for (size_t i = 0; i < next_layer_->GetLayer().size(); ++i) {
    sum += next_layer_->GetLayer()[i].GetError() *
           next_layer_->GetLayer()[i].GetWeightByIndex(index);
  }
  return sum;
}

std::vector<double> s21::Layer::GetPrevLayerValues() {
  std::vector<double> res(prev_layer_->GetSizeOfLayer());
  for (size_t i = 0; i < prev_layer_->GetSizeOfLayer(); ++i) {
    res[i] = prev_layer_->GetLayer()[i].GetValue();
  }
  return res;
}
