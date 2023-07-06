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

// void s21::Layer::SetNextLayer() {}

void s21::Layer::SetValue(std::vector<double>& values) {
  for (size_t i = 0; i < layer_.size(); ++i) {
    layer_[i].SetValue(values[i]);
  }
}

void s21::Layer::FillWeightsRandomly() {
  for (size_t i = 0; i < layer_.size(); ++i) {
    layer_[i].SetWeight(GenerateWeight());
  }
}

std::vector<double> s21::Layer::GenerateWeight() {
  std::vector<double> res;
  for (size_t i = 0; i < prev_layer_->GetSizeOfLayer(); ++i) {
    res.push_back(RandomGenerator());
  }
  return res;
}

double s21::Layer::RandomGenerator() {
  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_real_distribution<> choise(-1.0, 1.0);
  return choise(generator);
}

void s21::Layer::FeefForward() {
  std::vector<double> previous_layer_values;
  for (size_t i = 0; i < prev_layer_->GetLayer().size(); ++i) {
    previous_layer_values.push_back(prev_layer_->GetLayer()[i].GetValue());
  }
  for (size_t i = 0; i < layer_.size(); ++i) {
    layer_[i].CalculateValue(previous_layer_values);
  }
}

void s21::Layer::CalculateOutputError(size_t expected) {
  for (size_t i = 0; i < layer_.size(); ++i) {
    double value = layer_[i].GetValue();
    layer_[i].SetError(-value * (1 - value) *
                       (static_cast<double>(i == expected) - value));
  }
}

void s21::Layer::CalculateError() {
  for (size_t i = 0; i < layer_.size(); ++i)
    layer_[i].CalculateError(ErrorSum(i));
}

double s21::Layer::ErrorSum(size_t index) {
  double sum = 0.0;
  for (size_t i = 0; i < next_layer_->GetLayer().size(); ++i) {
    sum += next_layer_->GetLayer()[i].GetError() *
           next_layer_->GetLayer()[i].GetWeightByIndex(index);
  }
  return sum;
}
