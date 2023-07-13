#include "graph_perceptron.h"

s21::GraphPerceptron::GraphPerceptron(Data& data, size_t size)
    : data_(data), size_(size) {
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
  size_t size = layers_.size();
  for (size_t i = 1; i < size; ++i) {
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
  size_t size = layers_.size();
  for (size_t i = 1; i < size; ++i) {
    layers_[i]->FeedForward();
  }
}

void s21::GraphPerceptron::BackPropagation(size_t expected) {
  layers_[layers_.size() - 1]->CalculateOutputError(expected);

  for (int i = layers_.size() - 2; i >= 0; --i) {
    layers_[i]->CalculateError();
  }

  for (size_t i = layers_.size() - 1; i > 0; --i) {
    layers_[i]->UpdateWeights();
  }
}

void s21::GraphPerceptron::Train() {
  FillWeightsRandomly();

  // for (size_t i = 0; i < 4; ++i) {
  for (size_t j = 0; j < data_.GetData().size(); ++j) {
    std::vector<double> input_values = data_.GetData()[j].GetSignals();
    SetInputValues(input_values);
    FeedForward();
    BackPropagation(data_.GetData()[j].GetLetter());
  }
  // }

  int stroka = 56;
  std::vector<double> input_values = data_.GetData()[stroka].GetSignals();
  SetInputValues(input_values);
  FeedForward();

  int letter = data_.GetData()[stroka].GetLetter();
  std::cout << "EXPECTED: " << letter << std::endl;

  std::vector<Neuron> output = layers_[layers_.size() - 1]->GetLayer();

  double max = 0.0;
  int index = 0;
  for (size_t i = 0; i < output.size(); ++i) {
    if (output[i].GetValue() > max) {
      max = output[i].GetValue();
      index = i;
    }
  }

  std::cout << "GET: " << index << " " << max << std::endl;
  std::cout << output[letter].GetValue();
}

void s21::GraphPerceptron::SaveWeight(const std::filesystem::path& filename) {
  std::ofstream file;
  file.open(filename);

  for (auto it = layers_.begin() + 1; it != layers_.end(); ++it) {
    std::vector<Neuron> sloi = (*it)->GetLayer();
    for (auto it1 = sloi.begin(); it1 != sloi.end(); ++it1) {
      std::vector<double> weight = (*it1).GetWeight();
      for (auto it2 = weight.begin(); it2 != weight.end(); ++it2) {
        file << (*it2) << " ";
      }
      file << std::endl;
    }
    file << std::endl;
  }

  file.close();
}

// сохранить веса
// загрузить веса из файла
// получить результат
// трейн
// эксперимент