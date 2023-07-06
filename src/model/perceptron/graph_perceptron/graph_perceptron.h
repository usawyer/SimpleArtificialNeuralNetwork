// константы

// паблик
// войд инит лауерс
// войд тренировка(инт инт) с какой по какую строчку
// войд     установить веса(вектор весов) -
//     конченые
// войд загрузить веса
// войд установить тип нейросети
//     эксперимент(инт6 инт)(возвращает метрики)
//  рандомно заполнить ввеса

//     - вектор<layer*> layer_ - вектор слоев
// - размер скрытых слоев

//     - приват форвард бэк

#ifndef SRC_MODEL_GRAPH_PERCEPTRON_H_
#define SRC_MODEL_GRAPH_PERCEPTRON_H_

#include "../interface_perceptron.h"
#include "layer.h"

namespace s21 {
class GraphPerceptron : InterfacePerceptron {
 public:
  GraphPerceptron(size_t size);
  ~GraphPerceptron() {
    for (auto& layer : layers_) {
      delete layer;
    }
  }

  void InitLayers();
  void SetInputValues(std::vector<double>& input_values);
  void FeefForward();
  void BackPropagation();
  void Train(int from, int to);
  // void SetWeights(std::vector<double> weights);
  void SetWeights(std::vector<std::vector<std::vector<double>>> weights);

  void LoadWeights();
  // Metrics& Experiment(int from, int to);
  void FillWeightsRandomly();

  size_t GetSizeOfHiddenLayers() { return size_; }
  std::vector<Layer*>& GetLayerInfo() { return layers_; }

 private:
  std::vector<Layer*> layers_;
  size_t size_;
};
}  // namespace s21

#endif  // SRC_MODEL_GRAPH_PERCEPTRON_H_s