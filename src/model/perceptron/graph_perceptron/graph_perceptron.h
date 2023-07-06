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
  void Calculate();
  void Train(int from, int to);
  void SetWeights(std::vector<double> weights);
  void LoadWeights();
  // Metrics& Experiment(int from, int to);
  void FillWeightsRandomly();

  size_t GetSizeOfHiddenLayers() { return size_; }
  std::vector<Layer*>& GetLayerInfo() { return layers_; }

  std::vector<double> in_weight_1{0.9, 0.3, 0.4};
  std::vector<double> in_weight_2{0.2, 0.8, 0.2};
  std::vector<double> in_weight_3{0.1, 0.5, 0.6};

  std::vector<double> out_weight_1{0.3, 0.7, 0.5};
  std::vector<double> out_weight_2{0.6, 0.5, 0.2};
  std::vector<double> out_weight_3{0.8, 0.1, 0.9};

 private:
  std::vector<Layer*> layers_;
  size_t size_;
};
}  // namespace s21

#endif  // SRC_MODEL_GRAPH_PERCEPTRON_H_s