#ifndef SRC_MODEL_LAYER_H_
#define SRC_MODEL_LAYER_H_

#include <iostream>
#include <random>

#include "neuron.h"

namespace s21 {
class Layer {
 public:
  Layer(size_t size);
  Layer(size_t size, Layer* prev);

  void SetNextLayer();

  //  сайз т гет валью весов по индексу
  // вектор значений сетнейронс валью
  void SetValue(std::vector<double>& values);
  //   вектор сет вейтс бу
  //   а = индекс(индекс, ветор)

  //   паосчитать ошибку
  //   посчитать валью нейронс
  //  калк нейрон хидден
  //   обновить веса
  //   заполнить рандомно
  //    принт

  void FeefForward();
  void CalculateOutputError(size_t expected);
  void CalculateError();
  double ErrorSum(size_t index);

  void UpdateWeights();
  void FillWeightsRandomly();
  std::vector<double> GenerateWeight();
  double RandomGenerator();

  void Print();

  std::vector<Neuron>& GetLayer() { return layer_; }

  size_t GetSizeOfLayer() { return size_; }

  //           гет нейронс
  //           - Ю вектор нейронов кальк еррор рейт(инт)

 private:
  std::vector<Neuron> layer_;
  Layer* prev_layer_;
  Layer* next_layer_;

  size_t size_;
};

}  // namespace s21

#endif  // SRC_MODEL_LAYER_H_
