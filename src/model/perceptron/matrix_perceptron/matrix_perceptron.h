#ifndef SRC_MODEL_MATRIX_PERCEPTRON_H_
#define SRC_MODEL_MATRIX_PERCEPTRON_H_

#include "../../matrix/matrix.h"
#include "../interface_perceptron.h"

namespace s21 {
class MatrixPerceptron {
 public:
 private:
  Matrix value_;
  Matrix error_;
  Matrix weights_;
};
}  // namespace s21

#endif  // SRC_MODEL_MATRIX_PERCEPTRON_H_
