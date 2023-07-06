#ifndef SRC_MATRIX_MATRIX_H_
#define SRC_MATRIX_MATRIX_H_

#include <cmath>
#include <iostream>

constexpr double E = 1e-7;

namespace s21 {
class Matrix {
 public:
  Matrix();
  explicit Matrix(int rows, int cols);
  Matrix(int rows, int cols,
         std::initializer_list<std::initializer_list<double>> list);
  Matrix(const Matrix& other);
  Matrix(Matrix&& other);
  ~Matrix();

  unsigned int GetRows() const;
  unsigned int GetCols() const;
  bool EqMatrix(const Matrix& other);
  void SumMatrix(const Matrix& other);
  void MulNumber(const double num);
  void MulMatrix(const Matrix& other);
  Matrix Transpose();
  void SetToZero();
  Matrix operator*(const Matrix& other);
  Matrix operator*(const double num);
  Matrix& operator=(const Matrix& other);
  void operator+=(const Matrix& other);
  void operator*=(const Matrix& other);
  void operator*=(const double num);
  double& operator()(int rows, int cols);
  void Print() const;

 private:
  int rows_ = 0, cols_ = 0;
  double** matrix_ = nullptr;

  void CreateMatrix();
  void DeleteMatrix();
  int EqLoop(const Matrix& other);
};
}  // namespace s21

#endif  // SRC_MATRIX_MATRIX_H_
