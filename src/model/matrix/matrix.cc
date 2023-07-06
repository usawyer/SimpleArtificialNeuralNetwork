#include "matrix.h"

s21::Matrix::Matrix() {
  rows_ = 0;
  cols_ = 0;
  matrix_ = NULL;
}

s21::Matrix::Matrix(int rows, int cols) : rows_(rows), cols_(cols) {
  if (rows_ < 1 || cols_ < 1)
    throw std::out_of_range("Incorrect matrix size value");
  CreateMatrix();
}

s21::Matrix::Matrix(int rows, int cols,
                    std::initializer_list<std::initializer_list<double>> list)
    : Matrix(rows, cols) {
  int i = 0, j = 0;
  for (const auto& r : list) {
    for (const auto& c : r) {
      matrix_[i][j] = c;
      ++j;
    }
    j = 0;
    ++i;
  }
}

s21::Matrix::Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_) {
  CreateMatrix();

  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      matrix_[i][j] = other.matrix_[i][j];
    }
  }
}

s21::Matrix::Matrix(Matrix&& other)
    : rows_(other.rows_), cols_(other.cols_), matrix_(other.matrix_) {
  other.cols_ = other.rows_ = 0;
  other.matrix_ = nullptr;
}

s21::Matrix::~Matrix() {
  if (matrix_) {
    DeleteMatrix();
  }
  rows_ = 0;
  cols_ = 0;
}

unsigned int s21::Matrix::GetRows() const { return rows_; }
unsigned int s21::Matrix::GetCols() const { return cols_; }

void s21::Matrix::CreateMatrix() {
  matrix_ = new double*[rows_];
  for (int i = 0; i < rows_; i++) {
    matrix_[i] = new double[cols_];
  }
  SetToZero();
}

void s21::Matrix::SetToZero() {
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      matrix_[i][j] = 0.0;
    }
  }
}

void s21::Matrix::DeleteMatrix() {
  for (int i = 0; i < rows_; i++) {
    delete[] matrix_[i];
  }
  delete[] matrix_;
}

bool s21::Matrix::EqMatrix(const Matrix& other) {
  bool result = true;
  if (cols_ != other.cols_ || rows_ != other.rows_) {
    result = false;
  } else {
    if (EqLoop(other) != cols_ * rows_) {
      result = false;
    }
  }
  return result;
}

int s21::Matrix::EqLoop(const Matrix& other) {
  int match = 0;
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      if (fabs(matrix_[i][j] - other.matrix_[i][j]) < E) {
        match++;
      }
    }
  }
  return match;
}

void s21::Matrix::SumMatrix(const Matrix& other) {
  if (cols_ != other.cols_ || rows_ != other.rows_)
    throw std::out_of_range("Different matrix dimensions!");
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      matrix_[i][j] += other.matrix_[i][j];
    }
  }
}

void s21::Matrix::MulNumber(const double num) {
  if (std::isnan(num) || std::isinf(num))
    throw std::out_of_range("Invalid number!");
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      matrix_[i][j] *= num;
    }
  }
}

void s21::Matrix::MulMatrix(const Matrix& other) {
  if (cols_ != other.rows_)
    throw std::out_of_range("Incorrect matrices for multiplication!");
  Matrix result(rows_, other.cols_);
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < other.cols_; j++) {
      for (int k = 0; k < other.rows_; k++) {
        result.matrix_[i][j] += matrix_[i][k] * other.matrix_[k][j];
      }
    }
  }
  *this = result;
}

s21::Matrix s21::Matrix::Transpose() {
  Matrix result(cols_, rows_);
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      result.matrix_[j][i] = matrix_[i][j];
    }
  }
  return result;
}

s21::Matrix s21::Matrix::operator*(const Matrix& other) {
  Matrix result = *this;
  result.MulMatrix(other);
  return result;
}

s21::Matrix s21::Matrix::operator*(const double num) {
  Matrix result = *this;
  result.MulNumber(num);
  return result;
}

s21::Matrix& s21::Matrix::operator=(const Matrix& other) {
  DeleteMatrix();
  rows_ = other.rows_;
  cols_ = other.cols_;
  CreateMatrix();

  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      matrix_[i][j] = other.matrix_[i][j];
    }
  }
  return *this;
}

void s21::Matrix::operator+=(const Matrix& other) { return SumMatrix(other); }

void s21::Matrix::operator*=(const double num) { return MulNumber(num); }

void s21::Matrix::operator*=(const Matrix& other) { return MulMatrix(other); }

double& s21::Matrix::operator()(int rows, int cols) {
  if (rows_ <= rows || cols_ <= cols || rows < 0 || cols < 0) {
    throw std::out_of_range("Incorrect matrix indices!");
  }
  return matrix_[rows][cols];
}

void s21::Matrix::Print() const {
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      std::cout << matrix_[i][j] << ' ';
    }
    std::cout << '\n';
  }
}
