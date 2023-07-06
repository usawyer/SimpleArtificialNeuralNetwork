#ifndef SRC_MODEL_DATA_H_
#define SRC_MODEL_DATA_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

namespace s21 {
// const int kInputLayerSize = 784;

class Symbol {
 public:
  Symbol(const Symbol& other) = default;
  Symbol(Symbol&& other) = default;
  Symbol& operator=(const Symbol& other) = default;
  Symbol& operator=(Symbol&& other) = default;
  ~Symbol() = default;

  explicit Symbol(const std::string& str);

  size_t GetLetter() { return letter_; }
  std::vector<double> GetSignals() { return signals_; }

 private:
  std::vector<std::string> SplitLine(const std::string& str, char delim);
  void ReadLetter(const std::string& str);

  std::vector<double> signals_;
  size_t letter_;
};

class Data {
 public:
  std::vector<Symbol> Parse(const std::string& file_path);
  void Reshuffle();

  std::vector<Symbol> GetData() { return data_; }

 private:
  void ReadFile(std::ifstream& file);
  void Clear() { data_.clear(); }

  std::vector<Symbol> data_;
};
}  // namespace s21

#endif  // SRC_MODEL_DATA_H_
