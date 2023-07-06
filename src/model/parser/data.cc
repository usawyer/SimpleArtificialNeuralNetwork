#include "data.h"

s21::Symbol::Symbol(const std::string& str) {
  signals_.reserve(784);
  ReadLetter(str);
}

std::vector<s21::Symbol> s21::Data::Parse(const std::string& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    throw std::invalid_argument("No such file!");
  }
  Clear();
  ReadFile(file);
  return data_;
}

void s21::Data::ReadFile(std::ifstream& file) {
  std::string str;
  while (getline(file, str)) {
    data_.push_back(s21::Symbol(str));
  }
  file.close();
}

void s21::Symbol::ReadLetter(const std::string& str) {
  std::vector<std::string> res = SplitLine(str, ',');
  letter_ = std::stoi(res[0]) - 1;
  for (size_t i = 1; i < res.size(); ++i) {
    signals_.push_back(std::stod(res[i]) / 255.0);
  }
}

std::vector<std::string> s21::Symbol::SplitLine(const std::string& str,
                                                char delim) {
  std::stringstream ss(str);
  std::string item;
  std::vector<std::string> elems;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

void s21::Data::Reshuffle() {
  std::random_device random_device;
  std::shuffle(data_.begin(), data_.end(), random_device);
}

// int main() {
//   s21::Data test;
//   test.Parse("/Users/bfile/Projects/CPP7_MLP-2/src/source/dataset/test.csv");

//   std::cout << test.GetData().size() << std::endl;
//   std::cout << std::endl;

//   for (size_t i = 0; i < test.GetData().size(); ++i) {
//     std::cout << test.GetData()[i].GetLetter() << std::endl;
//     std::cout << test.GetData()[i].GetSignals().size() << std::endl;
//   }
//   std::cout << std::endl;

//   for (auto& signals : test.GetData()[0].GetSignals()) {
//     std::cout << signals << " ";
//   }

//   std::cout << test.GetData()[0].GetSignals().size() << std::endl;
// }
