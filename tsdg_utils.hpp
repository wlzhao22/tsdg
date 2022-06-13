#pragma once
#include <stdint.h>
namespace tsdg {
  
enum Metric {L2, IP};

uint64_t GenerateRandomNumber(uint64_t x) {
  x ^= x >> 12;  // a
  x ^= x << 25;  // b
  x ^= x >> 27;  // c
  return x * 0x2545F4914F6CDD1D;
}

template <typename T>
void IndexWrite(std::ofstream &out_stream, T obj) {
  out_stream.write(reinterpret_cast<const char *>(&obj), sizeof(T));
}

template <typename T>
void IndexWrite(std::ofstream &out_stream, const T *obj,
                const size_t write_num) {
  out_stream.write(reinterpret_cast<const char *>(obj), write_num);
}

template <typename T>
void IndexRead(std::ifstream &in_stream, T *obj, const size_t read_num = 0) {
  if (read_num == 0) {
    in_stream.read(reinterpret_cast<char *>(obj), sizeof(T));
  } else {
    in_stream.read(reinterpret_cast<char *>(obj), read_num);
  }
}
} // namespace tsdg