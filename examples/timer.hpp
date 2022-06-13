#ifndef TIMER_HPP
#define TIMER_HPP
#include <chrono>
class Timer {
  std::chrono::_V2::steady_clock::time_point start_;

 public:
  void start() { start_ = std::chrono::steady_clock::now(); };
  float end() {
    auto end = std::chrono::steady_clock::now();
    float tmp_time =
        (float)std::chrono::duration_cast<std::chrono::microseconds>(end - start_)
            .count() /
        1e6;
    return tmp_time;
  }
};
#endif