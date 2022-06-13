#ifndef NEIGHB_ELEMENT_CUH
#define NEIGHB_ELEMENT_CUH

#include <cmath>
#include <limits>

namespace olgraph {
constexpr float EPS = 1e-6;
enum ElemType { Max };
template <typename Dist_t = float, typename Lable_t = int>
struct ResultElem {
  Dist_t distance_;
  Lable_t label_;

  ResultElem(){};

  ResultElem(ElemType elem_type) {
    if (elem_type == ElemType::Max) {
      distance_ = 1e10;
      label_ = std::numeric_limits<Lable_t>::max();
    } else {
      distance_ = 0;
      label_ = 0;
    }
  }
  ResultElem(Lable_t label, Dist_t distance)
      : distance_(distance), label_(label) {}
  void SetLabel(const Lable_t new_label) { label_ = new_label; }
  void SetDistance(const Dist_t new_distance) { distance_ = new_distance; }
  Lable_t label() const { return label_; }
  Dist_t distance() const { return distance_; }
  bool IsInfinity() { return label_ == std::numeric_limits<Lable_t>::max(); }
  bool operator<(const ResultElem& other) const {
    if (distance_ == other.distance_) return label() < other.label();
    return distance_ < other.distance_;
  }
  bool operator==(const ResultElem& other) const {
    return label() == other.label() &&
           (fabs(distance_ - other.distance_) < EPS);
  }
  bool operator>=(const ResultElem& other) const { return !(*this < other); }
  bool operator<=(const ResultElem& other) const {
    return (*this == other) || (*this < other);
  }
  bool operator>(const ResultElem& other) const { return !(*this <= other); }
  bool operator!=(const ResultElem& other) const { return !(*this == other); }
};
}  // namespace cugann
#endif