#ifndef NNDESCENT_ELEMENT_HPP
#define NNDESCENT_ELEMENT_HPP
#include <math.h>

namespace tsdg {
constexpr float EPS = 1e-6;

template <typename Data_t = float, typename Index_t = int>
struct NNDElement {
 private:
  Data_t distance_;
  Index_t label_;

 public:
  NNDElement() { distance_ = 1e10, label_ = 0x3f3f3f3f; }
  NNDElement(Index_t label, Data_t distance, bool is_new = true)
      : distance_(distance), label_(label) {
    if (!is_new) {
      label_ = -label - 1;
    }
  }
  bool IsNew() const { return label_ >= 0; }
  bool IsOld() const { return !IsNew(); }
  void SetLabel(const Index_t new_label) { this->label_ = new_label; }
  void SetDistance(const Data_t new_distance) {
    this->distance_ = new_distance;
  }
  Index_t label() const {
    if (this->IsNew()) return label_;
    return -label_ - 1;
  }
  Data_t distance() const { return distance_; }
  void MarkOld() {
    if (label_ >= 0) label_ = -label_ - 1;
  }
  bool operator<(const NNDElement& other) const {
    if (fabs(this->distance_ - other.distance_) < EPS)
      return this->label() < other.label();
    return this->distance_ < other.distance_;
  }
  bool operator==(const NNDElement& other) const {
    return this->label() == other.label() &&
           (fabs(this->distance_ - other.distance_) < EPS);
  }
  bool operator>=(const NNDElement& other) const { return !(*this < other); }
  bool operator<=(const NNDElement& other) const {
    return (*this == other) || (*this < other);
  }
  bool operator>(const NNDElement& other) const { return !(*this <= other); }
  bool operator!=(const NNDElement& other) const { return !(*this == other); }
};

}  // namespace tsdg
#endif