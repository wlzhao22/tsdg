#ifndef INDEX_MANAGER_H
#define INDEX_MANAGER_H
#include <vector>

namespace tsdg {

template <typename Data_t = float, typename Index_t = int>
class IndexManager {
 private:
  std::vector<char> data_;
  std::vector<Data_t> dists_;
  int index_width_;
  int data_dim_;
  size_t size_per_node_;
  size_t data_off_;

  size_t size_per_node() {
    return sizeof(Index_t) * index_width_ + sizeof(Data_t) * data_dim_;
  }

  size_t data_off() {
    return sizeof(Index_t) * index_width_;
  }

 public:

  IndexManager() {}

  IndexManager(const Index_t index_width, const Data_t data_dim)
      : index_width_(index_width + 1), data_dim_(data_dim) {
    size_per_node_ = size_per_node();
    data_off_ = data_off();
  }

  void Reform(const Index_t index_width, const Data_t data_dim) {
    index_width_ = index_width + 1;
    data_dim_ = data_dim;
    size_per_node_ = size_per_node();
    data_off_ = data_off();
  }

  void Resize(Index_t new_size) {
    data_.resize(size_per_node() * new_size);
    dists_.resize((size_t)index_width_ * new_size);
  }

  Data_t *GetData(size_t id) {
    return reinterpret_cast<Data_t *>(data_.data() + id * size_per_node_ +
                                      data_off_);
  }

  Data_t *GetDists(size_t id) {
    return dists_.data() + id * index_width_;
  }

  Index_t *GetList(size_t id) {
    return reinterpret_cast<Index_t *>(data_.data() + id * size_per_node_) + 1;
  }

  Index_t GetListSize(size_t id) {
    return *reinterpret_cast<Index_t *>(data_.data() + id * size_per_node_);
  }

  void SetListSize(size_t id, int new_size) {
    *reinterpret_cast<Index_t *>(data_.data() + id * size_per_node_) = new_size;
    return;
  }

  void AddData(size_t id, const Data_t *data) {
    memcpy(GetData(id), data, sizeof(Data_t) * data_dim_);
  }

}; // class IndexManager

}  // namespace tsdg

#endif