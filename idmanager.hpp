#ifndef IDMANAGER_H
#define IDMANAGER_H
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <limits>
#include <vector>

template <typename ID_t, typename Label_t>
class IDManager {
 private:
  std::vector<ID_t> label_to_id_;
  std::vector<Label_t> id_to_label_;
  std::mutex manager_mutex_;
  ID_t size_;

 public:
  IDManager();
  bool IsExist(const Label_t &label);
  ID_t ActivateLabel(const Label_t &label);
  void Resize(const ID_t new_size);
  ID_t size();

  ID_t GetID(const Label_t &label);
  inline Label_t GetLabel(const ID_t id);
};

template <typename ID_t, typename Label_t>
IDManager<ID_t, Label_t>::IDManager() {
  size_ = 0;
}

template <typename ID_t, typename Label_t>
void IDManager<ID_t, Label_t>::Resize(const ID_t new_size) {
  label_to_id_.resize(new_size, std::numeric_limits<ID_t>::max());
  id_to_label_.resize(new_size, std::numeric_limits<Label_t>::max());
  size_ = 0;
}

template <typename ID_t, typename Label_t>
bool IDManager<ID_t, Label_t>::IsExist(const Label_t &label) {
  std::lock_guard<std::mutex> lock(manager_mutex_);
  if (label_to_id_[label] != std::numeric_limits<ID_t>::max()) {
    return true;
  }
  return false;
}

template <typename ID_t, typename Label_t>
ID_t IDManager<ID_t, Label_t>::ActivateLabel(const Label_t &label) {
  std::lock_guard<std::mutex> lock(manager_mutex_);
  ID_t id = size_;
  size_++;
  label_to_id_[label] = id;
  id_to_label_[id] = label;
  return id;
}

template <typename ID_t, typename Label_t>
ID_t IDManager<ID_t, Label_t>::size() {
  std::lock_guard<std::mutex> lock(manager_mutex_);
  return size_;
}

template <typename ID_t, typename Label_t>
ID_t IDManager<ID_t, Label_t>::GetID(const Label_t &label) {
  std::lock_guard<std::mutex> lock(manager_mutex_);
  if (label_to_id_[label] != std::numeric_limits<ID_t>::max()) {
    return label_to_id_[label];
  } else {
    throw std::runtime_error("No such label");
  }
}

template <typename ID_t, typename Label_t>
inline Label_t IDManager<ID_t, Label_t>::GetLabel(const ID_t id) {
  return id_to_label_[id];
}


#endif