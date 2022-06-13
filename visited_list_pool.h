#ifndef VISITED_LIST_POOL_H
#define VISITED_LIST_POOL_H
#include <deque>
#include <memory>
#include <mutex>
#include <vector>
typedef int Index_t;
typedef float Data_t;

class VisitedInfo {
 public:
  VisitedInfo(const Index_t max_size_of_list);
  VisitedInfo(const VisitedInfo &other) = delete;
  inline const bool operator[](const Index_t id) const;
  inline void SetVisited(const Index_t id);
  void Reset();
  void Resize(const Index_t size);
  const int GetInfoID() const;

 private:
  static int vl_cnt_;
  int info_id_;
  std::vector<char> if_visited_;
  std::vector<Index_t> visited_nodes_;
};

class VisitedListPool {
 public:
  VisitedListPool();
  VisitedListPool(const Index_t max_size_of_list);
  VisitedInfo &GetAFreeVisitedList();
  void ReleaseVisitedList(VisitedInfo &visited_list);
  void Resize(const int pool_size, const Index_t max_capacity);
  ~VisitedListPool();
  VisitedListPool(const VisitedListPool&) = delete;

//  private:
  std::mutex pool_mutex;
  std::vector<VisitedInfo *> visited_list_pool_;
  int max_capacity_;
};

int VisitedInfo::vl_cnt_ = 0;

VisitedInfo::VisitedInfo(const Index_t max_size_of_list)
    : if_visited_(max_size_of_list) {
  info_id_ = vl_cnt_++;
  return;
}

const int VisitedInfo::GetInfoID() const { return info_id_; }

inline const bool VisitedInfo::operator[](const Index_t id) const {
  return if_visited_[id];
}


inline void VisitedInfo::SetVisited(const Index_t id) {
  visited_nodes_.push_back(id);
  if_visited_[id] = true;
  return;
}

void VisitedInfo::Reset() {
  for (auto node_id : visited_nodes_) {
    if_visited_[node_id] = false;
  }
  visited_nodes_.clear();
  return;
}

void VisitedInfo::Resize(const Index_t size) {
  if_visited_.resize(size);
  visited_nodes_.clear();
  return;
}

VisitedListPool::VisitedListPool() : max_capacity_(0) { return; }

VisitedListPool::VisitedListPool(const Index_t max_size_of_list)
    : max_capacity_(max_size_of_list) {
  return;
}

VisitedInfo& VisitedListPool::GetAFreeVisitedList() {
  std::lock_guard<std::mutex> lock(pool_mutex);
  if (visited_list_pool_.empty()) {
    visited_list_pool_.push_back(new VisitedInfo(max_capacity_));
  }
  auto &vl = *visited_list_pool_.rbegin();
  visited_list_pool_.pop_back();
  return std::ref(*vl);
}

void VisitedListPool::ReleaseVisitedList(VisitedInfo& visited_list) {
  std::lock_guard<std::mutex> lock(pool_mutex);
  visited_list.Reset();
  visited_list_pool_.push_back(&visited_list);
  return;
}

void VisitedListPool::Resize(const int pool_size, const Index_t max_capacity) {
  std::lock_guard<std::mutex> lock(pool_mutex);
  if (visited_list_pool_.size() != pool_size) {
    visited_list_pool_.resize(pool_size);
    for (size_t i = 0; i < pool_size; i++) {
      visited_list_pool_[i] = new VisitedInfo(max_capacity);
    }
  }
  max_capacity_ = max_capacity;
  return;
}

VisitedListPool::~VisitedListPool() {
  for (auto &vl : visited_list_pool_) {
    delete vl;
  }
  return;
}

#endif