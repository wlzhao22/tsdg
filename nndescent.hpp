#ifndef NNDESCENT_HPP
#define NNDESCENT_HPP
#include <omp.h>

#include <limits>
#include <map>
#include <mutex>
#include <vector>

#include "index_manager.hpp"
#include "neighbor_elem.h"
#include "nndescent_elem.hpp"
#include "result_elem.cuh"
#include "space_ip.h"
#include "space_l2.h"
#include "tsdg_utils.hpp"

namespace tsdg {
namespace nndescent {

struct BuildParam {
  int graph_k{0};
  int sample_num{16};
  int iteration_num{10};
  bool enable_sc{true};
  Metric metric{L2};
  int threads_num{1};
};

template <typename Data_t = float, typename Index_t = int>
class NNDescent {
  using NNDElem_t = NNDElement<Data_t, Index_t>;
  using Graph_t = std::vector<std::vector<NNDElem_t>>;
  using List_t = std::vector<NNDElem_t>;

 public:
  NNDescent(const Data_t *dataset, const Index_t nrow, const int dim,
            const BuildParam &build_param);
  void SaveIvecs(const std::string &path);
  void Build();
  const Graph_t &graph(){return graph_;}
  NNDescent(const NNDescent &) = delete;
  NNDescent operator=(const NNDescent &) = delete;

 private:
  void Init(const BuildParam &build_param);
  int InsertToOrderedList(const Index_t list_id, NNDElem_t nb,
                          const int list_capacity);

  Graph_t graph_;
  Graph_t graph_new_;
  Graph_t graph_old_;
  void SampleGraph();
  void LocalJoin(const Index_t list_id);
  void LocalJoin2(const Index_t list_id);
  const Data_t *dataset_;
  Index_t nrow_;

  mutable std::mutex global_mutex_;
  mutable std::vector<std::mutex> lists_mutex_;

  BuildParam build_param_;
  std::string metric_name_;
  std::unique_ptr<SpaceInterface<Data_t>> metric_;
  DistFunc_t<Data_t> DistFunc_;
  const void *dist_func_param_;
  int dim_;
};

template <typename Data_t, typename Index_t>
void NNDescent<Data_t, Index_t>::Init(const BuildParam &build_param) {
  assert(dim_);
  if (build_param.metric == Metric::L2) {
    if constexpr (std::is_same<Data_t, float>::value) {
      metric_ = std::make_unique<L2Space>(dim_);
    } else if constexpr (std::is_same<Data_t, int>::value) {
      metric_ = std::make_unique<L2SpaceI>(dim_);
    } else {
      throw std::runtime_error("Unsupported data type.");
    }
  } else if (build_param.metric == Metric::IP) {
    if constexpr (std::is_same<Data_t, float>::value) {
      metric_ = std::make_unique<InnerProductSpace>(dim_);
    } else {
      throw std::runtime_error("Unsupported data type.");
    }
  } else {
    throw std::runtime_error("Unsupported metric.");
  }
  build_param_ = build_param;
  DistFunc_ = (*metric_).get_dist_func();
  dist_func_param_ = (*metric_).get_dist_func_param();
  metric_name_ = build_param.metric;
}

template <typename Data_t, typename Index_t>
NNDescent<Data_t, Index_t>::NNDescent(const Data_t *dataset, const Index_t nrow,
                                      const int dim,
                                      const BuildParam &build_param)
    : dim_(dim) {
  build_param_ = build_param;
  assert(build_param_.graph_k != 0);
  Init(build_param_);
  dataset_ = dataset;
  nrow_ = nrow;
  graph_.resize(nrow, List_t(build_param_.graph_k));
  std::vector<std::mutex>(nrow).swap(lists_mutex_);

  std::vector<int> random_array(graph_.size());
  for (size_t i = 0; i < random_array.size(); i++) {
    random_array[i] = i;
  }
  std::random_shuffle(random_array.begin(), random_array.end());
  size_t pos = 0;

#pragma omp parallel for num_threads(build_param_.threads_num)
  for (size_t i = 0; i < nrow; i++) {
    for (int j = 0; j < build_param_.graph_k; j++) {
      while (random_array[pos] == i) {
        pos = (pos + 1) % random_array.size();
      }
      graph_[i][j].SetLabel(random_array[pos]);
      pos = (pos + 1) % random_array.size();
      if (graph_[i][j].label() == i) {
        graph_[i][j].SetLabel((graph_[i][j].label() + 1) % nrow);
      }
    }
  }

#pragma omp parallel for num_threads(build_param_.threads_num)
  for (size_t i = 0; i < nrow; i++) {
    for (size_t j = 0; j < graph_[i].size(); j++) {
      size_t nb = graph_[i][j].label();
      Data_t dist =
          DistFunc_(dataset + i * dim, dataset + nb * dim, dist_func_param_);
      graph_[i][j].SetDistance(dist);
    }
    std::sort(graph_[i].begin(), graph_[i].end());
  }
}

template <typename Data_t, typename Index_t>
void NNDescent<Data_t, Index_t>::SaveIvecs(const std::string &path) {
  std::ofstream out(path, std::ios::binary);
  for (size_t i = 0; i < graph_.size(); i++) {
    IndexWrite(out, (Index_t)graph_[i].size());
    assert(graph_[i].size() == build_param_.graph_k);
    for (size_t j = 0; j < build_param_.graph_k; j++) {
      IndexWrite(out, graph_[i][j].label());
    }
  }
}

template <typename Data_t, typename Index_t>
void NNDescent<Data_t, Index_t>::SampleGraph() {
  graph_new_ = Graph_t(graph_.size());
  graph_old_ = Graph_t(graph_.size());

#pragma omp parallel for num_threads(build_param_.threads_num)
  for (size_t i = 0; i < graph_.size(); i++) {
    int cnt_new = 0;
    for (int j = 0; j < graph_[i].size(); j++) {
      if (graph_[i][j].IsNew()) {
        if (cnt_new < build_param_.sample_num) {
          graph_new_[i].push_back(graph_[i][j]);
          graph_[i][j].MarkOld();
          cnt_new++;
        }
      } else {
        graph_old_[i].push_back(graph_[i][j]);
      }
    }
  }

  Graph_t rgraph_new(graph_new_.size());
  Graph_t rgraph_old(graph_old_.size());
  for (size_t i = 0; i < graph_new_.size(); i++) {
    for (auto x : graph_new_[i]) {
      if (rgraph_new[x.label()].size() < build_param_.sample_num) {
        rgraph_new[x.label()].emplace_back(i, x.distance());
      }
    }
    for (auto x : graph_old_[i]) {
      if (rgraph_old[x.label()].size() < build_param_.sample_num) {
        rgraph_old[x.label()].emplace_back(i, x.distance());
      }
    }
  }

#pragma omp parallel for num_threads(build_param_.threads_num)
  for (size_t i = 0; i < graph_new_.size(); i++) {
    for (size_t j = 0; j < rgraph_new[i].size() && j < build_param_.sample_num;
         j++) {
      graph_new_[i].push_back(rgraph_new[i][j]);
    }
    std::sort(graph_new_[i].begin(), graph_new_[i].end());
    graph_new_[i].erase(unique(graph_new_[i].begin(), graph_new_[i].end()),
                        graph_new_[i].end());

    for (size_t j = 0; j < rgraph_old[i].size() && j < build_param_.sample_num;
         j++) {
      graph_old_[i].push_back(rgraph_old[i][j]);
    }
    std::sort(graph_old_[i].begin(), graph_old_[i].end());
    graph_old_[i].erase(unique(graph_old_[i].begin(), graph_old_[i].end()),
                        graph_old_[i].end());
  }
}

template <typename Data_t, typename Index_t>
int NNDescent<Data_t, Index_t>::InsertToOrderedList(const Index_t list_id,
                                                    NNDElem_t nb,
                                                    const int list_capacity) {
  std::lock_guard<std::mutex> lock(lists_mutex_[list_id]);
  if (list_id == nb.label()) {
    return list_capacity;
  }
  auto &list = graph_[list_id];
  if (list.size() == list_capacity &&
      nb.distance() >= (*list.rbegin()).distance()) {
    return list_capacity;
  }
  int it_to_expand = list.size();
  for (size_t i = 0; i < list.size(); i++) {
    if (list[i].label() == nb.label()) {
      return list_capacity;
    }
    if (list[i] > nb) {
      it_to_expand = i;
      break;
    }
  }
  if (list.size() < list_capacity) {
    list.resize(list.size() + 1);
  }
  for (size_t i = list.size() - 1; i > it_to_expand; i--) {
    list[i] = list[i - 1];
  }
  list[it_to_expand] = nb;
  return it_to_expand;
};

template <typename Data_t, typename Index_t>
void NNDescent<Data_t, Index_t>::LocalJoin(const Index_t list_id) {
  for (size_t i = 0; i < graph_new_[list_id].size(); i++) {
    size_t u_id = graph_new_[list_id][i].label();
    for (size_t j = i + 1; j < graph_new_[list_id].size(); j++) {
      size_t v_id = graph_new_[list_id][j].label();
      Data_t dist = DistFunc_(dataset_ + u_id * dim_, dataset_ + v_id * dim_,
                              dist_func_param_);
      InsertToOrderedList(u_id, NNDElem_t(v_id, dist),
                          build_param_.graph_k);
      InsertToOrderedList(v_id, NNDElem_t(u_id, dist),
                          build_param_.graph_k);
    }
    for (size_t j = 0; j < graph_old_[list_id].size(); j++) {
      size_t v_id = graph_old_[list_id][j].label();
      Data_t dist = DistFunc_(dataset_ + u_id * dim_, dataset_ + v_id * dim_,
                              dist_func_param_);
      InsertToOrderedList(u_id, NNDElem_t(v_id, dist),
                          build_param_.graph_k);
      InsertToOrderedList(v_id, NNDElem_t(u_id, dist),
                          build_param_.graph_k);
    }
  }
}

template <typename Data_t, typename Index_t>
void NNDescent<Data_t, Index_t>::LocalJoin2(const Index_t list_id) {
  std::vector<NNDElem_t> result_new_cache(graph_new_[list_id].size());
  std::vector<NNDElem_t> result_old_cache(graph_old_[list_id].size());

  for (size_t i = 0; i < graph_new_[list_id].size(); i++) {
    size_t u_id = graph_new_[list_id][i].label();
    for (size_t j = i + 1; j < graph_new_[list_id].size(); j++) {
      size_t v_id = graph_new_[list_id][j].label();
      Data_t dist = DistFunc_(dataset_ + u_id * dim_, dataset_ + v_id * dim_,
                              dist_func_param_);
      result_new_cache[i] =
          std::min(result_new_cache[i], NNDElem_t(v_id, dist));
      result_new_cache[j] =
          std::min(result_new_cache[j], NNDElem_t(u_id, dist));
    }
  }

  for (size_t i = 0; i < graph_new_[list_id].size(); i++) {
    size_t id = graph_new_[list_id][i].label();
    InsertToOrderedList(id, result_new_cache[i], build_param_.graph_k);
    result_new_cache[i] = NNDElem_t();
  }

  for (size_t i = 0; i < graph_new_[list_id].size(); i++) {
    size_t u_id = graph_new_[list_id][i].label();
    for (size_t j = 0; j < graph_old_[list_id].size(); j++) {
      size_t v_id = graph_old_[list_id][j].label();
      Data_t dist = DistFunc_(dataset_ + u_id * dim_, dataset_ + v_id * dim_,
                              dist_func_param_);
      result_new_cache[i] =
          std::min(result_new_cache[i], NNDElem_t(v_id, dist));
      result_old_cache[j] =
          std::min(result_old_cache[j], NNDElem_t(u_id, dist));
    }
  }

  for (size_t i = 0; i < graph_new_[list_id].size(); i++) {
    size_t id = graph_new_[list_id][i].label();
    InsertToOrderedList(id, result_new_cache[i], build_param_.graph_k);
  }
  for (size_t i = 0; i < graph_old_[list_id].size(); i++) {
    size_t id = graph_old_[list_id][i].label();
    InsertToOrderedList(id, result_old_cache[i], build_param_.graph_k);
  }
}

template <typename Data_t, typename Index_t>
void NNDescent<Data_t, Index_t>::Build() {
  for (int it = 0; it < build_param_.iteration_num; it++) {
    SampleGraph();
    std::cout << "Iteration: " << it << std::endl;
#pragma omp parallel for num_threads(build_param_.threads_num)
    for (size_t i = 0; i < graph_.size(); i++) {
      if (build_param_.enable_sc) {
        LocalJoin2(i);
      } else {
        LocalJoin(i);
      }
    }
  }
}

}  // namespace nndescent
}  // namespace tsdg

#endif
