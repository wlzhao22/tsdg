#ifndef TSDG_HPP
#define TSDG_HPP
#include <omp.h>

#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <unordered_set>
#include <vector>

#include "idmanager.hpp"
#include "index_manager.hpp"
#include "neighbor_elem.h"
#include "result_elem.cuh"
#include "space_ip.h"
#include "space_l2.h"
#include "tsdg_utils.hpp"
#include "visited_list_pool.h"

namespace tsdg {

struct BuildParam {
  int max_edge_num{60};
  float relax_factor{1.2};
  int occl_threshold{4};
  Metric metric{L2};
  int threads_num{1};
};

template <typename Data_t = float, typename Index_t = int>
class TSDG {
 public:
  TSDG(const BuildParam &build_param, const int data_dim);
  TSDG(const std::string &index_file);
  void Add(const Data_t *data, const Index_t label);
  void Build(const Data_t *data, const Index_t *base_graph, const int graph_k,
             const Index_t nrow);
  void Search(const Data_t *query, const int top_k, const int ef,
              Index_t *index_result, Data_t *dist_result);
  void Resize(Index_t new_size);
  void Save(const std::string &path, const int version = 0);
  void Load(const std::string &path);
  size_t GetSize() { return insider_.size(); };
  size_t GetMaxCapacity() { return (size_t)max_capacity_; };
  TSDG(const TSDG &) = delete;
  TSDG operator=(const TSDG &) = delete;

 private:
  IDManager<Index_t, Index_t> id_manager_;
  using Graph_t = std::vector<std::vector<Index_t>>;
  std::vector<std::vector<int>> occl_factors_;
  std::vector<std::vector<bool>> is_reverse_edge_;

  void Init(const BuildParam &build_param, const int data_dim);
  void RelaxedGraphDiversify(Graph_t &graph, const float relax_factor,
                             const int max_edge_num = -1);
  void AddReverseEdges(Graph_t &graph, const int max_edge_num = -1);
  void RankedGraphDiversify(Graph_t &graph, const int rank_threshold,
                            const int max_edge_num = -1);
  void AddNewNode(const Index_t id, const Data_t *vec);
  int CheckGraph(const Graph_t &graph);
  Index_t max_capacity_;
  IndexManager<Data_t, Index_t> index_manager_;

  mutable VisitedListPool visited_list_pool_;
  mutable std::mutex global_mutex_;
  mutable std::vector<std::mutex> lists_mutex_;

  BuildParam build_param_;
  std::string metric_name_;
  std::unique_ptr<SpaceInterface<Data_t>> metric_;
  DistFunc_t<Data_t> DistFunc_;
  std::unordered_set<Index_t> insider_;
  const void *dist_func_param_;
  int dim_;
};

template <typename Data_t, typename Index_t>
void TSDG<Data_t, Index_t>::Init(const BuildParam &build_param,
                                 const int data_dim) {
  dim_ = data_dim;
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
TSDG<Data_t, Index_t>::TSDG(const BuildParam &build_param, const int data_dim) {
  dim_ = data_dim;
  index_manager_ =
      IndexManager<Data_t, Index_t>(build_param.max_edge_num, data_dim);
  Init(build_param, data_dim);
}

template <typename Data_t, typename Index_t>
TSDG<Data_t, Index_t>::TSDG(const std::string &index_file) {
  Load(index_file);
}

template <typename Data_t, typename Index_t>
void TSDG<Data_t, Index_t>::Save(const std::string &path, const int version) {
  std::ofstream out(path, std::ios::binary);
  IndexWrite(out, version);
  IndexWrite(out, insider_.size());
  IndexWrite(out, dim_);
  IndexWrite(out, build_param_.max_edge_num);
  IndexWrite(out, build_param_.relax_factor);
  IndexWrite(out, build_param_.occl_threshold);
  IndexWrite(out, (int)build_param_.metric);
  assert(build_param_.threads_num);
  IndexWrite(out, build_param_.threads_num);

  for (size_t i = 0; i < insider_.size(); i++) {
    const auto vec = index_manager_.GetData(i);
    IndexWrite(out, vec, sizeof(Data_t) * dim_);
  }

  for (Index_t i = 0; i < insider_.size(); i++) {
    const auto list = index_manager_.GetList(i);
    const int list_size = index_manager_.GetListSize(i);

    IndexWrite(out, list_size);
    IndexWrite(out, list, sizeof(Index_t) * list_size);
  }
}

template <typename Data_t, typename Index_t>
void TSDG<Data_t, Index_t>::Load(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  int version;
  IndexRead(in, &version);
  size_t nrow;
  IndexRead(in, &nrow);
  assert(nrow <= std::numeric_limits<Index_t>::max());
  max_capacity_ = (Index_t)nrow;
  IndexRead(in, &dim_);
  IndexRead(in, &build_param_.max_edge_num);
  IndexRead(in, &build_param_.relax_factor);
  IndexRead(in, &build_param_.occl_threshold);
  int metric;
  IndexRead(in, &metric);
  build_param_.metric = Metric(metric);
  IndexRead(in, &build_param_.threads_num);

  index_manager_ =
      IndexManager<Data_t, Index_t>(build_param_.max_edge_num, dim_);

  Init(build_param_, dim_);
  Resize(max_capacity_);

  std::vector<Data_t> tmp_vec(dim_);
  for (Index_t i = 0; i < max_capacity_; i++) {
    IndexRead(in, tmp_vec.data(), sizeof(Data_t) * dim_);
    Index_t internal_id = id_manager_.ActivateLabel(i);
    AddNewNode(internal_id, tmp_vec.data());
  }

  for (Index_t i = 0; i < insider_.size(); i++) {
    Index_t internal_id = i;
    const auto list = index_manager_.GetList(internal_id);
    int list_size;
    IndexRead(in, &list_size);
    index_manager_.SetListSize(internal_id, list_size);
    IndexRead(in, list, sizeof(Index_t) * list_size);
  }
}

template <typename Data_t, typename Index_t>
void TSDG<Data_t, Index_t>::Search(const Data_t *query, const int top_k,
                                   const int ef, Index_t *index_result,
                                   Data_t *dist_result) {
  std::priority_queue<std::pair<Data_t, Index_t>> result_pq;
  std::priority_queue<std::pair<Data_t, Index_t>,
                      std::vector<std::pair<Data_t, Index_t>>,
                      std::greater<std::pair<Data_t, Index_t>>>
      candidates;
  std::vector<Index_t> start_points;
  VisitedInfo &visited_info = visited_list_pool_.GetAFreeVisitedList();

  for (int i = 0; i < 32; i++) {
    Index_t seed = GenerateRandomNumber(i) % (insider_.size());
    start_points.push_back(seed);
  }
  _mm_prefetch(index_manager_.GetData(start_points[0]), _MM_HINT_T0);
  for (size_t i = 0; i < start_points.size(); i++) {
    Index_t p = start_points[i];
    _mm_prefetch(index_manager_.GetData(start_points[i + 1]), _MM_HINT_T0);
    if (visited_info[p]) continue;
    auto dist = DistFunc_(query, index_manager_.GetData(p), dist_func_param_);
    candidates.emplace(dist, p);
    result_pq.emplace(dist, p);
    while (result_pq.size() > ef) {
      result_pq.pop();
    }
    visited_info.SetVisited(p);
  }
  while (!candidates.empty()) {
    Data_t min_dist = candidates.top().first;
    Index_t best_c = candidates.top().second;
    candidates.pop();
    if (result_pq.size() == ef) {
      if (min_dist > result_pq.top().first) {
        break;
      }
    }

    const auto *list = index_manager_.GetList(best_c);
    const int list_size = index_manager_.GetListSize(best_c);

    _mm_prefetch(index_manager_.GetData(list[0]), _MM_HINT_T0);
    _mm_prefetch(list, _MM_HINT_T0);

    // std::lock_guard<std::mutex> lock(lists_mutex_[list[0]]);
    for (size_t i = 0; i < list_size; i++) {
      _mm_prefetch(index_manager_.GetData(list[i + 1]), _MM_HINT_T0);
      size_t nb_id = list[i];
      if (visited_info[nb_id]) {
        continue;
      }
      auto dist =
          DistFunc_(query, index_manager_.GetData(nb_id), dist_func_param_);
      visited_info.SetVisited(nb_id);
      if (result_pq.size() >= ef && dist >= result_pq.top().first) {
        continue;
      }
      _mm_prefetch(index_manager_.GetList(candidates.top().second),
                   _MM_HINT_T0);
      candidates.emplace(dist, nb_id);
      result_pq.emplace(dist, nb_id);
      while (result_pq.size() > ef) {
        result_pq.pop();
      }
    }
  }
  std::vector<std::pair<Data_t, Index_t>> result;
  int i = 0;
  while (!result_pq.empty()) {
    result.push_back(result_pq.top());
    result_pq.pop();
  }
  std::reverse(result.begin(), result.end());
  for (int i = 0; i < top_k && i < result.size(); i++) {
    dist_result[i] = result[i].first;
    index_result[i] = id_manager_.GetLabel(result[i].second);
  }

  visited_list_pool_.ReleaseVisitedList(visited_info);
  return;
}

template <typename Data_t, typename Index_t>
void TSDG<Data_t, Index_t>::Resize(Index_t new_size) {
  max_capacity_ = new_size;
  visited_list_pool_.Resize(build_param_.threads_num, new_size);
  index_manager_.Resize(new_size);
  id_manager_.Resize(new_size);
  occl_factors_.resize(new_size);
  is_reverse_edge_.resize(new_size);
  std::vector<std::mutex>(new_size).swap(lists_mutex_);
}

template <typename Data_t, typename Index_t>
void TSDG<Data_t, Index_t>::Add(const Data_t *data, const Index_t label) {
  // to-do
}

template <typename Data_t, typename Index_t>
void TSDG<Data_t, Index_t>::AddNewNode(const Index_t id, const Data_t *vec) {
  insider_.insert(id);
  index_manager_.AddData(id, vec);
}

template <typename Data_t, typename Index_t>
void TSDG<Data_t, Index_t>::RelaxedGraphDiversify(Graph_t &graph,
                                                  const float relax_factor,
                                                  const int max_edge_num) {
#pragma omp parallel for num_threads(build_param_.threads_num)
  for (size_t i = 0; i < graph.size(); i++) {
    auto list_tmp = graph[i];
    auto list_size = graph[i].size();
    std::vector<Data_t> all_dist(list_size);
    for (size_t j = 0; j < list_size; j++) {
      all_dist[j] =
          DistFunc_(index_manager_.GetData(i),
                    index_manager_.GetData(list_tmp[j]), dist_func_param_);
    }
    std::vector<bool> removed(list_size);
    std::vector<Index_t> new_list;
    std::vector<int> occl_tmp(list_size);

    for (size_t j = 0; j < list_size; j++) {
      if (removed[j]) continue;
      new_list.push_back(list_tmp[j]);
      occl_factors_[i].push_back(occl_tmp[j]);
      for (size_t k = j + 1; k < list_size; k++) {
        if (removed[k]) continue;
        Index_t nb1 = list_tmp[j];
        Index_t nb2 = list_tmp[k];
        Data_t dist2 = all_dist[k];
        Data_t dist3 = DistFunc_(index_manager_.GetData(nb1),
                                 index_manager_.GetData(nb2), dist_func_param_);
        if (dist2 > relax_factor * dist3) {
          removed[k] = 1;
        } else if (dist2 > dist3) {
          occl_tmp[k]++;
        }
      }
    }
    if (max_edge_num != -1) {
      if (new_list.size() > max_edge_num) {
        new_list.erase(new_list.begin() + max_edge_num, new_list.end());
      }
    }
    graph[i] = new_list;
  }
  return;
}

template <typename Data_t, typename Index_t>
void TSDG<Data_t, Index_t>::AddReverseEdges(Graph_t &graph,
                                            const int max_edge_num) {
  constexpr int MAX_EDGE_NUM = 2048;
  Graph_t rgraph(graph.size());
  for (size_t i = 0; i < graph.size(); i++) {
    for (auto x : graph[i]) {
      if (graph[x].size() + rgraph[x].size() < MAX_EDGE_NUM) {
        rgraph[x].push_back(i);
      }
    }
  }

#pragma omp parallel for num_threads(build_param_.threads_num)
  for (size_t i = 0; i < graph.size(); i++) {
    size_t orig_size = graph[i].size();
    for (size_t j = 0; j < rgraph[i].size(); j++) {
      bool is_existed = false;
      for (size_t k = 0; k < orig_size; k++) {
        if (graph[i][k] == rgraph[i][j]) {
          is_existed = true;
          break;
        }
      }
      if (!is_existed) {
        graph[i].push_back(rgraph[i][j]);
      }
    }
  }
  Graph_t().swap(rgraph);

#pragma omp parallel for num_threads(build_param_.threads_num)
  for (size_t i = 0; i < graph.size(); i++) {
    using TempNB = std::pair<std::pair<Data_t, Index_t>, int>;
    std::vector<TempNB> nn_list;
    for (size_t j = 0; j < graph[i].size(); j++) {
      Index_t nb = graph[i][j];
      Data_t dist = DistFunc_(index_manager_.GetData(i),
                              index_manager_.GetData(nb), dist_func_param_);
      int occl_factor = j < occl_factors_[i].size() ? occl_factors_[i][j] : -1;
      nn_list.emplace_back(std::make_pair(dist, nb), occl_factor);
    }
    std::sort(nn_list.begin(), nn_list.end());

    if (max_edge_num != -1) {
      if (nn_list.size() > max_edge_num) {
        nn_list.erase(nn_list.begin() + max_edge_num, nn_list.end());
        graph[i].resize(max_edge_num);
      }
    }
    for (size_t j = 0; j < nn_list.size(); j++) {
      graph[i][j] = nn_list[j].first.second;
    }
    occl_factors_[i].resize(nn_list.size());
    is_reverse_edge_[i].resize(nn_list.size());

    for (size_t j = 0; j < nn_list.size(); j++) {
      if (nn_list[j].second == -1) {
        occl_factors_[i][j] = 0;
        is_reverse_edge_[i][j] = true;
      } else {
        occl_factors_[i][j] = nn_list[j].second;
      }
    }
  }
  return;
}

template <typename Data_t, typename Index_t>
void TSDG<Data_t, Index_t>::RankedGraphDiversify(Graph_t &graph,
                                                 const int occl_threshold,
                                                 const int max_edge_num) {
#pragma omp parallel for num_threads(build_param_.threads_num)
  for (size_t i = 0; i < graph.size(); i++) {
    std::vector<int> occl_factor = occl_factors_[i];
    std::vector<Data_t> all_dist(graph[i].size());
    std::vector<Index_t> new_list;
    for (size_t j = 0; j < graph[i].size(); j++) {
      all_dist[j] =
          DistFunc_(index_manager_.GetData(i),
                    index_manager_.GetData(graph[i][j]), dist_func_param_);
    }
    for (size_t j = 0; j < graph[i].size(); j++) {
      for (size_t k = j + 1; k < graph[i].size(); k++) {
        if (!is_reverse_edge_[i][j] && !is_reverse_edge_[i][k]) continue;
        int nb1 = graph[i][j];
        int nb2 = graph[i][k];
        Data_t dist2 = all_dist[k];
        Data_t dist3 = DistFunc_(index_manager_.GetData(nb1),
                                 index_manager_.GetData(nb2), dist_func_param_);
        if (dist2 > dist3) occl_factor[k]++;
      }
    }

    std::vector<std::pair<int, int>> occl_factor_tmp;
    for (size_t j = 0; j < graph[i].size(); j++) {
      occl_factor_tmp.emplace_back(occl_factor[j], j);
    }
    std::sort(occl_factor_tmp.begin(), occl_factor_tmp.end());
    for (auto p : occl_factor_tmp) {
      if (p.first > occl_threshold) break;
      new_list.push_back(graph[i][p.second]);
    }
    if (max_edge_num != -1) {
      if (new_list.size() > (size_t)max_edge_num) {
        new_list.erase(new_list.begin() + max_edge_num, new_list.end());
      }
    }
    graph[i] = new_list;
  }

  occl_factors_.clear();
  return;
}

template <typename Data_t, typename Index_t>
int TSDG<Data_t, Index_t>::CheckGraph(const Graph_t &graph) {
  int Max = 0;
  int Min = 1e9;
  long long Avg = 0;
  for (size_t i = 0; i < graph.size(); i++) {
    Max = std::max(graph[i].size(), (size_t)Max);
    Min = std::min(graph[i].size(), (size_t)Min);
    Avg += graph[i].size();
  }
  std::cout << "Min.: " << Min << std::endl;
  std::cout << "Max.: " << Max << std::endl;
  std::cout << "Avg.: " << (double)Avg / graph.size() << std::endl;
  return Max;
}

template <typename Data_t, typename Index_t>
void TSDG<Data_t, Index_t>::Build(const Data_t *data, const Index_t *base_graph,
                                  const int graph_k, const Index_t nrow) {
  Resize(nrow);
  for (int i = 0; i < nrow; i++) {
    Index_t id = id_manager_.ActivateLabel(i);
    AddNewNode(id, data + (size_t)i * dim_);
  }
  Graph_t graph(nrow, std::vector<Index_t>(graph_k));
  for (size_t i = 0; i < nrow; i++) {
    memcpy(graph[i].data(), base_graph + i * graph_k, sizeof(Index_t) * graph_k);
  }

  RelaxedGraphDiversify(graph, build_param_.relax_factor);
  AddReverseEdges(graph);
  RankedGraphDiversify(graph, build_param_.occl_threshold,
                       build_param_.max_edge_num);

  for (size_t i = 0; i < graph.size(); i++) {
    index_manager_.SetListSize(i, graph[i].size());
    memcpy(index_manager_.GetList(i), graph[i].data(),
           sizeof(Index_t) * graph[i].size());
  }
  CheckGraph(graph);
}

}  // namespace tsdg

#endif
