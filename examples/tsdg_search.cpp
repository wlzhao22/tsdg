#include <bits/stdc++.h>

#include <tsdg.hpp>

#include "filetool.hpp"
#include "timer.hpp"
using namespace std;

float Evaluate(vector<vector<int>> &result_graph, vector<vector<int>> &gt_graph,
               const int recall_at, int cmp_list_cnt = -1) {
  int true_positive = 0, false_negative = 0;
  vector<int> a, b, c;
  int gt_graph_size = gt_graph.size();
  int graph_size = result_graph.size();
  if (cmp_list_cnt == -1) {
    cmp_list_cnt = min(gt_graph_size, graph_size);
  }
  for (int i = 0; i < cmp_list_cnt; i++) {
    a.clear();
    b.clear();
    for (int j = 0; j < recall_at; j++) {
      a.push_back(result_graph[i][j]);
    }
    for (int j = 0; j < recall_at; j++) {
      b.push_back(gt_graph[i][j]);
    }
    c.resize(a.size() + b.size());
    sort(a.begin(), a.end());
    sort(b.begin(), b.end());
    int cnt =
        set_intersection(a.begin(), a.end(), b.begin(), b.end(), c.begin()) -
        c.begin();
    true_positive += cnt;
    false_negative += recall_at - cnt;
  }
  float recall = true_positive / (1.0 * true_positive + false_negative);
  return recall;
}

int main(int argc, char *argv[]) {
  std::string index_file = "../../data/sift1m/index/sift1m.tsdg";
  std::string query_file = "../../data/sift1m/sift1m_query.fvecs";
  std::string gt_file = "../../data/sift1m/sift1m_gt.ivecs";

  if (argc != 1) {
    for (int i = 1; i < argc; i++) {
      cout << argv[i] << "\t";
    }
    cout << endl;
    index_file = argv[1];
    query_file = argv[2];
    gt_file = argv[3];
  }

  int *flat_gt_graph;
  int graph_size;
  int graph_k;
  FileTool::ReadBinaryVecs(gt_file, &flat_gt_graph, &graph_size, &graph_k);

  std::vector<std::vector<int>> gt_graph(graph_size);
  for (int i = 0; i < graph_size; i++) {
    gt_graph[i].resize(graph_k);
    for (int j = 0; j < graph_k; j++) {
      gt_graph[i][j] = flat_gt_graph[(size_t)i * graph_k + j];
    }
  }

  float *query_data;
  int dim;
  int query_num;
  FileTool::ReadBinaryVecs(query_file, &query_data, &query_num, &dim);

  tsdg::TSDG<float, int> tsdg(index_file);

  int top_k = 10;
  int it_num = 3;
  vector<size_t> efs = {10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 70, 100, 128};

  std::vector<std::pair<float, float>> result(efs.size());

  std::vector<int> index_ans(top_k);
  std::vector<float> dist_ans(top_k);

  for (int it = 0; it < it_num; it++) {
    for (int ef_i = 0; ef_i < efs.size(); ef_i++) {
      auto ef = efs[ef_i];
      vector<vector<int>> res(query_num);
      auto s = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < query_num; i++) {
        tsdg.Search(query_data + i * dim, top_k, ef, index_ans.data(),
                       dist_ans.data());
        res[i] = index_ans;
      }
      auto e = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = e - s;
      result[ef_i].first =
          max(result[ef_i].first, (float)Evaluate(res, gt_graph, top_k));
      result[ef_i].second =
          max(result[ef_i].second, (float)(query_num / diff.count()));
      std::cerr << it << "\t" << ef_i << "\r";
    }
    std::cout << std::endl;
  }
  for (int i = 0; i < result.size(); i++) {
    std::cout << result[i].first << "\t" << result[i].second << std::endl;
  }

  delete[] flat_gt_graph;
  delete[] query_data;

  return 0;
}