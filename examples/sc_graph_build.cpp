#include <bits/stdc++.h>

#include <nndescent.hpp>
#include <tsdg.hpp>

#include "filetool.hpp"
#include "timer.hpp"
using namespace std;

int main(int argc, char *argv[]) {
  string data_file = "../data/sift1m/sift1m_base.fvecs";
  string output_file = "../data/sift1m/index/sift1m_scg.ivecs";
  auto build_param = tsdg::nndescent::BuildParam();
  build_param.threads_num = omp_get_max_threads();

  if (argc != 1) {
    for (int i = 1; i < argc; i++) {
      cout << argv[i] << "\t";
    }
    cout << endl;
    data_file = argv[1];
    build_param.graph_k = atoi(argv[2]);
    build_param.sample_num = atoi(argv[3]);
    build_param.iteration_num = atoi(argv[4]);
    output_file = argv[5];
  }

  float *dataset;
  int nrow;
  int dim;
  FileTool::ReadBinaryVecs(data_file, &dataset, &nrow, &dim);
  tsdg::nndescent::NNDescent<float, int> nnd(dataset, nrow, dim, build_param);

  auto timer = Timer();
  timer.start();
  nnd.Build();
  std::cout << "Time costs: " << timer.end() << std::endl;

  nnd.SaveIvecs(output_file);

  delete[] dataset;
  return 0;
}