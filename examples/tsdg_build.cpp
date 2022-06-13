#include <bits/stdc++.h>

#include <nndescent.hpp>
#include <tsdg.hpp>

#include "filetool.hpp"
#include "timer.hpp"
using namespace std;

int main(int argc, char *argv[]) {
  std::string data_file = "../../data/sift1m/sift1m_base.fvecs";
  std::string scg_file = "../../data/sift1m/index/sift1m_scg.ivecs";
  std::string output_file = "../../data/sift1m/index/sift1m.tsdg";

  auto build_param = tsdg::BuildParam();
  build_param.threads_num = omp_get_max_threads();

  if (argc != 1) {
    for (int i = 1; i < argc; i++) {
      cout << argv[i] << "\t";
    }
    cout << endl;
    data_file = argv[1];
    scg_file = argv[2];
    build_param.relax_factor = atof(argv[3]);
    build_param.occl_threshold = atoi(argv[4]);
    build_param.max_edge_num = atoi(argv[5]);
    output_file = argv[6];
  }

  float *dataset;
  int nrow;
  int dim;
  FileTool::ReadBinaryVecs(data_file, &dataset, &nrow, &dim);

  int *scg;
  int scg_row;
  int scg_k;
  FileTool::ReadBinaryVecs(scg_file, &scg, &scg_row, &scg_k);

  tsdg::TSDG<float, int> tsdg(build_param, dim);

  auto timer = Timer();
  timer.start();
  tsdg.Build(dataset, scg, scg_k, scg_row);

  std::cout << timer.end() << std::endl;

  tsdg.Save(output_file);
  delete[] dataset;
  return 0;
}