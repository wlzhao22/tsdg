#ifndef FILETOOL_HPP
#define FILETOOL_HPP
#include <assert.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace FileTool {
  template <typename Data_t, typename Index_t>
  static void ReadBinaryVecs(const std::string &data_path, Data_t **vectors_ptr,
                             Index_t *num_ptr, int *dim_ptr) {
    Data_t *&vectors = *vectors_ptr;
    Index_t &num = *num_ptr;
    int &dim = *dim_ptr;
    FILE *file_ptr = fopen(data_path.c_str(), "r");
    int status = 0;
    if (file_ptr) {
      status += fread(&dim, sizeof(dim), 1, file_ptr);
      fseek(file_ptr, 0, SEEK_END);
      size_t file_size = ftell(file_ptr);
      Index_t total_num = file_size / (4 + dim * sizeof(Data_t));
      num = total_num;
      rewind(file_ptr);
      vectors = new Data_t[(size_t)num * dim];
      int tmp;
      for (Index_t i = 0; i < num; i++) {
        status += fread(&tmp, sizeof(int), 1, file_ptr);
        status += fread((char *)(vectors + (size_t)i * dim), sizeof(char),
                       dim * sizeof(Data_t), file_ptr);
        if (ferror(file_ptr)) {
          fclose(file_ptr);
          throw std::runtime_error("Read file " + data_path + " error.");
        }
      }
    } else {
      fclose(file_ptr);
      throw std::runtime_error("Open " + data_path + " failed.");
    }
    fclose(file_ptr);
    assert(status);
  }

  template <typename T>
  static void WriteBinaryVecs(const std::string &data_path, const T *vectors,
                              const int num, const int dim) {
    FILE *file_ptr = fopen(data_path.c_str(), "w");
    if (file_ptr != NULL) {
      for (int i = 0; i < num; i++) {
        fwrite((char *)&dim, sizeof(char), 4, file_ptr);
        fwrite((char *)(vectors + (size_t)i * dim), sizeof(char),
               dim * sizeof(T), file_ptr);
        if (ferror(file_ptr)) {
          fclose(file_ptr);
          throw std::runtime_error("Write file " + data_path + " error.");
        }
      }
    } else {
      fclose(file_ptr);
      std::cerr << "Open " << data_path << " failed." << std::endl;
      exit(-1);
    }
    fclose(file_ptr);
  }

  template <typename T>
  static std::vector<std::vector<T>> ReadTxtGraph(const std::string &graph_file) {
    std::vector<std::vector<T>> ret; 
    std::ifstream in(graph_file);
    T graph_size;
    in >> graph_size;
    for (T i = 0; i < graph_size; i++) {
      int k;
      in >> k;
      std::vector<T> list;
      for (int j = 0; j < k; j++) {
        T elem;
        in >> elem;
        list.push_back(elem);
      }
      ret.push_back(list);
    }
    return ret;
  }
};
#endif