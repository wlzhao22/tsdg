#pragma once
#include <cstdlib>
#include <vector>

namespace tsdg {

struct NeighbElem {
  int id = -1;
  float dst = 0;
  NeighbElem(int id, float dst) : id(id), dst(dst){};
  NeighbElem(){};
  bool operator<(const NeighbElem& other) const {
    if (dst == other.dst) return id < other.id;
    return dst < other.dst;
  }
};

struct NBHood {
  std::vector<NeighbElem> nbs;
  std::vector<int> neighbs;
};

}