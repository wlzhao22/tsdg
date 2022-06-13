#ifndef DISTOPT_H
#define DISTOPT_H

#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
#endif

template <typename Metric_t>
using DistFunc_t = Metric_t (*)(const void *, const void *, const void *);

template <typename Metric_t>
class SpaceInterface {
 public:
  // virtual void Search(void *);
  virtual size_t get_data_size() = 0;

  virtual DistFunc_t<Metric_t> get_dist_func() = 0;

  virtual void *get_dist_func_param() = 0;

  virtual ~SpaceInterface() {}
};

#endif  // DISTOPT_H_INCLUDED
