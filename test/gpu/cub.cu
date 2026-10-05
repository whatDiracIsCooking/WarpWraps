// Compile-time acceptance for src/cub.h, the include switch plus one namespace
// alias. The header defines a single name, wwr::wwrcub, and has no in-tree
// caller yet, so without this TU nothing compiles it and a break on one backend
// would ship unseen -- <cub/cub.cuh> and <hipcub/hipcub.hpp> are separately
// written implementations of a shared spelling (hipCUB is a PORT of CUB over
// rocPRIM, not one header behind two paths), the weaker guarantee wmma.cu also
// carries.
//
// Building this .cu under the selected backend's device pass IS the test: every
// wwrcub entity named below has to resolve on both backends -- CUB (CCCL) on
// CUDA, hipCUB on HIP. The kernels are never launched; this tier configures with
// no device. Reached through wwr.device, exactly as a real device consumer
// reaches the header. The launchers have external linkage (as thrust.cu's does)
// so the whole TU is "used" without a ctest entry.
//
// Scope is the portable intersection the header documents: the device-wide, the
// block-level and the warp-level PRIMITIVES, their algorithm-selector enums, and
// the two shared iterators. The util_ptx intrinsics (cub::WARP_THREADS,
// hipcub::LaneId), the one-sided classes (hipcub::BlockShuffle) and the
// radix-rank enum values the header calls out are deliberately NOT named here --
// they do not agree, so naming them would (correctly) break one backend.
#include <cstddef>

#include "cub.h"

namespace c = wwr::wwrcub;

namespace {

constexpr int kBlock = 128;
constexpr int kItems = 4;

// A comparator for the merge-sort families: neither cub:: nor hipcub:: carries a
// ready-made `Less` (both dropped it), so the portable spelling is a caller's
// own functor -- which is the real usage anyway.
struct LessOp {
  __host__ __device__ bool operator()(const int a, const int b) const {
    return a < b;
  }
};

} // namespace

// Block-level collectives: naming each class forces the template -- and its
// nested TempStorage, which needs the class fully instantiated -- so sizeof is a
// presence-and-instantiation check immune to per-method signature quirks. Reduce
// and scan additionally make a representative call, as thrust.cu does per family.
__global__ void wwr_cub_block(float *in, float *out, int *keys) {
  using BReduce = c::BlockReduce<float, kBlock, c::BLOCK_REDUCE_RAKING>;
  using BScan = c::BlockScan<float, kBlock, c::BLOCK_SCAN_WARP_SCANS>;
  __shared__ union {
    typename BReduce::TempStorage reduce;
    typename BScan::TempStorage scan;
  } shared;

  const float v = in[threadIdx.x];
  out[0] = BReduce(shared.reduce).Sum(v);
  float excl;
  BScan(shared.scan).ExclusiveSum(v, excl);
  out[threadIdx.x] = excl;

  // The remaining block families: instantiate and size, proving each name
  // resolves on the selected backend.
  (void)sizeof(typename c::BlockRadixSort<int, kBlock, kItems>::TempStorage);
  (void)sizeof(typename c::BlockLoad<float, kBlock, kItems, c::BLOCK_LOAD_DIRECT>::TempStorage);
  (void)sizeof(typename c::BlockStore<float, kBlock, kItems, c::BLOCK_STORE_DIRECT>::TempStorage);
  (void)sizeof(typename c::BlockDiscontinuity<float, kBlock>::TempStorage);
  (void)sizeof(typename c::BlockExchange<float, kBlock, kItems>::TempStorage);
  (void)sizeof(typename c::BlockHistogram<float, kBlock, kItems, 16>::TempStorage);
  (void)sizeof(typename c::BlockMergeSort<int, kBlock, kItems>::TempStorage);
  (void)keys;
}

// Warp-level collectives: same shape, one warp's worth.
__global__ void wwr_cub_warp(int *in, int *out) {
  using WReduce = c::WarpReduce<int>;
  using WScan = c::WarpScan<int>;
  __shared__ union {
    typename WReduce::TempStorage reduce;
    typename WScan::TempStorage scan;
  } shared;

  const int v = in[threadIdx.x];
  out[0] = WReduce(shared.reduce).Sum(v);
  int excl;
  WScan(shared.scan).ExclusiveSum(v, excl);
  out[threadIdx.x] = excl;

  (void)sizeof(typename c::WarpExchange<int, kItems>::TempStorage);
  (void)sizeof(typename c::WarpLoad<int, kItems, c::WARP_LOAD_DIRECT>::TempStorage);
  (void)sizeof(typename c::WarpStore<int, kItems, c::WARP_STORE_DIRECT>::TempStorage);
  (void)sizeof(typename c::WarpMergeSort<int, kItems, kBlock>::TempStorage);
}

// Device-wide primitives: the two-call (query temp bytes, then run) protocol is
// the whole surface's shape, so one representative call per family both resolves
// the name and exercises that protocol. Host-side launchers in a device TU, as
// thrust.cu's are; never run. The returned status is intentionally discarded.
void wwr_cub_device(void *d_temp, std::size_t &bytes, float *fin, float *fout,
                    int *kin, int *kout, int *flags, int n) {
  (void)c::DeviceReduce::Sum(d_temp, bytes, fin, fout, n);
  (void)c::DeviceScan::ExclusiveSum(d_temp, bytes, fin, fout, n);
  (void)c::DeviceRadixSort::SortKeys(d_temp, bytes, kin, kout, n);
  (void)c::DeviceSegmentedRadixSort::SortKeys(d_temp, bytes, kin, kout, n, 1,
                                              flags, flags);
  (void)c::DeviceSelect::Flagged(d_temp, bytes, kin, flags, kout, flags, n);
  (void)c::DevicePartition::Flagged(d_temp, bytes, kin, flags, kout, flags, n);
  (void)c::DeviceRunLengthEncode::Encode(d_temp, bytes, kin, kout, flags, flags,
                                         n);
  (void)c::DeviceSegmentedReduce::Sum(d_temp, bytes, fin, fout, 1, flags, flags);
  (void)c::DeviceMergeSort::SortKeys(d_temp, bytes, kin, n, LessOp());
  (void)c::DeviceAdjacentDifference::SubtractLeft(d_temp, bytes, kin, n);
  (void)c::DeviceHistogram::HistogramEven(d_temp, bytes, fin, kout, 17, 0.0f,
                                          1.0f, n);

  // Shared iterators: the two hipCUB and CUB both carry. Named as types, which
  // is enough to check the spelling resolves against the backend.
  (void)sizeof(c::ArgIndexInputIterator<int *>);
  (void)sizeof(c::CacheModifiedInputIterator<c::LOAD_CA, int>);
}
