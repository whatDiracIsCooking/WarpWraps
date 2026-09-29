// Compile-time test for src/cooperative_groups.cuh, the include
// switch. The header defines no wwr* names of its own, and its one in-tree
// caller (example/warp_reduce) reaches a fraction of what it exposes -- so
// without this TU most of the header goes uncompiled, and a break on one
// backend would ship unseen (nvcc and clang's -x hip disagree on more than the
// include path).
//
// Building this .cu under the selected backend's device pass IS the test: every
// cooperative_groups entity named below has to resolve through the one include
// switch on both backends. The kernels are never launched -- the shfl_down
// ladder's answer is checked on a device by example/warp_reduce. Reached
// through wwr.device, exactly as a real device consumer reaches the header.
//
// The static_asserts turn the silent divergences the header documents into
// compile-time tripwires: they name a difference that otherwise costs nothing to
// get wrong until it runs on the other vendor. Backend is selected on
// WWR_SELECTED_CUDA -- the device-pass macro the header itself switches on,
// available here for free and consistent across both of HIP's compile passes
// (unlike the WWR_GPU_BACKEND_* define the host-compiled .cppm tests use).
#include "cooperative_groups.cuh"

namespace cg = cooperative_groups;

// thread_block and a full-warp tile: the core portable surface, plus the two
// guards. A tile is bounded by WWR_WARP_SIZE (32 on NVIDIA and RDNA, 64 on
// CDNA); tiled_partition and thread_block_tile<N> are the portable spellings.
__global__ void wwr_coop_block_and_tile(unsigned *out) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<WWR_WARP_SIZE> tile = cg::tiled_partition<WWR_WARP_SIZE>(block);

  // Silent divergence #1: ballot() is 32-bit on CUDA and 64-bit on
  // HIP, so a caller storing it in a fixed-width type is wrong on the other
  // vendor with no diagnostic. Pin the width per backend so a vendor changing it
  // -- or this header switching to a group type whose ballot differs -- fails
  // the build instead of a wave64 run.
#if defined(WWR_SELECTED_CUDA)
  static_assert(sizeof(decltype(tile.ballot(true))) == 4,
                "CUDA cooperative-groups ballot() is expected to be 32-bit");
#else
  static_assert(sizeof(decltype(tile.ballot(true))) == 8,
                "HIP cooperative-groups ballot() is expected to be 64-bit");
#endif

  // The flip side: thread_rank() is the same width on both, so a portable caller
  // can rely on it where ballot() traps. Pinning it documents the contrast.
  static_assert(sizeof(decltype(tile.thread_rank())) == 4,
                "cooperative-groups thread_rank() is expected to be 32-bit "
                "on both backends");

  unsigned acc = 0;
  acc += block.thread_rank();
  acc += block.num_threads();
  acc += block.size();
  acc += block.group_index().x;
  acc += block.thread_index().x;
  acc += block.group_dim().x; // static on CUDA, a member on HIP -- both dispatch
  block.sync();

  acc += tile.thread_rank();
  acc += tile.num_threads();
  acc += tile.size();
  acc += tile.meta_group_rank();
  acc += tile.meta_group_size();
  acc += static_cast<unsigned>(tile.shfl(static_cast<int>(acc), 0));
  acc += static_cast<unsigned>(tile.shfl_down(static_cast<int>(acc), 1));
  acc += static_cast<unsigned>(tile.shfl_up(static_cast<int>(acc), 1));
  acc += static_cast<unsigned>(tile.shfl_xor(static_cast<int>(acc), 1));
  acc += tile.any(true) ? 1u : 0u;
  acc += tile.all(true) ? 1u : 0u;
  acc += static_cast<unsigned>(tile.ballot(true));
  tile.sync();

  out[0] = acc;
}

// Sub-warp partitioning: a tile smaller than the warp is the other tiled_
// partition size that has to work, and its meta_group_* members index it within
// the parent. Both are portable.
__global__ void wwr_coop_subwarp_tile(unsigned *out) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<2> sub = cg::tiled_partition<2>(block);

  unsigned acc =
      sub.thread_rank() + sub.num_threads() + sub.meta_group_rank() + sub.meta_group_size();
  acc += static_cast<unsigned>(sub.shfl_down(static_cast<int>(acc), 1));
  out[0] = acc;
}

// grid_group's portable members. Its .sync() needs a cooperative launch, which
// nothing in this tree performs, so it is named nowhere; block_rank() and
// friends are CUDA-only (no HIP counterpart), so they are deliberately absent
// from this portable TU.
__global__ void wwr_coop_grid(unsigned *out) {
  cg::grid_group grid = cg::this_grid();
  unsigned acc = grid.thread_rank();
  acc += grid.num_threads();
  acc += grid.size();
  acc += grid.is_valid() ? 1u : 0u;
  out[0] = acc;
}

// coalesced_group: a distinct portable group type, the currently-active threads
// of the warp. Both vendors spell it the same way.
__global__ void wwr_coop_coalesced(unsigned *out) {
  cg::coalesced_group active = cg::coalesced_threads();
  unsigned acc = active.thread_rank() + active.num_threads();
  acc += static_cast<unsigned>(active.shfl(static_cast<int>(acc), 0));
  out[0] = acc;
}
