/**
 * @file cooperative_groups.cuh
 * @brief The vendor cooperative-groups header, for device-compiled TUs
 *
 * `#include`d into a .cu (CUDA) or `-x hip` device-compiled (HIP) TU; link
 * `gpumod.device`. Resolves the one thing that differs,
 * `<cooperative_groups.h>` against `<hip/hip_cooperative_groups.h>`, and
 * defines no names of its own: both vendors use `namespace
 * cooperative_groups` and agree on the spellings inside it.
 *
 * Constraints the caller carries, detailed in docs/architecture.md, section 2:
 *
 *   - `cooperative_groups::reduce`, the scan family and `memcpy_async` are
 *     CUDA-only, and not in `<cooperative_groups.h>` on CUDA either.
 *   - `ballot()` is 32-bit on CUDA and 64-bit on HIP; `thread_rank()` and
 *     `num_threads()` vary too, though not for `thread_block` or the tiles.
 *   - Tiles are bounded by WWR_WARP_SIZE, `grid_group` has five portable
 *     members, and `this_grid().sync()` needs a cooperative launch.
 */

#pragma once

// The device-pass #error, WWR_SELECTED_CUDA / WWR_SELECTED_HIP, and
// WWR_WARP_SIZE, which is what a portable tile size is built from.
#include "runtime.cuh"

#if defined(WWR_SELECTED_CUDA)

#include <cooperative_groups.h>

#else

#include <hip/hip_cooperative_groups.h>

#endif
