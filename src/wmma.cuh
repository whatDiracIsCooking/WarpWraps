/**
 * @file wmma.cuh
 * @brief The vendor warp-level MMA (tensor-core) API, for device-compiled TUs
 *
 * `#include`d into a .cu (CUDA) or `-x hip` device-compiled (HIP) TU; link
 * `wwr.device`. Resolves the two things that differ -- `<mma.h>` against
 * `<rocwmma/rocwmma.hpp>`, and `nvcuda::wmma` against `rocwmma` -- and defines
 * exactly one name, `wwr::gpuwmma`, aliasing whichever the build selected.
 * Inside it the spellings agree: `fragment`, `matrix_a`/`matrix_b`/
 * `accumulator`, `row_major`/`col_major`, `layout_t`, `fill_fragment`,
 * `load_matrix_sync`, `store_matrix_sync` and `mma_sync`.
 *
 * That one name is the whole difference from `cooperative_groups.cuh`, whose
 * vendors agree on the namespace so it can define nothing; aliasing
 * this one in the caller would need the backend `#if` only this layer carries.
 *
 * Constraints the caller carries:
 *
 *   - 16x16x16 is the only portable shape; CUDA's 32x8x16 and 8x32x16 are not.
 *   - `gpuHalf` is a portable element type; `gpuBfloat16` is NOT.
 *   - `fragment::num_elements` is not portable, and under HIP it differs
 *     between the two compile passes of one TU -- never `static_assert` it.
 *   - CUDA-only: `precision::tf32`, `bmma_sync`, `experimental::precision::*`.
 *     rocWMMA-only: `synchronize_workgroup()`.
 */

#pragma once

// The device-pass #error, WWR_SELECTED_CUDA / WWR_SELECTED_HIP, and
// WWR_WARP_SIZE, which is what a wave index into a tiled kernel is built
// from -- mma_sync is a whole-warp (whole-wavefront) collective, so a kernel
// mapping tiles to waves needs it for the same reason a portable
// cooperative-groups tile size does.
#include "runtime.cuh"

#if defined(WWR_SELECTED_CUDA)

#include <mma.h>

#else

#include <rocwmma/rocwmma.hpp>

#endif

namespace wwr {

// ========================================================================
// The vendor namespace, under one name -- the only name this file defines
// ========================================================================

#if defined(WWR_SELECTED_CUDA)

namespace gpuwmma = ::nvcuda::wmma;

#else

namespace gpuwmma = ::rocwmma;

#endif

} // namespace wwr
