/**
 * @file wmma.h
 * @brief The vendor warp-level MMA (tensor-core) API, for device-compiled TUs
 *
 * `#include`d into a .cu (CUDA) or `-x hip` device-compiled (HIP) TU; link
 * `wwr.device`. Resolves the two things that differ -- `<mma.h>` against
 * `<rocwmma/rocwmma.hpp>`, and `nvcuda::wmma` against `rocwmma` -- and defines
 * exactly one name, `wwr::wwrwmma`, aliasing whichever the build selected.
 * Inside it the spellings agree: `fragment`, `matrix_a`/`matrix_b`/
 * `accumulator`, `row_major`/`col_major`, `layout_t`, `fill_fragment`,
 * `load_matrix_sync`, `store_matrix_sync` and `mma_sync`.
 *
 * A `.h`, not a `.cuh`, though it is device-only: like `cooperative_groups.h`
 * its whole body -- the vendor headers and that one alias -- sits behind the
 * device-pass macros, so a host TU sees an empty header, not the `#error` a
 * `.cuh` carries. That one alias is the whole difference from
 * `cooperative_groups.h`, whose vendors agree on the namespace so it defines
 * nothing; aliasing this one in the caller would need the backend `#if` only
 * this layer carries. The alias names a device namespace and has no host use, so
 * this header is host-*safe*, not host-*usable* -- which is what sets both apart
 * from `complex.h` / `rand.h`, whose `.h` holds a host-usable surface.
 * `runtime.h` is reached from inside the gate for the backend switch and
 * WWR_WARP_SIZE (mma_sync is a whole-warp collective, so a kernel mapping tiles
 * to waves indexes with it).
 *
 * Constraints the caller carries:
 *
 *   - 16x16x16 is the only portable shape; CUDA's 32x8x16 and 8x32x16 are not.
 *   - `wwrHalf` is a portable element type; `wwrBfloat16` is NOT.
 *   - `fragment::num_elements` is not portable, and under HIP it differs
 *     between the two compile passes of one TU -- never `static_assert` it.
 *   - CUDA-only: `precision::tf32`, `bmma_sync`, `experimental::precision::*`.
 *     rocWMMA-only: `synchronize_workgroup()`.
 */

#pragma once

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

// The device-pass runtime: inside this gate runtime.h's device section is
// active, giving WWR_SELECTED_CUDA / WWR_SELECTED_HIP for the switch below and
// WWR_WARP_SIZE, which is what a wave index into a tiled kernel is built from --
// mma_sync is a whole-warp (whole-wavefront) collective, so a kernel mapping
// tiles to waves needs it for the same reason a portable cooperative-groups
// tile size does. (runtime.h is host-safe; this header's own gate is what keeps
// it out of a host TU.)
#include "runtime.h"

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

namespace wwrwmma = ::nvcuda::wmma;

#else

namespace wwrwmma = ::rocwmma;

#endif

} // namespace wwr

#endif // device-compile pass
