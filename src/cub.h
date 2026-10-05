/**
 * @file cub.h
 * @brief The vendor block/warp/device primitive library (CUB/hipCUB), for device TUs
 *
 * `#include`d into a .cu (CUDA) or `-x hip` device-compiled (HIP) TU; link
 * `wwr.device`. Resolves the one thing that differs -- CUB's `::cub` namespace
 * against hipCUB's `::hipcub` -- and defines exactly one name, `wwr::wwrcub`,
 * aliasing whichever the build selected. Inside it the spellings agree: the
 * device-wide `DeviceReduce`/`DeviceScan`/`DeviceRadixSort`/`DeviceSelect`/...,
 * the block-level `BlockReduce`/`BlockScan`/`BlockRadixSort`/`BlockLoad`/
 * `BlockStore`/..., the warp-level `WarpReduce`/`WarpScan`/..., their
 * algorithm-selector enums (`BLOCK_REDUCE_RAKING`, `BLOCK_SCAN_WARP_SCANS`, ...),
 * and the `TempStorage` / two-call-with-`d_temp_storage` protocol those take.
 *
 * This is `wmma.h`'s shape, not `src/thrust`'s. `src/thrust` needs no backend
 * `#if` for names because both backends spell it `thrust::`, so a plain
 * `using ::thrust::<name>` leaf works there. CUB does not -- it is `cub::` on
 * CUDA and `hipcub::` on HIP -- so the per-header leaf shape would carry that
 * `#if` ~20 times over; one namespace alias carries it once and every member
 * name comes across untouched, the cost `wmma.h` exists to pay a single time.
 * The name follows `wmma` too (`wwrcub`, not `wwr::cub`): this layer must COIN a
 * neutral namespace because the vendors' own names diverge (`cub` != `hipcub`),
 * exactly as `wmma` coins `wwrwmma` for `nvcuda::wmma` != `rocwmma`; `wwr::thrust`
 * keeps the vendor name only because `thrust` is already identical on both. See
 * src/README.md and docs/architecture.md §22.
 *
 * A `.h`, not a `.cuh`, though it is device-only: like `cooperative_groups.h`
 * and `wmma.h` its whole body sits behind the device-pass macros, so a host TU
 * sees an empty header, not the `#error` a `.cuh` carries -- host-*safe* without
 * being host-*usable*. The backend switch is reached through `selected_backend.h`
 * directly, not `runtime.h`: CUB needs only which backend, not `WWR_WARP_SIZE`
 * (each vendor's primitives size their own warp internally) -- the one place
 * cub.h is lighter than wmma.h.
 *
 * Constraints the caller carries (hipCUB is a PORT of CUB, not a clone, so the
 * device/block/warp *primitives* -- the reason to reach for CUB -- agree by
 * name, but three corners do not):
 *
 *   - Block/warp collectives are warp-size-dependent. CUB's `WARP_THREADS` is a
 *     compile-time 32; a HIP build's warp is the device's (32 on RDNA, 64 on
 *     CDNA), so a kernel hard-coding 32 lanes is silently wrong on wave64 -- the
 *     §16/`fragment::num_elements` class of trap. Never a literal warp width.
 *   - util_ptx intrinsics are NOT portable: `hipcub::LaneId()`/`WarpId()` exist
 *     but CUB 13 moved its equivalents out of `::cub`, and `cub::WARP_THREADS`
 *     has no hipCUB counterpart. Reach lane/warp identity through the vendor
 *     runtime, not `wwrcub`.
 *   - Whole classes are one-sided: `hipcub::BlockShuffle`,
 *     `hipcub::TransformInputIterator` and `hipcub::DeviceSpmv` are HIP-only;
 *     CUB's `RADIX_RANK_BASIC` radix-rank enum value is CUDA-only. And neither
 *     carries a ready-made `Less`/`Greater` comparator -- a merge-sort caller
 *     brings its own functor. `test/gpu/cub.cu` pins exactly the portable set.
 */

#pragma once

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from the compiler's own device-pass
// macro. Directly, not through runtime.h: CUB needs only the backend switch,
// not WWR_WARP_SIZE or the neutral runtime surface runtime.h also carries.
// selected_backend.h is host-safe and #errors nowhere, so this header's own
// device gate is the only thing keeping it out of a host TU.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <cub/cub.cuh>

#else

// hipCUB's util_mdspan.hpp spells `std::extents` (its `__cplusplus >= 202302L`
// branch, taken because this tree builds -std=c++23) WITHOUT including <mdspan>,
// so the umbrella below never defines hipcub::extents and device_for.hpp fails
// to parse. Pre-including it is the same "std header the HIP vendor header
// forgot" fix docs/architecture.md §9/§10 make for <array>/<algorithm>; libc++
// provides std::extents here. Must precede the hipCUB header.
#include <mdspan>

#include <hipcub/hipcub.hpp>

#endif

namespace wwr {

// ========================================================================
// The vendor namespace, under one name -- the only name this file defines
// ========================================================================

#if defined(WWR_SELECTED_CUDA)

namespace wwrcub = ::cub;

#else

namespace wwrcub = ::hipcub;

#endif

} // namespace wwr

#endif // device-compile pass
