/**
 * @file cooperative_groups.h
 * @brief The vendor cooperative-groups header, for device-compiled TUs
 *
 * `#include`d into a .cu (CUDA) or `-x hip` device-compiled (HIP) TU; link
 * `wwr.device`. Resolves the one thing that differs,
 * `<cooperative_groups.h>` against `<hip/hip_cooperative_groups.h>`, and
 * defines no names of its own: both vendors use `namespace
 * cooperative_groups` and agree on the spellings inside it.
 *
 * A `.h`, not a `.cuh`, though it is device-only: its whole body sits behind the
 * device-pass macros, so a host TU that includes it sees an empty header rather
 * than the `#error` a `.cuh` carries through `device_guard.h`. It holds no
 * host-usable symbol -- unlike `complex.h` / `rand.h`, whose `.h` pairs a
 * host-usable type surface with a gated device section, this one is device-only
 * through and through and merely host-*safe*. `runtime.cuh` is reached from
 * inside the gate, so a device caller still gets WWR_WARP_SIZE (a portable tile
 * size is built from it) and the backend switch through this one include.
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

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

// The device-pass runtime: the device-pass macros stay satisfied here, plus
// WWR_SELECTED_CUDA / WWR_SELECTED_HIP for the switch below and WWR_WARP_SIZE,
// which is what a portable tile size is built from.
#include "runtime.cuh"

#if defined(WWR_SELECTED_CUDA)

// #include_next, not #include: this header is itself named cooperative_groups.h,
// and src/ is on the angle-bracket search path (-I), so a plain
// `#include <cooperative_groups.h>` resolves back to THIS file (a no-op under
// #pragma once) instead of CUDA's. #include_next resumes the search past src/,
// reaching the toolkit's header. The old .cuh name sidestepped the clash; the .h
// name reintroduces it, and this is the standard wrapper-header fix. The HIP
// branch needs no such thing -- <hip/hip_cooperative_groups.h> does not collide.
#include_next <cooperative_groups.h>

#else

#include <hip/hip_cooperative_groups.h>

#endif

#endif // device-compile pass
