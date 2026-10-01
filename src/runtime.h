/**
 * @file runtime.h
 * @brief The GPU stream type, plus the device-compile runtime, for the runtime layer
 *
 * A src/-root header carrying two things, split by compile context rather than
 * by file -- the `.h` + `.cppm` shape complex.h / rand.h already took:
 *   - wwrStream_t (always), aliased to the vendor's cudaStream_t / hipStream_t
 *     for whichever backend was selected -- the one type a host-allocated stream
 *     and a kernel parameter must agree on;
 *   - the full vendor runtime header, WWR_GRID_CONSTANT and WWR_WARP_SIZE, in a
 *     section gated behind the compiler's device-pass macros -- the device half
 *     that used to live in the separate runtime.cuh, folded in so a device .cu
 *     includes this one neutral header.
 *
 * It reaches selected_backend.h directly, not device_guard.h, so it carries no
 * device-pass #error and compiles in a host TU too -- the device section simply
 * gates itself out there (where __grid_constant__ is not a keyword and the full
 * runtime header is unwanted, so those must be ABSENT rather than #error). That
 * is why this stays a `.h`: a host GMF or plain .cu that only names wwrStream_t
 * (the *_bridge.h, example/warp_reduce) gets just the type, while a device pass
 * gets the runtime surface on top. The device-only wrappers that must refuse a
 * host compile outright stay `.cuh` (atomic.cuh, parallel_for.cuh) and reach the
 * #error through device_guard.h, which they now include directly; the two gated
 * `.h` (cooperative_groups.h, wmma.h) include this header from inside their own
 * device gate. See src/README.md, "The switch points".
 *
 *   WWR_GRID_CONSTANT   __grid_constant__ under CUDA, empty under HIP
 *   WWR_WARP_SIZE       warp/wavefront size, as a constant expression. Set
 *                       with -DWWR_WARP_SIZE (default 32; 64 for CDNA)
 *
 * The minimal vendor header for the type (cuda_runtime_api.h / hip_runtime_api.h)
 * is #included always; the full runtime (cuda_runtime.h / hip/hip_runtime.h) only
 * in the device section, where it is a superset of the minimal one. The rest of
 * the runtime API surface (functions, constants, the other handle types) is
 * wwr.runtime_api's, reached by import. runtime_api.cppm exports the SAME
 * ::cudaStream_t / ::hipStream_t to importers, but does NOT include this header:
 * it draws the handle (and the rest of the surface) from its own vendor-include
 * header runtime_api.h, which #undef's the allocation-flag macros that would
 * collide with its WWR_RT_VALUE expansions -- the #undef the type-only path here
 * leaves undone, since it needs only the handle. The types agree regardless:
 * both headers name the one vendor handle. See src/complex.h, src/backend.h, and
 * docs/architecture.md, section 3.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from the compiler's device macro in a
// device pass or from WWR_GPU_BACKEND_* in a host compile. Directly, not via
// device_guard.h: this header is host-safe and must not #error.
#include "selected_backend.h"

// ========================================================================
// The stream type -- the same one runtime_api.cppm exports and every TU naming
// a stream across the host/device boundary #includes
// ========================================================================

#if defined(WWR_SELECTED_CUDA)

#include <cuda_runtime_api.h>

namespace wwr {
using wwrStream_t = ::cudaStream_t;
} // namespace wwr

#else

#include <hip/hip_runtime_api.h>

namespace wwr {
using wwrStream_t = ::hipStream_t;
} // namespace wwr

#endif

// ========================================================================
// The device-compile runtime -- present only in a device-compile pass
//
// The full vendor runtime header plus WWR_GRID_CONSTANT and WWR_WARP_SIZE, gated
// behind the compiler's own device-pass macros so the rest of this header still
// compiles in a host TU (where __grid_constant__ is not a keyword, so this must
// be ABSENT rather than #error). A .cu that #includes runtime.h gets them; the
// *_bridge.h and runtime_api.cppm, compiled as host C++, do not -- they use only
// the stream type above. This is the device half that once lived in the separate
// runtime.cuh. Link wwr.device for the include path, the runtime libs and
// WWR_WARP_SIZE's -D.
// ========================================================================

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

#if defined(WWR_SELECTED_CUDA)

#include <cuda_runtime.h>

#define WWR_GRID_CONSTANT __grid_constant__

#else

#include <hip/hip_runtime.h>

#define WWR_GRID_CONSTANT

#endif

// Backend-independent, and identical in both of HIP's compile passes -- which
// neither backend's own warp-size spelling is.
#ifndef WWR_WARP_SIZE
#define WWR_WARP_SIZE 32
#endif

static_assert(WWR_WARP_SIZE == 32 || WWR_WARP_SIZE == 64,
              "WWR_WARP_SIZE must be 32 or 64 "
              "(32 for NVIDIA and RDNA, 64 for CDNA)");

#endif // device-compile pass
