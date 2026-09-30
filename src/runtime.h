/**
 * @file runtime.h
 * @brief The GPU stream type shared by runtime.cuh and runtime_api.cppm
 *
 * wwrStream_t, aliased to the vendor's cudaStream_t / hipStream_t for whichever
 * backend was selected. Companion to complex.h: a src/-root shared-type header,
 * reaching selected_backend.h directly (not device_guard.h) so it carries no
 * device-pass #error and compiles in a host TU too.
 *
 * It is the include-only home wwrStream_t moved to when stream_bridge.h was
 * retired, for the translation units that name the type but cannot import it in
 * the context that needs it: runtime.cuh (a device pass), and a .cppm's global
 * module fragment or a plain .cu that declares a stream-taking function across
 * the host/device boundary (the *_bridge.h headers, example/warp_reduce). Code
 * that CAN import gets the same ::cudaStream_t / ::hipStream_t from
 * wwr.runtime_api instead -- and runtime_api.cppm's GMF deliberately does NOT
 * include this header, because the vendor runtime macros would collide with its
 * WWR_RT_VALUE expansions (see that file). The types agree regardless: both
 * paths name the one vendor handle. This is where runtime diverges from
 * complex.h, whose complex.cppm DOES include it (cuComplex.h defines no such
 * colliding macros).
 *
 * Only the type lives here. The vendor header included is the minimal one that
 * declares it (cuda_runtime_api.h / hip_runtime_api.h), not the full runtime:
 * runtime.cuh layers <cuda_runtime.h> / <hip/hip_runtime.h> and the device
 * macros on top, and the runtime API surface (functions, constants, the other
 * handle types) is wwr.runtime_api's, reached by import. See
 * docs/architecture.md, section 3.
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from the compiler's device macro in a
// device pass or from WWR_GPU_BACKEND_* in a host compile.
#include "selected_backend.h"

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
