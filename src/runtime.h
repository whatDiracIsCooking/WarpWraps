/**
 * @file runtime.h
 * @brief The GPU stream type, plus the device-compile runtime, for the runtime layer
 *
 * A src/-root header carrying two things, split by compile context rather than
 * by file -- the `.h` + `.cppm` shape complex.h / rand.h already took:
 *   - wwrStream_t (always), aliased to the vendor's cudaStream_t / hipStream_t
 *     for whichever backend was selected -- the one type a host-allocated stream
 *     and a kernel parameter must agree on;
 *   - the full vendor runtime header, WWR_GRID_CONSTANT, WWR_WARP_SIZE and the
 *     whole backend-neutral runtime surface (every wwr* name wwr.runtime_api
 *     exports -- wwrError_t, wwrMalloc, wwrStreamSynchronize, ...), in a section
 *     gated behind the compiler's device-pass macros -- the device half that
 *     used to live in the separate runtime.cuh, folded in so a device .cu
 *     includes this one neutral header. The surface is the SAME list the module
 *     exports (runtime_api_surface.h), so a device .cu/.cuh -- which cannot
 *     import the module -- still allocates, copies, launches and synchronises
 *     through wwr* names; the device section's banner has the no-collision
 *     argument.
 *
 * It reaches selected_backend.h directly, not device_guard.h, so it carries no
 * device-pass #error and compiles in a host TU too -- the device section simply
 * gates itself out there (where __grid_constant__ is not a keyword and the full
 * runtime header is unwanted, so those must be ABSENT rather than #error). That
 * is why this stays a `.h`: a host GMF or plain .cu that only names wwrStream_t
 * (the *_bridge.h, example/warp_reduce) gets just the type, while a device pass
 * gets the runtime surface on top. The device-only wrappers that must refuse a
 * host compile outright stay `.cuh` (parallel_for.cuh, math.cuh) and reach the
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
 * the runtime API surface (functions, constants, the other handle types) reaches
 * a HOST TU as wwr.runtime_api's, by import; a DEVICE TU, which cannot import,
 * gets that same surface from this header's device section below (the shared
 * runtime_api_surface.h). runtime_api.cppm exports the SAME
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
// The full vendor runtime header plus WWR_GRID_CONSTANT, WWR_WARP_SIZE and the
// whole backend-neutral runtime surface (wwrError_t, wwrMalloc, wwrMemcpyAsync,
// wwrStreamSynchronize, ... -- every wwr* name wwr.runtime_api exports), gated
// behind the compiler's own device-pass macros so the rest of this header still
// compiles in a host TU (where __grid_constant__ is not a keyword, so this must
// be ABSENT rather than #error). A .cu that #includes runtime.h gets them; the
// *_bridge.h and runtime_api.cppm, compiled as host C++, do not -- they use only
// the stream type above. This is the device half that once lived in the separate
// runtime.cuh. Link wwr.device for the include path, the runtime libs and
// WWR_WARP_SIZE's -D.
//
// Why the surface is here as well as in the module: a device .cu/.cuh CANNOT
// import wwr.runtime_api, yet a launcher written in one legitimately allocates,
// copies, launches and synchronises. So runtime.h pastes the SAME list the
// module exports -- runtime_api_surface.h -- binding it to the vendor's ::cuda*
// / ::hip* names from the MINIMAL api header, BEFORE the full runtime header
// below adds the C++ convenience overloads that would defeat the reference
// binding (the section below has the ordering argument). Selection keys on
// WWR_SELECTED_* (a device pass has no WWR_GPU_BACKEND_*, backend.h's key); the
// allocation-flag #undef-to-constexpr dance comes from runtime_api.h.
//
// No collision with the module's export of the same names: this section is
// inside the device gate, and no device-compiled TU imports a module (they reach
// everything by #include) while a host TU that imports wwr.runtime_api skips the
// gate entirely. The always-on wwrStream_t above is re-aliased to the identical
// vendor handle by the fragment -- a well-formed typedef redefinition, and the
// precedent that the two spellings agree.
// ========================================================================

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

// The neutral runtime surface is bound BEFORE the full vendor runtime header
// below, against the MINIMAL api header (cuda_runtime_api.h / hip_runtime_api.h)
// the always-on section already pulled in. That order is load-bearing: the full
// runtime header adds C++ convenience OVERLOADS of some entry points (e.g. a
// second cudaEventCreate(event*, unsigned)), and an overloaded ::cudaX defeats
// the `inline constexpr auto& wwrX = ::cudaX` reference binding ("cannot deduce
// auto"). Binding here, while only the single extern "C" declaration is visible,
// is what the module gets for free from never including the full header at all.
//
// runtime_api.h runs the allocation-flag #undef-to-constexpr dance (its own
// vendor #include is a guarded no-op -- the always-on section included the api
// header already); <cstddef> gives the forwarders std::size_t without an
// `import`. The _RAW selection keys on WWR_SELECTED_* -- the one macro the shared
// fragment lets each site define -- then the fragment is pasted into namespace
// wwr, and every macro this header introduced is #undef'd so none leaks into the
// including TU.
#include <cstddef>

#include "runtime_api.h"

#if defined(WWR_SELECTED_CUDA)
#define WWR_SELECT_RAW(cuda_name, hip_name) ::cuda_name
#else
#define WWR_SELECT_RAW(cuda_name, hip_name) ::hip_name
#endif
#define WWR_TYPE_RAW(wwr_name, cuda_name, hip_name)                                                 \
  using wwr_name = WWR_SELECT_RAW(cuda_name, hip_name);
#define WWR_VALUE_RAW(wwr_name, cuda_name, hip_name)                                                \
  inline constexpr auto wwr_name = WWR_SELECT_RAW(cuda_name, hip_name);
// NOT the module's `inline constexpr auto& = ::cudaX` reference alias: a device
// (.cu) TU force-includes the full vendor runtime header, whose C++ convenience
// OVERLOADS (a second cudaEventCreate(event*, unsigned), ...) leave ::cudaX an
// overload set, and `auto&` cannot deduce against one. A perfect-forwarding
// wrapper sidesteps that -- it restates no signature (it forwards whatever it is
// called with, resolving the overload at the call site) and is callable exactly
// like the reference was. Safe because nothing takes a wwr* runtime function's
// address, only calls it. Types and constants are never overloaded, so their
// _RAW macros keep the module's plain aliasing form above.
#define WWR_FUNCTION_RAW(wwr_name, cuda_name, hip_name)                                             \
  template<typename... WwrArgs>                                                                     \
  inline auto wwr_name(WwrArgs &&...args)                                                           \
      -> decltype(WWR_SELECT_RAW(cuda_name, hip_name)(static_cast<WwrArgs &&>(args)...)) {          \
    return WWR_SELECT_RAW(cuda_name, hip_name)(static_cast<WwrArgs &&>(args)...);                   \
  }
#define WWR_RT_TYPE(x) WWR_TYPE_RAW(wwr##x, cuda##x, hip##x)
#define WWR_RT_VALUE(x) WWR_VALUE_RAW(wwr##x, cuda##x, hip##x)
#define WWR_RT_FUNCTION(x) WWR_FUNCTION_RAW(wwr##x, cuda##x, hip##x)

namespace wwr {
#include "runtime_api_surface.h"
} // namespace wwr

#undef WWR_SELECT_RAW
#undef WWR_TYPE_RAW
#undef WWR_VALUE_RAW
#undef WWR_FUNCTION_RAW
#undef WWR_RT_TYPE
#undef WWR_RT_VALUE
#undef WWR_RT_FUNCTION

// The full vendor runtime header and the device-pass keyword, AFTER the surface
// is bound. This carries __grid_constant__, the <<<>>> launch syntax and the
// device intrinsics kernel code uses; the extra C++ overloads it brings are now
// harmless, every wwr* reference above having already resolved.
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
