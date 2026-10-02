/**
 * @file rand.cppm
 * @brief Backend-neutral random number generation: wwrrand* names for cuRAND / hipRAND
 *
 * The cuRAND / hipRAND host API, plus the device generator STATE TYPES and
 * nothing else of the device API: device functions are __device__-qualified
 * and live in rand.h's device-pass-gated section, while the state types are plain
 * data and the host is what sizes the per-thread state array. wwrrand<X> stands for curand<X> /
 * hiprand<X>, WWRRAND_<X> for CURAND_<X> / HIPRAND_<X>, each written out in
 * full. See backend.h.
 *
 * Everything the two host APIs share is listed. Absent for want of a
 * counterpart -- cuRAND: curandGetProperty, curandGeneratePoissonMethod,
 * curandGenerateBinomial(Method), curandMethod_t, curandDistribution_t;
 * hipRAND: hiprandGenerateChar/Short, the *Half generators,
 * HIPRAND_STATUS_NOT_IMPLEMENTED. Neither has a status-to-string function.
 *
 * The compile-time CURAND_VERSION / HIPRAND_VERSION macro is not aliased: its
 * value is backend-specific, and the runtime wwrrandGetVersion query is the
 * portable equivalent.
 *
 * wwrrandState and wwrrandStateXORWOW are one type on CUDA and two on HIP, and
 * enumerator VALUES differ even where names agree. See docs/architecture.md,
 * sections 1 and 6.
 *
 * The host API surface itself -- types, constants, functions and the two
 * scramble-constant forwarders -- is NOT restated here: it is the one fragment
 * the two host paths share, detail/rand_names.h, pasted below inside the exported
 * `namespace wwr` exactly as wwr/rand.h (the non-module #include path) pastes it.
 * Add a host name there, once, and both paths gain it.
 *
 * Usage:
 *   import wwr.rand;
 *
 *   wwrrandGenerator_t gen;
 *   wwrrandCreateGenerator(&gen, WWRRAND_RNG_PSEUDO_DEFAULT);
 */

module;

#include "backend.h"

// The single vendor-include point for the rand layer. It brings in BOTH the
// device generator state types (re-exported below, see "Device generator state
// types") AND the host API (curand.h / hiprand.h), which detail/rand_names.h
// binds to directly with the _RAW macros. No raw vendor module is imported: the
// host API is a real external-linkage library, so a reference or type alias needs
// only the declarations this header supplies. rand.h is also included by the
// extension bridges (and reached by a device .cu for its gated generators), so
// every one of them names one identical state type. See rand.h and backend.h.
#include "rand.h"

export module wwr.rand;

export namespace wwr {

// The whole neutral HOST surface -- types, constants, functions and the two
// scramble-constant forwarders -- lives in detail/rand_names.h, the one list the
// two host paths share: this module and wwr/rand.h (the non-module #include
// path). The WWR_*_RAW macros from backend.h and WWR_SELECTED_* and the vendor
// host header from rand.h's #include are exactly what that fragment's header
// documents it needs in scope. The device state types below are a disjoint
// surface and stay out of it.
#include "detail/rand_names.h"

// ========================================================================
// Device generator state types (re-exported from rand.h)
//
// Types only -- the device functions that consume them live in rand.h's
// device-pass-gated section, for the reason in rand.h's header. What the host
// needs these for is
// sizing and allocating the per-thread state array:
//
//   DeviceBufferWrapper<wwrrandState, kit::AbortPolicy<wwrError_t>, kit::AbortPolicy<wwrError_t>,
//                       kit::AbortPolicy<wwrError_t>, MyDeviceHandle>
//       states(count, device);  // device: shared_ptr to a device handle (kit::DeviceHandle,
//                               // a StreamWrapper, or your own); error policy is the kit's
//                               // opt-in kit::AbortPolicy, or your own -- the core forces none
//
// wwrrandState is the default pseudorandom state on both backends, and on
// both backends that default is xorwow-backed. It is NOT, however, the same
// type as wwrrandStateXORWOW on both:
//
//   CUDA:  curandState IS curandStateXORWOW      (one type, two names)
//   HIP:   hiprandState and hiprandStateXORWOW are two DISTINCT structs,
//          separately emitted by hiprand_kernel_rocm.h's
//          DEFINE_HIPRAND_STATE macro over the same rocrand_state_xorwow
//          base -- same layout and size, different C++ types
//
// Verified on ROCm 7.2.4; test/hip/hiprand_kernel.cppm pins it. So a
// wwrrandState* and a wwrrandStateXORWOW* are interchangeable on a CUDA
// build and a compile error on a HIP one. Pick one spelling and keep it: an
// API that takes wwrrandState must be fed wwrrandState.
//
// The sizes differ between backends too (different generator internals), so
// a state array is never a host-side fixed byte count -- always sizeof the
// type.
// ========================================================================

// Re-exported from rand.h (included in the global module fragment above), the
// single home for the state-type list. A plain `using` in this exported
// namespace re-exports the global-module alias to importers; WWR_TYPE is not
// used here because the backend mapping already lives in rand.h -- restating it
// would be the duplication this arrangement removes. A non-module consumer of
// wwr/rand.h sees these straight from rand.h, so that path restates nothing.

// Pseudorandom generators
using wwr::wwrrandStateXORWOW;
using wwr::wwrrandStateXORWOW_t;
using wwr::wwrrandStateMRG32k3a;
using wwr::wwrrandStateMRG32k3a_t;
using wwr::wwrrandStateMtgp32;
using wwr::wwrrandStateMtgp32_t;
using wwr::wwrrandStatePhilox4_32_10;
using wwr::wwrrandStatePhilox4_32_10_t;

// Quasirandom generators
using wwr::wwrrandStateSobol32;
using wwr::wwrrandStateSobol32_t;
using wwr::wwrrandStateScrambledSobol32;
using wwr::wwrrandStateScrambledSobol32_t;
using wwr::wwrrandStateSobol64;
using wwr::wwrrandStateSobol64_t;
using wwr::wwrrandStateScrambledSobol64;
using wwr::wwrrandStateScrambledSobol64_t;

// Default state -- see the section comment above
using wwr::wwrrandState;
using wwr::wwrrandState_t;

} // namespace wwr
