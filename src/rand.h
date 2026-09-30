/**
 * @file rand.h
 * @brief The cuRAND / hipRAND device generator STATE TYPES, in one place
 *
 * A src/-root shared-type header (companion to complex.h and runtime.h): it
 * names the vendor generator state types that cross the host/device boundary by
 * pointer, and is the single home the whole rand layer draws them from --
 * rand.cppm re-exports these under its own names, rand.cuh defines its
 * __device__ generators against them, and the extension bridges
 * (init_state_bridge.h, random_normal_bridge.h) name wwrrandState* in a global
 * module fragment where an `import` cannot reach. Like complex.h and runtime.h
 * it reaches selected_backend.h directly, not device_guard.h, so it carries no
 * device-pass #error and compiles in a host TU too.
 *
 * It differs from complex.h / runtime.h in one way that is the whole design
 * trade: their vendor header is host-cheap (cuComplex.h, cuda_runtime_api.h),
 * so they name a complete type at no cost; the state types live only in
 * curand_kernel.h / hiprand_kernel.h, which are heavy (hiprand_kernel.h drags in
 * the whole rocRAND device-generator implementation). Including them here means
 * the parse lands in every TU that #includes rand.h -- but that is exactly three
 * host TUs: rand.cppm's own compile (which builds the wwr.rand BMI) and the two
 * extension wrapper module units, whose global module fragments must #include a
 * bridge because a GMF cannot import. Every consumer that `import`s wwr.rand or
 * the extension modules pays nothing: the BMI exports names, never the header.
 * That bounded cost buys one source of truth for the state-type list, in place
 * of the three copies rand.cppm, rand.cuh and the old forward-declaring bridge
 * each carried. See docs/architecture.md, section 1, and src/README.md.
 *
 * wwrrandState and wwrrandStateXORWOW are one type on CUDA and two on HIP; the
 * _t forms and the other states are all named against the real vendor typedefs
 * here, which is why this header includes the vendor kernel headers rather than
 * forward-declaring (a typedef cannot be forward-declared).
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from the compiler's device macro in a
// device pass or from WWR_GPU_BACKEND_* in a host compile. Directly, not via
// device_guard.h: this header is included from host TUs and must not #error.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <curand_kernel.h>

#else

// <cstdio> first, and it is load-bearing: hiprand_kernel.h ->
// hiprand_kernel_rocm.h -> rocrand/rocrand_kernel.h -> rocrand/rocrand_mtgp32.h
// calls bare `printf` without declaring it. A real `-x hip` compile usually
// drags a declaration in through hip_runtime.h before this point, so the
// failure only shows up when this header is reached first -- include it here so
// the order of includes in the consuming TU cannot matter. Same reason
// src/hip/hiprand_kernel.cppm does it. See docs/architecture.md, section 7.
#include <cstdio>

#include <hiprand/hiprand_kernel.h>

#endif

namespace wwr {

#if defined(WWR_SELECTED_CUDA)

// Pseudorandom generators
using wwrrandStateXORWOW = ::curandStateXORWOW;
using wwrrandStateXORWOW_t = ::curandStateXORWOW_t;
using wwrrandStateMRG32k3a = ::curandStateMRG32k3a;
using wwrrandStateMRG32k3a_t = ::curandStateMRG32k3a_t;
using wwrrandStateMtgp32 = ::curandStateMtgp32;
using wwrrandStateMtgp32_t = ::curandStateMtgp32_t;
using wwrrandStatePhilox4_32_10 = ::curandStatePhilox4_32_10;
using wwrrandStatePhilox4_32_10_t = ::curandStatePhilox4_32_10_t;

// Quasirandom generators
using wwrrandStateSobol32 = ::curandStateSobol32;
using wwrrandStateSobol32_t = ::curandStateSobol32_t;
using wwrrandStateScrambledSobol32 = ::curandStateScrambledSobol32;
using wwrrandStateScrambledSobol32_t = ::curandStateScrambledSobol32_t;
using wwrrandStateSobol64 = ::curandStateSobol64;
using wwrrandStateSobol64_t = ::curandStateSobol64_t;
using wwrrandStateScrambledSobol64 = ::curandStateScrambledSobol64;
using wwrrandStateScrambledSobol64_t = ::curandStateScrambledSobol64_t;

// Default state -- IS wwrrandStateXORWOW here, unlike HIP
using wwrrandState = ::curandState;
using wwrrandState_t = ::curandState_t;

#else

// Pseudorandom generators
using wwrrandStateXORWOW = ::hiprandStateXORWOW;
using wwrrandStateXORWOW_t = ::hiprandStateXORWOW_t;
using wwrrandStateMRG32k3a = ::hiprandStateMRG32k3a;
using wwrrandStateMRG32k3a_t = ::hiprandStateMRG32k3a_t;
using wwrrandStateMtgp32 = ::hiprandStateMtgp32;
using wwrrandStateMtgp32_t = ::hiprandStateMtgp32_t;
using wwrrandStatePhilox4_32_10 = ::hiprandStatePhilox4_32_10;
using wwrrandStatePhilox4_32_10_t = ::hiprandStatePhilox4_32_10_t;

// Quasirandom generators
using wwrrandStateSobol32 = ::hiprandStateSobol32;
using wwrrandStateSobol32_t = ::hiprandStateSobol32_t;
using wwrrandStateScrambledSobol32 = ::hiprandStateScrambledSobol32;
using wwrrandStateScrambledSobol32_t = ::hiprandStateScrambledSobol32_t;
using wwrrandStateSobol64 = ::hiprandStateSobol64;
using wwrrandStateSobol64_t = ::hiprandStateSobol64_t;
using wwrrandStateScrambledSobol64 = ::hiprandStateScrambledSobol64;
using wwrrandStateScrambledSobol64_t = ::hiprandStateScrambledSobol64_t;

// Default state -- its own struct here, NOT wwrrandStateXORWOW
using wwrrandState = ::hiprandState;
using wwrrandState_t = ::hiprandState_t;

#endif

} // namespace wwr
