/**
 * @file rtc.h
 * @brief The single NVRTC / hipRTC vendor-include point for the rtc layer
 *
 * A src/-root header that brings in the vendor runtime-compilation header for
 * whichever backend was selected, and nothing else: the wwrrtc* surface itself
 * is the shared fragment detail/rtc_names.h, bound to the `::nvrtc*` /
 * `::hiprtc*` declarations this header supplies. It is the "rand.h shape" for a
 * host-only vendor library -- rtc.cppm draws its whole surface from here
 * (binding wwr* references straight to the declarations with the _RAW macros,
 * importing no raw vendor module), and wwr/rtc.h (the non-module #include path)
 * does the same, so this is the one place the vendor header enters the rtc
 * layer. The raw module wwr.cuda.nvrtc / wwr.hip.hiprtc stays as a
 * single-vendor surface for the backend-only extras (NVRTC's CUBIN / LTO IR /
 * OptiX IR getters; hipRTC's hiprtcLink* linking surface), off this path. See
 * src/rtc.cppm and src/README.md.
 *
 * HOST only -- NVRTC / hipRTC has no device surface, so unlike rand.h this
 * header carries no device-pass-gated section. It reaches selected_backend.h
 * directly, not device_guard.h, so it compiles in the host TUs that include it.
 *
 * Unlike solver.h this header needs no pre-included <array>: hip/hiprtc.h is a
 * declarations-only extern "C" header that pulls in none of the HIP vector-type
 * headers, so the host_defines.h / __noinline__ hazard docs/architecture.md
 * section 9 describes does not apply (the raw wwr.hip.hiprtc module includes it
 * with no pre-include for the same reason).
 */

#pragma once

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from WWR_GPU_BACKEND_* in a host compile.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <nvrtc.h>

#else

#include <hip/hiprtc.h>

#endif
