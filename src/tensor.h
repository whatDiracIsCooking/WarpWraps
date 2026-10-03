/**
 * @file tensor.h
 * @brief The single cuTENSOR / hipTensor vendor-include point for the tensor
 *        layer
 *
 * A src/-root header that brings in the vendor tensor headers for whichever
 * backend was selected, plus <cstdint> for the std::int32_t the logger
 * forwarder names, and nothing else: the wwrtensor* surface itself is the shared
 * fragment detail/tensor_names.h, bound to the `::cutensor*` / `::hiptensor*`
 * declarations this header supplies. It is the "rand.h shape" for a host-only
 * vendor library -- tensor.cppm draws its whole surface from here (binding wwr*
 * references straight to the declarations with the _RAW macros, importing no raw
 * vendor module), and wwr/tensor.h (the non-module #include path) does the same,
 * so this is the one place the vendor header enters the tensor layer. The raw
 * module wwr.cuda.cutensor / wwr.hip.hiptensor stays as a single-vendor surface
 * for the per-backend extras (block-sparse, trinary contraction, the extra
 * compute descriptors), off this path. See src/tensor.cppm and src/README.md.
 *
 * HOST only -- cuTENSOR/hipTensor is a host API, so unlike rand.h / complex.h
 * this header carries no device-pass-gated section. It reaches
 * selected_backend.h directly, not device_guard.h, so it compiles in the host
 * TUs that include it.
 */

#pragma once

// std::int32_t, for the wwrtensorLoggerSetLevel forwarder in
// detail/tensor_names.h -- the one hand-written function this layer carries.
// Replaces the `import std` the module used before the rand.h shape, so the
// fragment is self-sufficient on both host paths.
#include <cstdint>

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from WWR_GPU_BACKEND_* in a host compile.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <cutensor.h>
#include <cutensor/types.h>

#else

// Load-bearing, and must stay before the HIP headers: host_defines.h (pulled in
// transitively by the hipTensor headers) poisons __noinline__ for libc++'s
// __config, so a HIP header reached before <array> makes __config fail to
// compile. Same pre-include src/hip/hiptensor.cppm's global module fragment
// carries. docs/architecture.md, section 9.
#include <array>

// hiptensor 2.2.0 (ROCm 7.2) ships hiptensor.h, which pulls its own version and
// type headers; 2.1.0 (the ROCm floor) ships only hiptensor.hpp, which -- unlike
// the .h -- does not pull its own version header, so hiptensor-version.hpp is
// included alongside it. __has_include picks between them, exactly as
// src/hip/hiptensor.cppm's global module fragment does. docs/architecture.md §21.
#if __has_include(<hiptensor/hiptensor.h>)
#include <hiptensor/hiptensor.h>
#else
#include <hiptensor/hiptensor-version.hpp>
#include <hiptensor/hiptensor.hpp>
#endif

#endif
