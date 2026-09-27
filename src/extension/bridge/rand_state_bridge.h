/**
 * @file rand_state_bridge.h
 * @brief wwr::gpurandState for translation units that cannot `import`
 *
 * gpu_stream_bridge.h's counterpart for the one other type that crosses a
 * host/device boundary by pointer. Same reach -- a .cpp, a .cppm's global
 * module fragment, a .cu or a .cuh -- and the same backend selection, from
 * selected_backend.h.
 *
 * It forward-declares the vendor struct and stops. A caller that only passes
 * gpurandState* through needs the type declared, not complete, which keeps
 * curand_kernel.h and hiprand/hiprand_kernel.h (the latter drags in the whole
 * rocRAND device generator machinery) out of every host compile.
 *
 * The CUDA branch declares ::curandStateXORWOW because ::curandState is itself
 * a typedef and a typedef cannot be forward-declared; on CUDA the two are one
 * type. On HIP they are not -- see docs/architecture.md, section 1.
 */

#pragma once

#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

struct curandStateXORWOW;

namespace wwr {
using gpurandState = ::curandStateXORWOW;
} // namespace wwr

#else

struct hiprandState;

namespace wwr {
using gpurandState = ::hiprandState;
} // namespace wwr

#endif
