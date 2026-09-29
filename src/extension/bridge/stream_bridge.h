/**
 * @file stream_bridge.h
 * @brief wwr::wwrStream_t for translation units that cannot `import`
 *
 * A bridge header: it carries a declaration across the host/device boundary,
 * so it compiles in both a host compile and a device pass. Include it from a
 * .cpp, a .cppm's global module fragment, a .cu or a .cuh.
 *
 * Neither of the other switch points can serve a module unit's global module
 * fragment -- backend.h expands to names only an `import` provides, and
 * runtime.cuh #errors outside a device pass. selected_backend.h covers
 * both, so this header needs no #if beyond choosing the alias.
 *
 * The vendor header included is the minimal one that declares the type, not
 * the full runtime.
 */

#pragma once

#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <cuda_runtime_api.h>

namespace wwr {
using wwrStream_t = cudaStream_t;
} // namespace wwr

#else

#include <hip/hip_runtime_api.h>

namespace wwr {
using wwrStream_t = hipStream_t;
} // namespace wwr

#endif
