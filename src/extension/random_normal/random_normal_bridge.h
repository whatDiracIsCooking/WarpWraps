/**
 * @file random_normal_bridge.h
 * @brief Declaration shared between this module's interface units and its
 *        device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by
 * random_normal.cu directly.
 *
 * The declaration must live in the GMF, not the module purview: a purview name
 * gets module linkage and can never bind to a definition compiled in a plain
 * TU, which is what random_normal.cu is. See docs/architecture.md, section 14.
 *
 * The `_bridge` suffix marks exactly that -- a header carrying a declaration
 * across the host/device boundary, so it must compile in BOTH modes. The
 * extension alone cannot say it: a `.cuh` is device-pass-only, but a plain
 * `.h` may be host-only. See src/README.md.
 *
 * The two pointer types come from the gpu* layer's include-only bridge headers
 * rather than an `import`, since a GMF cannot import. They are the SAME types
 * gpumod.runtime_api / gpumod.rand export, so the wrapper passes its arguments
 * straight through and the device side needs no cast. Reading the backend
 * define they depend on is why this module links gpumod_backend PRIVATE -- see
 * this directory's CMakeLists.txt.
 */

#pragma once

#include "extension/bridge/gpu_stream_bridge.h"
#include "extension/bridge/rand_state_bridge.h"

#include <cstddef>

namespace gpumod::extension::detail {

/// @brief Fill `output` with `count` standard normal values
/// @param states one initialized state per element
template<typename OutputType>
void random_normal(gpuStream_t stream, std::size_t count, gpurandState *states, OutputType *output);

} // namespace gpumod::extension::detail
