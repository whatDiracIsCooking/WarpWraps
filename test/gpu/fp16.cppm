// fp16.cppm - Compile-time tests for gpumod.fp16
//
// gpuHalf is the backend's own __half. See gpu_check_macros.h.

module;

#include "gpu_check_macros.h"

export module gpumod.test.gpu.fp16;

import std;
import gpumod.fp16;
#if defined(GPUMOD_GPU_BACKEND_CUDA)
import gpumod.cuda.cuda_fp16;
#else
import gpumod.hip.hip_fp16;
#endif

namespace gpumod::test {

using namespace gpumod;

#if defined(GPUMOD_GPU_BACKEND_CUDA)
GPUMOD_SAME_TYPE(gpumod::gpuHalf, gpumod::cuda::__half)
#else
GPUMOD_SAME_TYPE(gpumod::gpuHalf, gpumod::hip::__half)
#endif

// The host conversion wrappers are forwarding functions, not GPUMOD_FUNCTION
// reference bindings, so &gpu != &backend and GPUMOD_SAME_FUNCTION cannot apply. A bare
// GPUMOD_LINK_CHECK from this importing TU is the build-time claim: the exported inline
// wrapper is reachable by name across the import and links. The device-side
// conversions in fp16.cuh are proved separately by fp16.cu.
GPUMOD_LINK_CHECK(gpuFloat2Half)
GPUMOD_LINK_CHECK(gpuHalf2Float)

} // namespace gpumod::test
