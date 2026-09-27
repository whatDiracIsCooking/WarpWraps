// fp16.cppm - Compile-time tests for gpumod.fp16
//
// gpuHalf is the backend's own __half. See gpu_check_macros.h.

module;

#include "gpu_check_macros.h"

export module gpumod.test.gpu.fp16;

import std;
import gpumod.fp16;
#if defined(WWR_GPU_BACKEND_CUDA)
import gpumod.cuda.cuda_fp16;
#else
import gpumod.hip.hip_fp16;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)
WWR_SAME_TYPE(wwr::gpuHalf, wwr::cuda::__half)
#else
WWR_SAME_TYPE(wwr::gpuHalf, wwr::hip::__half)
#endif

// The host conversion wrappers are forwarding functions, not WWR_FUNCTION
// reference bindings, so &gpu != &backend and WWR_SAME_FUNCTION cannot apply. A bare
// WWR_LINK_CHECK from this importing TU is the build-time claim: the exported inline
// wrapper is reachable by name across the import and links. The device-side
// conversions in fp16.cuh are proved separately by fp16.cu.
WWR_LINK_CHECK(gpuFloat2Half)
WWR_LINK_CHECK(gpuHalf2Float)

} // namespace wwr::test
