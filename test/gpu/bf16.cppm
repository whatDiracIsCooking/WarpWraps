// bf16.cppm - Compile-time tests for gpumod.bf16
//
// gpuBfloat16 is the backend's own bfloat16 type. See gpu_check_macros.h.

module;

#include "gpu_check_macros.h"

export module gpumod.test.gpu.bf16;

import std;
import gpumod.bf16;
#if defined(WWR_GPU_BACKEND_CUDA)
import gpumod.cuda.cuda_bf16;
#else
import gpumod.hip.hip_bf16;
#endif

namespace wwr::test {

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)
WWR_SAME_TYPE(wwr::gpuBfloat16, wwr::cuda::__nv_bfloat16)
#else
WWR_SAME_TYPE(wwr::gpuBfloat16, wwr::hip::__hip_bfloat16)
#endif

// The host conversion wrappers are forwarding functions, not WWR_FUNCTION
// reference bindings, so &gpu != &backend and WWR_SAME_FUNCTION cannot apply. A bare
// WWR_LINK_CHECK from this importing TU is the build-time claim: the exported inline
// wrapper is reachable by name across the import and links. The device-side
// conversions in bf16.cuh are proved separately by bf16.cu.
WWR_LINK_CHECK(gpuFloat2Bfloat16)
WWR_LINK_CHECK(gpuBfloat162Float)

} // namespace wwr::test
