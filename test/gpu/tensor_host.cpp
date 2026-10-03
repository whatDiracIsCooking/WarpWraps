// tensor_host.cpp - the non-module HOST #include path for the tensor surface
//
// The counterpart to tensor.cppm's module test, through the OTHER path a
// consumer has -- #include "wwr/tensor.h" from a plain, non-module TU -- proving
// that path yields the identical surface bound to the identical backend entities.
// Like the module test it samples rather than restating every name: the full
// surface is pinned name-by-name by tensor.cppm (there is no dispatch test
// downstream), and this shares detail/tensor_names.h with the module, so what is
// under test here is the include PATH.
//
// A plain .cpp that imports nothing and links wwr::tensor::host, the header-only
// target, which carries WWR_GPU_BACKEND_* via wwr::backend. Unlike the module
// test's own executable (gpu_compile_tests_tensor, NO_RUN because libhiptensor
// aborts at process startup on a GPU-less host), this TU takes only addresses /
// values and never calls a vendor entry point, so it is safe to run -- but it is
// kept a compile/link-only check to match the module side's shape.

#include <type_traits>

#include "wwr/tensor.h"

#include "gpu_check_macros.h"

using namespace wwr;

#if defined(WWR_GPU_BACKEND_CUDA)

WWR_SAME_TYPE(wwrtensorHandle_t, cutensorHandle_t)
WWR_SAME_TYPE(wwrtensorComputeDescriptor_t, cutensorComputeDescriptor_t)

WWR_SAME_VALUE(WWRTENSOR_STATUS_SUCCESS, CUTENSOR_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRTENSOR_R_32F, CUTENSOR_R_32F)

WWR_SAME_FUNCTION(wwrtensorCreate, cutensorCreate)
WWR_SAME_FUNCTION(wwrtensorContract, cutensorContract)

// Compute descriptors are extern-const opaque pointers here -- pin object
// identity, the way tensor.cppm's module test does.
static_assert(&WWRTENSOR_COMPUTE_DESC_32F == &CUTENSOR_COMPUTE_DESC_32F,
              "WWRTENSOR_COMPUTE_DESC_32F is not CUTENSOR_COMPUTE_DESC_32F");

#else

WWR_SAME_TYPE(wwrtensorHandle_t, hiptensorHandle_t)
WWR_SAME_TYPE(wwrtensorComputeDescriptor_t, hiptensorComputeDescriptor_t)

WWR_SAME_VALUE(WWRTENSOR_STATUS_SUCCESS, HIPTENSOR_STATUS_SUCCESS)
WWR_SAME_VALUE(WWRTENSOR_R_32F, HIPTENSOR_R_32F)

WWR_SAME_FUNCTION(wwrtensorCreate, hiptensorCreate)
WWR_SAME_FUNCTION(wwrtensorContract, hiptensorContract)

// Compute descriptors are enum values here -- WWR_SAME_VALUE applies.
WWR_SAME_VALUE(WWRTENSOR_COMPUTE_DESC_32F, HIPTENSOR_COMPUTE_DESC_32F)

#endif

// wwrtensorLoggerSetLevel is a forwarding function (cuTENSOR int32_t, hipTensor
// an enum), not a WWR_FUNCTION_RAW alias, so it cannot be WWR_SAME_FUNCTION'd;
// pinning its neutral signature exercises the forwarder through the #include
// path and forces it to link.
WWR_SAME_TYPE(std::remove_cvref_t<decltype(wwrtensorLoggerSetLevel)>,
              wwrtensorStatus_t(std::int32_t))

int main() { return 0; }
