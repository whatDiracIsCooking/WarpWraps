/**
 * @file tensor.cppm
 * @brief Backend-neutral tensor primitives: wwrtensor* names for cuTENSOR / hipTensor
 *
 * Tensor contraction, reduction, permutation and element-wise ops. wwrtensor<X>
 * stands for cutensor<X> on CUDA and hiptensor<X> on HIP, WWRTENSOR_<X> for the
 * corresponding CUTENSOR_<X> / HIPTENSOR_<X> constant. See backend.h.
 *
 * NOT A HIPIFY PAIR. Unlike RCCL/NCCL (src/ccl.cppm, one source-compatible
 * reimplementation, identical spellings) this is TWO independent
 * implementations of the same idea -- cuTENSOR is closed-source over CUDA,
 * hipTensor is built on composable-kernel -- that happen to mirror each other's
 * naming. The intersection was therefore MEASURED, not assumed. Reproduce it:
 *
 *   devtools/header_intersection.py \
 *     --cuda /opt/nvidia/cutensor/include/cutensor.h \
 *            /opt/nvidia/cutensor/include/cutensor/types.h \
 *     --hip  /opt/rocm/include/hiptensor/hiptensor.h \
 *            /opt/rocm/include/hiptensor/hiptensor_types.h
 *
 * cuTENSOR 2.8.1.0 exposes 45 functions; hipTensor 2.2.0 exposes 38; 37 are
 * shared by name, with positionally identical signatures (only the vendor
 * prefix and cudaStream_t/hipStream_t, cutensorDataType_t/hiptensorDataType_t
 * differ -- exactly the substitutions WWR_FUNCTION_RAW's reference binding
 * absorbs). Of those 37, 36 are aliased in the fragment with WWR_FUNCTION_RAW;
 * the one signature divergence is wwrtensorLoggerSetLevel (see below). This
 * layer carries:
 *   - 18 shared types (opaque handles/plans + the enums + the logger callback);
 *   - 93 shared constants whose VALUES also agree per backend -- 28 data types
 *     (hipTensor deliberately numbers HIPTENSOR_R_* to match cudaDataType_t),
 *     29 operators, 11 status codes, 3 workspace prefs, 7 op-descriptor attrs,
 *     6 plan-preference attrs, 1 plan attr, and 2 each autotune/cache/JIT modes;
 *   - the 4 shared compute descriptors, through a per-backend #if (see below).
 * Every value is pinned per backend in test/gpu/tensor.cppm: matching names
 * never guarantee matching values (docs/architecture.md, sections 1 and 6).
 *
 * TWO DIVERGENCES THAT WWR_FUNCTION_RAW / WWR_VALUE_RAW CANNOT REACH (see
 * architecture §4, §5 -- a backend #if is the sanctioned escape hatch for
 * one-off mismatches):
 *
 *   1. Compute descriptors (WWRTENSOR_COMPUTE_DESC_16F/16BF/32F/64F). On CUDA
 *      these are `extern const` opaque *pointers* (runtime globals, not constant
 *      expressions) -- only a reference alias works. On HIP they are enum
 *      *values* (prvalues) -- only a by-value alias works. No single macro spans
 *      both, so they are declared in a per-backend #if. The neutral TYPE
 *      wwrtensorComputeDescriptor_t (a pointer on CUDA, an enum on HIP) always
 *      matches the descCompute parameter of the backend's own create-op calls,
 *      so a portable caller passes WWRTENSOR_COMPUTE_DESC_* straight through.
 *
 *   2. wwrtensorLoggerSetLevel. cuTENSOR takes `int32_t`, hipTensor takes the
 *      hipTensor-only enum `hiptensorLogLevel_t`. A reference alias would give
 *      the neutral name a different parameter type per backend; a forwarding
 *      function gives it a uniform `int32_t` (HIP static_casts into the enum).
 *
 * ABSENT FROM THIS LAYER (no counterpart -- reachable only through the raw
 * wwr.cuda.cutensor / wwr.hip.hiptensor modules):
 *   - cuTENSOR-only: the block-sparse API (cutensorCreateBlockSparse*,
 *     cutensorBlockSparseContract, cutensor{Create,Destroy}BlockSparseTensor-
 *     Descriptor), the trinary contraction (cutensor{Create,}ContractTrinary),
 *     cutensorPlanPreferenceGetAttribute, cutensorGetCudartVersion, the extra
 *     operators (MISH/SWISH/SOFT_PLUS/SOFT_SIGN), algos (GETT/TGETT/TTGT),
 *     compute descriptors (TF32/3XTF32/9X16BF/8XINT8/4X16F), GPU_ARCH plan
 *     preference, BLOCKSPARSE_REPRODUCIBLE, and the CUDA-specific status codes
 *     (MAPPING_ERROR/LICENSE_ERROR/CUBLAS_ERROR/CUDA_ERROR).
 *   - hipTensor-only: hiptensorGetHiprtVersion, the hiptensorLogLevel_t enum,
 *     HIPTENSOR_ALGO_ACTOR_CRITIC, the extra compute descriptors
 *     (C32F/C64F/NONE/8U/8I/32U/32I), and the CK_ERROR/HIP_ERROR status codes.
 *   The version macros (CUTENSOR_VERSION / HIPTENSOR_VERSION and friends) are
 *   backend-specific; wwrtensorGetVersion is the portable runtime query.
 *
 * The wwrtensor* surface itself is NOT restated here: it is the one fragment the
 * two host paths share, detail/tensor_names.h, pasted below inside the exported
 * `namespace wwr` exactly as wwr/tensor.h (the non-module #include path) pastes
 * it. Add a name there, once, and both paths gain it. The vendor header is drawn
 * from tensor.h, the single vendor-include point (the "rand.h shape"): this
 * module binds wwr* references straight to the `::cutensor*` / `::hiptensor*`
 * declarations with the _RAW macros and imports no raw vendor module -- the host
 * API is a real external-linkage library, so a reference needs only the
 * declaration. No import is needed for the surface: its only hand-written
 * function is the logger forwarder, whose std::int32_t comes from <cstdint> (via
 * tensor.h).
 *
 * Usage:
 *   import wwr.tensor;
 *   wwrtensorHandle_t handle;
 *   wwrtensorCreate(&handle);
 *   ... wwrtensorCreateContraction(handle, &desc, ..., WWRTENSOR_COMPUTE_DESC_32F);
 */

module;

#include "backend.h"

// The single vendor-include point for the tensor layer: cutensor.h +
// cutensor/types.h / hiptensor/hiptensor.h + hiptensor_types.h plus <cstdint>,
// whose `::cutensor*` / `::hiptensor*` declarations the _RAW bindings in
// detail/tensor_names.h resolve against. No raw vendor module is imported -- the
// host API is external-linkage, so a reference or type alias needs only these
// declarations. See tensor.h and backend.h.
#include "tensor.h"

export module wwr.tensor;

export namespace wwr {

// The whole neutral tensor surface -- types, constants, the compute-descriptor
// escape hatch, the functions and the logger forwarder -- lives in
// detail/tensor_names.h, the one list both host paths share: this module and
// wwr/tensor.h (the non-module #include path). The WWR_*_RAW macros from
// backend.h and WWR_SELECTED_* and the vendor header from tensor.h's #include
// are exactly what that fragment's header documents it needs in scope.
#include "detail/tensor_names.h"

} // namespace wwr
