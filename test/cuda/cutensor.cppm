// cutensor.cppm - Compile-time tests for wwr.cuda.cutensor

module;

#include "test/shared/link_check.h"

export module wwr.test.cuda.cutensor;

import std;
import wwr.cuda.cutensor;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.cuda.cutensor
//
// Link-check-focused (#117): forces the linker to resolve every re-exported
// function symbol, so a re-export the libcutensor.so does not actually export fails the
// build. A symbol the header declares but the library does not define uses
// WWR_DECLARED_CHECK instead, naming the version it was found missing from.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::cuda::test {

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(cutensorCreate)
WWR_LINK_CHECK(cutensorDestroy)
WWR_LINK_CHECK(cutensorGetErrorString)
WWR_LINK_CHECK(cutensorGetVersion)
WWR_LINK_CHECK(cutensorGetCudartVersion)
WWR_LINK_CHECK(cutensorHandleResizePlanCache)
WWR_LINK_CHECK(cutensorHandleWritePlanCacheToFile)
WWR_LINK_CHECK(cutensorHandleReadPlanCacheFromFile)
WWR_LINK_CHECK(cutensorWriteKernelCacheToFile)
WWR_LINK_CHECK(cutensorReadKernelCacheFromFile)
WWR_LINK_CHECK(cutensorCreateTensorDescriptor)
WWR_LINK_CHECK(cutensorDestroyTensorDescriptor)
WWR_LINK_CHECK(cutensorCreateBlockSparseTensorDescriptor)
WWR_LINK_CHECK(cutensorDestroyBlockSparseTensorDescriptor)
WWR_LINK_CHECK(cutensorCreateElementwiseTrinary)
WWR_LINK_CHECK(cutensorCreateElementwiseBinary)
WWR_LINK_CHECK(cutensorCreatePermutation)
WWR_LINK_CHECK(cutensorCreateContraction)
WWR_LINK_CHECK(cutensorCreateContractionTrinary)
WWR_LINK_CHECK(cutensorCreateBlockSparseContraction)
WWR_LINK_CHECK(cutensorCreateReduction)
WWR_LINK_CHECK(cutensorDestroyOperationDescriptor)
WWR_LINK_CHECK(cutensorOperationDescriptorSetAttribute)
WWR_LINK_CHECK(cutensorOperationDescriptorGetAttribute)
WWR_LINK_CHECK(cutensorCreatePlanPreference)
WWR_LINK_CHECK(cutensorDestroyPlanPreference)
WWR_LINK_CHECK(cutensorPlanPreferenceGetAttribute)
WWR_LINK_CHECK(cutensorPlanPreferenceSetAttribute)
WWR_LINK_CHECK(cutensorEstimateWorkspaceSize)
WWR_LINK_CHECK(cutensorCreatePlan)
WWR_LINK_CHECK(cutensorDestroyPlan)
WWR_LINK_CHECK(cutensorPlanGetAttribute)
WWR_LINK_CHECK(cutensorElementwiseTrinaryExecute)
WWR_LINK_CHECK(cutensorElementwiseBinaryExecute)
WWR_LINK_CHECK(cutensorPermute)
WWR_LINK_CHECK(cutensorContract)
WWR_LINK_CHECK(cutensorContractTrinary)
WWR_LINK_CHECK(cutensorBlockSparseContract)
WWR_LINK_CHECK(cutensorReduce)
WWR_LINK_CHECK(cutensorLoggerSetCallback)
WWR_LINK_CHECK(cutensorLoggerSetFile)
WWR_LINK_CHECK(cutensorLoggerOpenFile)
WWR_LINK_CHECK(cutensorLoggerSetLevel)
WWR_LINK_CHECK(cutensorLoggerSetMask)
WWR_LINK_CHECK(cutensorLoggerForceDisable)

} // namespace wwr::cuda::test
