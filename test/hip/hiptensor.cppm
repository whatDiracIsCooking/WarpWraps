// hiptensor.cppm - Compile-time tests for wwr.hip.hiptensor

module;

#include "test/shared/link_check.h"

export module wwr.test.hip.hiptensor;

import std;
import wwr.hip.hiptensor;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-time tests for wwr.hip.hiptensor
//
// Link-check-focused (#117): forces the linker to resolve every re-exported
// function symbol, so a re-export the library does not actually export fails the
// build. A symbol the header declares but the library does not define uses
// WWR_DECLARED_CHECK instead, naming the version it was found missing from.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::hip::test {

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// ────────────────────────────────────────────────────────────────────────

WWR_LINK_CHECK(hiptensorCreate)
WWR_LINK_CHECK(hiptensorDestroy)
WWR_LINK_CHECK(hiptensorGetErrorString)
WWR_LINK_CHECK(hiptensorGetVersion)
WWR_LINK_CHECK(hiptensorGetHiprtVersion)
WWR_LINK_CHECK(hiptensorHandleResizePlanCache)
WWR_LINK_CHECK(hiptensorHandleWritePlanCacheToFile)
WWR_LINK_CHECK(hiptensorHandleReadPlanCacheFromFile)
WWR_LINK_CHECK(hiptensorWriteKernelCacheToFile)
WWR_LINK_CHECK(hiptensorReadKernelCacheFromFile)
WWR_LINK_CHECK(hiptensorCreateTensorDescriptor)
WWR_LINK_CHECK(hiptensorDestroyTensorDescriptor)
WWR_LINK_CHECK(hiptensorCreateElementwiseBinary)
WWR_LINK_CHECK(hiptensorCreateElementwiseTrinary)
WWR_LINK_CHECK(hiptensorCreatePermutation)
WWR_LINK_CHECK(hiptensorCreateContraction)
WWR_LINK_CHECK(hiptensorCreateReduction)
WWR_LINK_CHECK(hiptensorDestroyOperationDescriptor)
WWR_LINK_CHECK(hiptensorOperationDescriptorSetAttribute)
WWR_LINK_CHECK(hiptensorOperationDescriptorGetAttribute)
WWR_LINK_CHECK(hiptensorCreatePlanPreference)
WWR_LINK_CHECK(hiptensorDestroyPlanPreference)
WWR_LINK_CHECK(hiptensorPlanPreferenceSetAttribute)
WWR_LINK_CHECK(hiptensorEstimateWorkspaceSize)
WWR_LINK_CHECK(hiptensorCreatePlan)
WWR_LINK_CHECK(hiptensorDestroyPlan)
WWR_LINK_CHECK(hiptensorPlanGetAttribute)
WWR_LINK_CHECK(hiptensorElementwiseBinaryExecute)
WWR_LINK_CHECK(hiptensorElementwiseTrinaryExecute)
WWR_LINK_CHECK(hiptensorPermute)
WWR_LINK_CHECK(hiptensorContract)
WWR_LINK_CHECK(hiptensorReduce)
WWR_LINK_CHECK(hiptensorLoggerSetCallback)
WWR_LINK_CHECK(hiptensorLoggerSetFile)
WWR_LINK_CHECK(hiptensorLoggerOpenFile)
WWR_LINK_CHECK(hiptensorLoggerSetLevel)
WWR_LINK_CHECK(hiptensorLoggerSetMask)
WWR_LINK_CHECK(hiptensorLoggerForceDisable)

} // namespace wwr::hip::test
