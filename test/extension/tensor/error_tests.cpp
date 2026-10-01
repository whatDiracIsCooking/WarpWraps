// error_tests.cpp - wwr.extension.tensor's wwrtensorStatus_t error mapping
//
// Host-only: success_code / error_name / error_string are pure table lookups
// over the vendor status enum and its wwrtensorGetErrorString, with no device
// contact, so this suite is NOT REQUIRES_GPU and runs in CI (ctest -LE gpu).
// It pins the one contract a wrong specialization would silently break --
// success_code mapping to the zero enumerator, which gpu_check compares against --
// and that the string accessors return a usable (non-null) description for a
// known non-success code.
//
// Backend-neutral -- built for either WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.extension.common; // success_code / error_name / error_string, error_type
import wwr.extension.tensor; // re-exports wwr.tensor: wwrtensorStatus_t, the status enums

namespace wwr::extension::test {

// The specializations must satisfy the error_type concept, exactly as the handle
// wrappers' P_create/P_destroy constraints require at instantiation.
static_assert(error_type<wwrtensorStatus_t>);

TEST(TensorErrorTests, SuccessCodeIsTheSuccessEnumerator) {
  EXPECT_EQ(success_code<wwrtensorStatus_t>(), WWRTENSOR_STATUS_SUCCESS);
  // Pinned at compile time too -- gpu_check compares against the zero enumerator.
  static_assert(std::to_underlying(success_code<wwrtensorStatus_t>()) == 0);
}

TEST(TensorErrorTests, ErrorStringIsNonNullForKnownCode) {
  // cuTENSOR/hipTensor expose one string function, so error_string returns a
  // human-readable description for any valid status enumerator.
  EXPECT_NE(error_string<wwrtensorStatus_t>(WWRTENSOR_STATUS_NOT_SUPPORTED), nullptr);
  EXPECT_NE(error_string<wwrtensorStatus_t>(WWRTENSOR_STATUS_NOT_INITIALIZED), nullptr);
}

TEST(TensorErrorTests, ErrorNameIsNonNullForKnownCode) {
  // error_name routes to the same single vendor string function (there is no
  // separate name/spelling function for cuTENSOR/hipTensor), so it is likewise
  // non-null for a valid enumerator.
  EXPECT_NE(error_name<wwrtensorStatus_t>(WWRTENSOR_STATUS_INVALID_VALUE), nullptr);
}

} // namespace wwr::extension::test
