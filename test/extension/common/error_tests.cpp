// error_tests.cpp - host-only error-mapping tests for wwr.extension.common
//
// common/error.cppm specializes success_code / error_name / error_string for
// wwrError_t by forwarding to the vendor host lookups (wwrGetErrorName /
// wwrGetErrorString, i.e. cuda/hipGetErrorName/String) -- plain string tables,
// no device touched. success_code rides on every gpu_check, but nothing took a
// real error code and stringified it, so error_name / error_string went
// uncovered. These cases pin that forwarding.
//
// Host-only: no device, no handle, nothing allocated -- so this suite is NOT
// REQUIRES_GPU and runs under `ctest -LE gpu` in CI, same as comp_error_tests.

#include <gtest/gtest.h>

import std;
import wwr.extension.common; // error_type / success_code / error_name / error_string
import wwr.runtime_api;       // wwrError_t, wwrSuccess, wwrErrorInvalidValue

namespace wwr::extension::test {

// The concept is the registration: it holds only when all three facilities are
// specialized, so this is the compile-time proof common/error registered
// wwrError_t with the error-handling layer at all.
static_assert(error_type<wwrError_t>);

TEST(CommonErrorTests, SuccessCodeIsSuccess) {
  EXPECT_EQ(success_code<wwrError_t>(), wwrSuccess);
}

TEST(CommonErrorTests, KnownCodesMapToNonEmptyStrings) {
  for (const wwrError_t code : {wwrSuccess, wwrErrorInvalidValue}) {
    const char *name = error_name(code);
    const char *desc = error_string(code);
    ASSERT_NE(name, nullptr);
    ASSERT_NE(desc, nullptr);
    EXPECT_GT(std::strlen(name), 0u);
    EXPECT_GT(std::strlen(desc), 0u);
  }
}

TEST(CommonErrorTests, ErrorCodeNameDiffersFromSuccess) {
  // A real error must not stringify to the success name, or a log could not
  // tell a failure from a success.
  EXPECT_STRNE(error_name(wwrErrorInvalidValue), error_name(wwrSuccess));
}

} // namespace wwr::extension::test
