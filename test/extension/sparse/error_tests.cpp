// error_tests.cpp - host-only error-mapping tests for wwr.extension.sparse
//
// sparse_error.cppm specializes success_code / error_name / error_string for
// wwrsparseStatus_t by forwarding to wwrsparseGetErrorName / ...String, which
// map to the vendor host lookups (cu/hipsparseGetErrorName/String) -- plain
// string tables, no device. success_code is exercised on every gpu_check, but
// nothing stringified a real status, so error_name / error_string went
// uncovered. These cases pin that forwarding.
//
// Host-only: no device, so NOT REQUIRES_GPU -- runs under `ctest -LE gpu` in CI,
// same as comp_error_tests.

#include <gtest/gtest.h>

import std;
import wwr.extension.common; // error_type / success_code / error_name / error_string
import wwr.extension.sparse; // re-exports wwr.sparse: wwrsparseStatus_t, WWRSPARSE_STATUS_SUCCESS

namespace wwr::extension::test {

// The concept holds only when all three facilities are specialized -- the
// compile-time proof sparse_error registered wwrsparseStatus_t at all.
static_assert(error_type<wwrsparseStatus_t>);

TEST(SparseErrorTests, SuccessCodeIsZeroValuedSuccess) {
  EXPECT_EQ(success_code<wwrsparseStatus_t>(), WWRSPARSE_STATUS_SUCCESS);
  EXPECT_EQ(std::to_underlying(success_code<wwrsparseStatus_t>()), 0);
}

TEST(SparseErrorTests, SuccessCodeStringsAreNonEmpty) {
  const char *name = error_name(success_code<wwrsparseStatus_t>());
  const char *desc = error_string(success_code<wwrsparseStatus_t>());
  ASSERT_NE(name, nullptr);
  ASSERT_NE(desc, nullptr);
  EXPECT_GT(std::strlen(name), 0u);
  EXPECT_GT(std::strlen(desc), 0u);
}

TEST(SparseErrorTests, EveryCodeMapsToNonNullStrings) {
  // The vendor lookups are total -- every status maps to a non-null name and
  // description. Casting small integers keeps this backend-agnostic while
  // reaching error_name / error_string for non-success values, the path
  // success_code's own coverage never exercises.
  for (int raw = 0; raw <= 8; ++raw) {
    const auto code = static_cast<wwrsparseStatus_t>(raw);
    EXPECT_NE(error_name(code), nullptr) << "name for raw code " << raw;
    EXPECT_NE(error_string(code), nullptr) << "string for raw code " << raw;
  }
}

} // namespace wwr::extension::test
