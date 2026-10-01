// error_tests.cpp - host-only error-mapping tests for wwr.extension.comp
//
// wwr.comp exposes no vendor status->string function, so comp_error hand-writes
// the success_code / error_name / error_string specializations for
// wwrcompStatus_t as a switch over the six WWRCOMP_* codes. These cases pin that
// mapping: the success code is WWRCOMP_SUCCESS, every known code maps to a
// non-null name and description, and an out-of-range value falls through to the
// "unknown" default rather than reading off the end of a table.
//
// Host-only: no device, no vendor handle, nothing allocated -- so this suite is
// NOT REQUIRES_GPU and runs under `ctest -LE gpu` in CI, which is where the error
// mapping gets its coverage (the scratch RAII, which needs a device, is the
// separate REQUIRES_GPU suite in scratch_tests.cpp).

#include <gtest/gtest.h>

import std;
import wwr.extension.common; // error_type / success_code / error_name / error_string
import wwr.extension.comp;    // re-exports wwr.comp: wwrcompStatus_t, WWRCOMP_* codes

namespace wwr::extension::test {

// error_type<wwrcompStatus_t> holds iff all three facilities are specialized --
// the concept is the registration, so this is the compile-time proof comp_error
// registered the status type with the error-handling layer at all.
static_assert(error_type<wwrcompStatus_t>);

TEST(CompErrorTests, SuccessCodeIsSuccess) {
  EXPECT_EQ(success_code<wwrcompStatus_t>(), WWRCOMP_SUCCESS);
  // The success enumerator is zero -- the same contract comp_error static_asserts.
  EXPECT_EQ(std::to_underlying(success_code<wwrcompStatus_t>()), 0);
}

TEST(CompErrorTests, EveryKnownCodeMapsToNonNullStrings) {
  for (const wwrcompStatus_t code :
       {WWRCOMP_SUCCESS, WWRCOMP_ERROR_INVALID_VALUE, WWRCOMP_ERROR_NOT_SUPPORTED,
        WWRCOMP_ERROR_CANNOT_DECOMPRESS, WWRCOMP_ERROR_CUDA_ERROR, WWRCOMP_ERROR_INTERNAL}) {
    const char *name = error_name(code);
    const char *desc = error_string(code);
    ASSERT_NE(name, nullptr);
    ASSERT_NE(desc, nullptr);
    EXPECT_GT(std::strlen(name), 0u);
    EXPECT_GT(std::strlen(desc), 0u);
    // A known code must not fall through to the "unknown" default.
    EXPECT_STRNE(name, "unknown");
    EXPECT_STRNE(desc, "unknown compression status");
  }
}

TEST(CompErrorTests, NamesAreDistinctAndSpellTheEnumerator) {
  // The hand-written name switch must return each enumerator's own spelling, not
  // a shared placeholder -- otherwise a log could not tell two failures apart.
  EXPECT_STREQ(error_name(WWRCOMP_SUCCESS), "WWRCOMP_SUCCESS");
  EXPECT_STREQ(error_name(WWRCOMP_ERROR_INVALID_VALUE), "WWRCOMP_ERROR_INVALID_VALUE");
  EXPECT_STREQ(error_name(WWRCOMP_ERROR_NOT_SUPPORTED), "WWRCOMP_ERROR_NOT_SUPPORTED");
  EXPECT_STREQ(error_name(WWRCOMP_ERROR_CANNOT_DECOMPRESS), "WWRCOMP_ERROR_CANNOT_DECOMPRESS");
  EXPECT_STREQ(error_name(WWRCOMP_ERROR_CUDA_ERROR), "WWRCOMP_ERROR_CUDA_ERROR");
  EXPECT_STREQ(error_name(WWRCOMP_ERROR_INTERNAL), "WWRCOMP_ERROR_INTERNAL");
}

TEST(CompErrorTests, UnknownCodeHitsTheDefault) {
  // A value outside the six defined codes must hit the switch default rather than
  // index off the end of any table. 999 is far past the intersection's range.
  const auto bogus = static_cast<wwrcompStatus_t>(999);
  EXPECT_STREQ(error_name(bogus), "unknown");
  EXPECT_STREQ(error_string(bogus), "unknown compression status");
}

} // namespace wwr::extension::test
