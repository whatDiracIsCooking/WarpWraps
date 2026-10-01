// error_tests.cpp - host-only error-mapping tests for wwr.extension.fft
//
// fft_error.cppm specializes success_code / error_name / error_string for
// wwrfftResult_t by forwarding to wwrfftGetStatusName / ...String -- hand-written
// switches in wwr.fft, pure host string lookups with no device. success_code is
// pinned by a static_assert in the module, but error_name / error_string were
// never called by any test, so they went uncovered. These cases pin the
// forwarding.
//
// Host-only: no device, so NOT REQUIRES_GPU -- runs under `ctest -LE gpu` in CI,
// same as comp_error_tests.

#include <gtest/gtest.h>

import std;
import wwr.extension.common; // error_type / success_code / error_name / error_string
import wwr.extension.fft;    // re-exports wwr.fft: wwrfftResult_t, WWRFFT_SUCCESS

namespace wwr::extension::test {

// The concept holds only when all three facilities are specialized -- the
// compile-time proof fft_error registered wwrfftResult_t at all.
static_assert(error_type<wwrfftResult_t>);

TEST(FftErrorTests, SuccessCodeIsZeroValuedSuccess) {
  EXPECT_EQ(success_code<wwrfftResult_t>(), WWRFFT_SUCCESS);
  EXPECT_EQ(std::to_underlying(success_code<wwrfftResult_t>()), 0);
}

TEST(FftErrorTests, SuccessCodeStringsAreNonEmpty) {
  const char *name = error_name(success_code<wwrfftResult_t>());
  const char *desc = error_string(success_code<wwrfftResult_t>());
  ASSERT_NE(name, nullptr);
  ASSERT_NE(desc, nullptr);
  EXPECT_GT(std::strlen(name), 0u);
  EXPECT_GT(std::strlen(desc), 0u);
}

TEST(FftErrorTests, EveryCodeMapsToNonNullStrings) {
  // Whatever the backend's enumerators are, the switch is total: every value
  // maps to a non-null name and description (the default arm returns a
  // "...UNKNOWN" spelling, never nullptr). Casting small integers keeps this
  // backend-agnostic while reaching error_name / error_string for non-success
  // values -- the path success_code's own coverage never exercises.
  for (int raw = 0; raw <= 8; ++raw) {
    const auto code = static_cast<wwrfftResult_t>(raw);
    EXPECT_NE(error_name(code), nullptr) << "name for raw code " << raw;
    EXPECT_NE(error_string(code), nullptr) << "string for raw code " << raw;
  }
}

} // namespace wwr::extension::test
