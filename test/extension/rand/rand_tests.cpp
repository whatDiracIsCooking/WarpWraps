// rand_tests.cpp - Runtime tests for wwr.extension.init_state and
//                  wwr.extension.random_normal
//
// These two modules are one integration test on purpose: random_normal draws
// from states that init_state seeds, so every case here needs both. It runs
// the kernels on-device -- init_state seeds one generator state per element,
// random_normal draws from them, and the results come back to the host to be
// checked. That is the only way to test them -- the device functions they are
// built on (src/rand.cuh) cannot be called from host code at all, so there is
// no compile-time equivalent of test/gpu/rand.cppm for them.
//
// A plain .cpp, not a .cppm -- same reasoning as every other test/extension
// suite (see test/extension/memory_buffer/buffer_tests.cpp): GoogleTest
// headers and `import std;` coexist fine in an ordinary TU, but not inside a
// module interface unit's global module fragment.
//
// What is checked, and what deliberately is not:
//
//   * Statistical claims are kept coarse ON PURPOSE. The sample mean and
//     variance of n draws are themselves random, so a tight bound here would
//     be a test that fails occasionally for no reason. The tolerances below
//     are many standard errors wide -- they catch a generator that is broken
//     (all zeros, wrong scale, uninitialized state), not one that is
//     mis-seeded in the last bit.
//   * The exact VALUES are not pinned to a golden list. cuRAND and hipRAND do
//     not produce the same stream for the same seed -- different generator
//     internals -- so any such expectation could only hold on one backend.
//   * Independence between elements is checked the cheap way: two elements
//     seeded onto different subsequences must not produce identical values.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;
import wwr.rand;
import wwr.complex;
import wwr.fp16;
import wwr.bf16;
import wwr.extension.runtime;
import wwr.extension.memory_buffer;
import wwr.extension.init_state;
import wwr.extension.random_normal;

namespace wwr::extension::test {

namespace {

constexpr std::size_t kCount = 100000;
constexpr unsigned long long kSeed = 20250923;

/// @brief Draw `count` values of OutputType and copy them back to the host
///
/// `scale` is forwarded to random_normal (defaults to the module's own default
/// of 1) and `offset` to init_state's per-subsequence skip-ahead (its 6th arg,
/// distinct from `sequence_offset`).
template<typename OutputType>
std::vector<OutputType> draw(const std::size_t count, const unsigned long long seed = kSeed,
                             const unsigned long long sequence_offset = 0,
                             const OutputType scale = OutputType{1.0f},
                             const unsigned long long offset = 0) {
  auto handle = std::make_shared<DeviceHandle>(0);
  GpuStreamWrapper<> &stream = handle->alloc_stream();
  DeviceBufferWrapper<wwrrandState> states(count, handle);
  DeviceBufferWrapper<OutputType> values(count, handle);

  init_state(stream.get(), count, states.data(), seed, sequence_offset, offset);
  random_normal(stream.get(), count, states.data(), values.data(), scale);

  HostBufferWrapper<OutputType> host(count);
  EXPECT_EQ(copy(host, values, stream.get()), wwrSuccess);
  EXPECT_EQ(wwrStreamSynchronize(stream.get()), wwrSuccess);

  return std::vector<OutputType>(host.data(), host.data() + count);
}

/// @brief Mean and (population) variance of a sample
struct Moments {
  double mean;
  double variance;
};

Moments moments(const std::vector<double> &values) {
  const double n = static_cast<double>(values.size());
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / n;
  double sum_sq = 0.0;
  for (const double v : values) {
    sum_sq += (v - mean) * (v - mean);
  }
  return {mean, sum_sq / n};
}

/// @brief Assert a sample looks standard normal, loosely (see file header)
void expect_standard_normal(const std::vector<double> &values) {
  ASSERT_FALSE(values.empty());

  const auto [mean, variance] = moments(values);

  // Standard error of the mean is 1/sqrt(n); 0.05 is ~15 of those at
  // n = 100000. Wide on purpose.
  EXPECT_NEAR(mean, 0.0, 0.05) << "sample mean is not near 0";
  EXPECT_NEAR(variance, 1.0, 0.05) << "sample variance is not near 1";

  // A standard normal sample of this size essentially always has values on
  // both sides of zero and some beyond 2 sigma. All-zeros or a constant
  // fails here even if the moments somehow did not.
  const auto negatives = std::ranges::count_if(values, [](double v) { return v < 0.0; });
  EXPECT_GT(negatives, 0) << "no negative values at all";
  EXPECT_LT(negatives, static_cast<std::ptrdiff_t>(values.size()))
      << "no non-negative values at all";
  EXPECT_TRUE(std::ranges::any_of(values, [](double v) { return std::abs(v) > 2.0; }))
      << "no value beyond 2 sigma";

  EXPECT_TRUE(std::ranges::all_of(values, [](double v) { return std::isfinite(v); }))
      << "sample contains a non-finite value";
}

} // namespace

TEST(RandTests, FloatIsStandardNormal) {
  const auto values = draw<float>(kCount);
  expect_standard_normal(std::vector<double>(values.begin(), values.end()));
}

TEST(RandTests, DoubleIsStandardNormal) {
  expect_standard_normal(draw<double>(kCount));
}

TEST(RandTests, HalfIsStandardNormal) {
  const auto values = draw<wwrHalf>(kCount);
  std::vector<double> as_double;
  as_double.reserve(values.size());
  for (const wwrHalf v : values) {
    as_double.push_back(static_cast<double>(static_cast<float>(v)));
  }
  // Half precision quantizes heavily (~2^-11 relative), so the moments are
  // a little looser here than the shared helper's bound would allow.
  const auto [mean, variance] = moments(as_double);
  EXPECT_NEAR(mean, 0.0, 0.05);
  EXPECT_NEAR(variance, 1.0, 0.1);
  EXPECT_TRUE(std::ranges::any_of(as_double, [](double v) { return v < 0.0; }));
  EXPECT_TRUE(std::ranges::any_of(as_double, [](double v) { return v > 0.0; }));
}

TEST(RandTests, Bfloat16IsStandardNormal) {
  const auto values = draw<wwrBfloat16>(kCount);
  std::vector<double> as_double;
  as_double.reserve(values.size());
  for (const wwrBfloat16 v : values) {
    as_double.push_back(static_cast<double>(static_cast<float>(v)));
  }
  // bfloat16 keeps only 8 mantissa bits -- looser still than half.
  const auto [mean, variance] = moments(as_double);
  EXPECT_NEAR(mean, 0.0, 0.05);
  EXPECT_NEAR(variance, 1.0, 0.15);
  EXPECT_TRUE(std::ranges::any_of(as_double, [](double v) { return v < 0.0; }));
  EXPECT_TRUE(std::ranges::any_of(as_double, [](double v) { return v > 0.0; }));
}

// Each component of a complex draw is an independent standard normal: each
// part has variance 1 and |z|^2 has expectation 2. random_normal does no
// per-component normalization -- a caller wanting the complex value as a whole
// to be standard normal passes scale = 1/sqrt(2). This is the claim
// random_normal.cppm's header makes, and the one most likely to be got wrong.
TEST(RandTests, FloatComplexComponentsAreStandardNormal) {
  const auto values = draw<wwrFloatComplex>(kCount);

  std::vector<double> parts;
  parts.reserve(2 * values.size());
  double sum_magnitude_sq = 0.0;
  for (const wwrFloatComplex z : values) {
    parts.push_back(static_cast<double>(z.x));
    parts.push_back(static_cast<double>(z.y));
    sum_magnitude_sq += static_cast<double>(z.x) * z.x + static_cast<double>(z.y) * z.y;
  }

  const auto [mean, variance] = moments(parts);
  EXPECT_NEAR(mean, 0.0, 0.05) << "component mean is not near 0";
  EXPECT_NEAR(variance, 1.0, 0.05) << "component variance is not near 1";
  EXPECT_NEAR(sum_magnitude_sq / static_cast<double>(values.size()), 2.0, 0.05)
      << "E[|z|^2] is not near 2";
}

TEST(RandTests, DoubleComplexComponentsAreStandardNormal) {
  const auto values = draw<wwrDoubleComplex>(kCount);

  std::vector<double> parts;
  parts.reserve(2 * values.size());
  double sum_magnitude_sq = 0.0;
  for (const wwrDoubleComplex z : values) {
    parts.push_back(z.x);
    parts.push_back(z.y);
    sum_magnitude_sq += z.x * z.x + z.y * z.y;
  }

  const auto [mean, variance] = moments(parts);
  EXPECT_NEAR(mean, 0.0, 0.05);
  EXPECT_NEAR(variance, 1.0, 0.05);
  EXPECT_NEAR(sum_magnitude_sq / static_cast<double>(values.size()), 2.0, 0.05);
}

// The `scale` argument of random_normal is otherwise entirely dark: every test
// above draws with the default scale of 1. For a real type it scales the
// standard deviation, so the sample variance must scale by scale^2. This drives
// the real (non-complex, non-half) scale-multiply path.
TEST(RandTests, RealScaleScalesVarianceBySquare) {
  constexpr double s = 2.0;
  const auto values = draw<double>(kCount, kSeed, 0, s);

  const auto [mean, variance] = moments(values);
  EXPECT_NEAR(mean, 0.0, 0.1) << "scaled sample mean is not near 0";
  EXPECT_NEAR(variance, s * s, 0.2) << "variance did not scale by scale^2";
}

// A complex draw has component variance 1 and E[|z|^2] = 2 unscaled. Passing
// scale = 1/sqrt(2) (as a complex value) normalizes E[|z|^2] to 1 -- the exact
// claim random_normal's header makes. This drives the complex-multiply scale
// path (wwrCmulf), distinct from the real one above.
TEST(RandTests, ComplexScaleNormalizesMagnitudeVariance) {
  wwrFloatComplex s{};
  s.x = static_cast<float>(1.0 / std::sqrt(2.0));
  s.y = 0.0f;
  const auto values = draw<wwrFloatComplex>(kCount, kSeed, 0, s);

  double sum_magnitude_sq = 0.0;
  for (const wwrFloatComplex z : values) {
    sum_magnitude_sq += static_cast<double>(z.x) * z.x + static_cast<double>(z.y) * z.y;
  }
  EXPECT_NEAR(sum_magnitude_sq / static_cast<double>(values.size()), 1.0, 0.05)
      << "scale = 1/sqrt(2) did not normalize E[|z|^2] to 1";
}

// The half types are drawn in single precision, scaled, then converted -- a
// third distinct scale path (scale flows through the float<->half conversion).
// Tolerances are looser than the real path, matching HalfIsStandardNormal.
TEST(RandTests, ScaleAppliesThroughHalfConversion) {
  const wwrHalf s = static_cast<wwrHalf>(2.0f);
  const auto values = draw<wwrHalf>(kCount, kSeed, 0, s);

  std::vector<double> as_double;
  as_double.reserve(values.size());
  for (const wwrHalf v : values) {
    as_double.push_back(static_cast<double>(static_cast<float>(v)));
  }
  const auto [mean, variance] = moments(as_double);
  EXPECT_NEAR(mean, 0.0, 0.1);
  EXPECT_NEAR(variance, 4.0, 0.3) << "scale did not apply on the half conversion path";
}

// init_state's `offset` (its 6th argument) skips each state ahead within its
// own subsequence. Same seed and same subsequences but a non-zero offset must
// therefore produce a different stream -- otherwise the argument does nothing.
// (SequenceOffsetShiftsTheStreams covers the 5th argument; this covers the 6th.)
TEST(RandTests, SubsequenceOffsetSkipsAhead) {
  const auto base = draw<double>(1000, kSeed, 0, 1.0, 0);
  const auto skipped = draw<double>(1000, kSeed, 0, 1.0, 500);
  EXPECT_NE(base, skipped);
}

// init_state puts element i on subsequence sequence_offset + i. If that offset
// were dropped -- every state seeded identically -- every element would draw
// the same value, which is the single most damaging way this module could
// silently break.
TEST(RandTests, ElementsAreIndependent) {
  const auto values = draw<double>(1000);

  const std::size_t distinct = std::set<double>(values.begin(), values.end()).size();
  EXPECT_EQ(distinct, values.size()) << "elements repeat -- states are not on distinct "
                                        "subsequences";
}

// Same seed, same subsequences, same draws: the generator is deterministic.
TEST(RandTests, SameSeedReproducesTheSameDraws) {
  const auto first = draw<double>(1000, kSeed);
  const auto second = draw<double>(1000, kSeed);
  EXPECT_EQ(first, second);
}

TEST(RandTests, DifferentSeedGivesDifferentDraws) {
  const auto first = draw<double>(1000, kSeed);
  const auto second = draw<double>(1000, kSeed + 1);
  EXPECT_NE(first, second);
}

// A second call on the SAME states continues the streams rather than
// restarting them -- random_normal advances each state as it draws.
TEST(RandTests, StatesAdvanceAcrossCalls) {
  constexpr std::size_t n = 1000;

  auto handle = std::make_shared<DeviceHandle>(0);
  GpuStreamWrapper<> &stream = handle->alloc_stream();
  DeviceBufferWrapper<wwrrandState> states(n, handle);
  DeviceBufferWrapper<double> values(n, handle);
  HostBufferWrapper<double> host(n);

  init_state(stream.get(), n, states.data(), kSeed);

  random_normal(stream.get(), n, states.data(), values.data());
  ASSERT_EQ(copy(host, values, stream.get()), wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream.get()), wwrSuccess);
  const std::vector<double> first(host.data(), host.data() + n);

  random_normal(stream.get(), n, states.data(), values.data());
  ASSERT_EQ(copy(host, values, stream.get()), wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream.get()), wwrSuccess);
  const std::vector<double> second(host.data(), host.data() + n);

  EXPECT_NE(first, second);
}

// sequence_offset carves a disjoint block of subsequences out of one seed, so
// two blocks of the same seed must not coincide.
TEST(RandTests, SequenceOffsetShiftsTheStreams) {
  const auto base = draw<double>(1000, kSeed, 0);
  const auto shifted = draw<double>(1000, kSeed, 1000);
  EXPECT_NE(base, shifted);
}

// Both entry points take count == 0 as a no-op, and must not launch anything
// or touch the buffer.
TEST(RandTests, ZeroCountIsANoOp) {
  auto handle = std::make_shared<DeviceHandle>(0);
  GpuStreamWrapper<> &stream = handle->alloc_stream();
  DeviceBufferWrapper<wwrrandState> states(4, handle);
  DeviceBufferWrapper<double> values(4, handle);

  ASSERT_EQ(memset(values, 0, stream.get()), wwrSuccess);

  init_state(stream.get(), 0, states.data(), kSeed);
  random_normal(stream.get(), 0, states.data(), values.data());
  ASSERT_EQ(wwrStreamSynchronize(stream.get()), wwrSuccess);

  HostBufferWrapper<double> host(4);
  ASSERT_EQ(copy(host, values, stream.get()), wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream.get()), wwrSuccess);

  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(host[i], 0.0) << "at index " << i;
  }
}

} // namespace wwr::extension::test
