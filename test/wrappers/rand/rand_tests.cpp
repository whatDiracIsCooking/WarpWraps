// rand_tests.cpp - Runtime tests for wwr.wrappers.rand
//
// The typed host-generation wrappers (generate_uniform / generate_normal /
// generate_lognormal) are token-paste forwarders onto wwrrandGenerate*; the
// build-time rand_dispatch.toml check already proves each instantiation pastes
// the right name. What that check cannot do is run the pasted call against a
// live cuRAND / hipRAND generator -- so this suite does exactly that: it creates
// a generator, seeds it, generates into a device buffer for both float and
// double, and asserts the returned wwrrandStatus_t is success. A coarse
// copied-back sanity check guards against an all-zeros / non-finite result; it
// is deliberately not a statistical or golden-value test (cuRAND and hipRAND do
// not share a stream for a given seed -- see test/extension/rand/rand_tests.cpp
// for that reasoning).
//
// A plain .cpp, not a .cppm -- GoogleTest headers and `import std;` coexist in
// an ordinary TU, but not inside a module interface unit's global module
// fragment (same reasoning as every other test/ suite).

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;
import wwr.rand;
import wwr.wrappers.rand;
import wwr.extension.runtime;
import wwr.extension.memory_buffer;
import wwr.test.shared.device_handle;
import wwr.test.shared.abort_policy;

namespace wwr::extension::test {
namespace {

using Abort = AbortPolicy<wwrError_t>;

constexpr std::size_t kCount = 4096;
constexpr unsigned long long kSeed = 20251001;

/// @brief A seeded pseudo-random generator, destroyed at scope exit.
class Generator {
public:
  Generator() {
    EXPECT_EQ(wwrrandCreateGenerator(&gen_, WWRRAND_RNG_PSEUDO_DEFAULT), WWRRAND_STATUS_SUCCESS);
    EXPECT_EQ(wwrrandSetPseudoRandomGeneratorSeed(gen_, kSeed), WWRRAND_STATUS_SUCCESS);
  }
  ~Generator() { wwrrandDestroyGenerator(gen_); }
  Generator(const Generator &) = delete;
  Generator &operator=(const Generator &) = delete;

  wwrrandGenerator_t get() const { return gen_; }

private:
  wwrrandGenerator_t gen_{};
};

/// @brief Generate `kCount` values with `fn`, copy them back, and assert the
/// status was success and the sample is finite and not all-zero.
template<typename T, typename Fn>
void run_and_check(Fn &&fn) {
  auto handle = std::make_shared<DeviceHandle>(0);
  DeviceBufferWrapper<T, Abort, Abort, Abort, DeviceHandle> values(kCount, handle);

  Generator gen;
  EXPECT_EQ(std::forward<Fn>(fn)(gen.get(), values.data(), kCount), WWRRAND_STATUS_SUCCESS);

  std::vector<T> host(kCount);
  EXPECT_EQ(wwrMemcpy(host.data(), values.data(), kCount * sizeof(T), wwrMemcpyDeviceToHost),
            wwrSuccess);

  EXPECT_TRUE(std::ranges::all_of(host, [](T v) { return std::isfinite(static_cast<double>(v)); }))
      << "sample contains a non-finite value";
  EXPECT_TRUE(std::ranges::any_of(host, [](T v) { return v != T{0}; }))
      << "sample is all zeros -- generator produced nothing";
}

} // namespace

TEST(WrappersRandTests, GenerateUniformFloat) {
  run_and_check<float>([](wwrrandGenerator_t g, float *out, std::size_t n) {
    return generate_uniform<float>(g, out, n);
  });
}

TEST(WrappersRandTests, GenerateUniformDouble) {
  run_and_check<double>([](wwrrandGenerator_t g, double *out, std::size_t n) {
    return generate_uniform<double>(g, out, n);
  });
}

TEST(WrappersRandTests, GenerateNormalFloat) {
  run_and_check<float>([](wwrrandGenerator_t g, float *out, std::size_t n) {
    return generate_normal<float>(g, out, n, 0.0f, 1.0f);
  });
}

TEST(WrappersRandTests, GenerateNormalDouble) {
  run_and_check<double>([](wwrrandGenerator_t g, double *out, std::size_t n) {
    return generate_normal<double>(g, out, n, 0.0, 1.0);
  });
}

TEST(WrappersRandTests, GenerateLogNormalFloat) {
  run_and_check<float>([](wwrrandGenerator_t g, float *out, std::size_t n) {
    return generate_lognormal<float>(g, out, n, 0.0f, 1.0f);
  });
}

TEST(WrappersRandTests, GenerateLogNormalDouble) {
  run_and_check<double>([](wwrrandGenerator_t g, double *out, std::size_t n) {
    return generate_lognormal<double>(g, out, n, 0.0, 1.0);
  });
}

} // namespace wwr::extension::test
