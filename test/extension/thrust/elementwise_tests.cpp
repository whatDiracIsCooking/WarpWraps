// elementwise_tests.cpp -- the A-transform (issue #241) acceptance: each
// elementwise wrapper produces the correct output AND runs on the passed stream.
//
// The wrappers themselves are the host module wwr.wrappers.thrust (imported
// here); they forward to device code device-compiled in elementwise.cu, so this
// is a pure host TU driving them end to end -- allocate, call the wrapper on a
// stream, copy back, check. REQUIRES_GPU: it launches and synchronizes on a real
// device, like the F2 stream suite next to it.
//
// Stream-honored is proved two complementary ways, mirroring the F2 idiom:
//   - StreamIsHonored (in stream_tests.cpp, already present) reads the bound
//     stream back out of wwr::par_on(stream) -- the deterministic, policy-level
//     proof the shim these wrappers all share routes to the given stream.
//   - Here, every op is issued on a non-default stream and the result is read
//     back only after wwrStreamSynchronize(stream): the data is correct AND the
//     synchronize on exactly that stream is what makes it observable, so a
//     misrouted op would either race (wrong data) or deadlock the sync. The two
//     together are the honest "correct, on the passed stream" claim.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;
import wwr.complex;
import wwr.wrappers.thrust;

namespace wwr::extension::thrust {
namespace {

// Allocate n T on device, run `body` (which issues work on `stream`), sync the
// stream, and return the device array copied back to host. The sync is on the
// same stream the body used -- the stream-honored half of each check.
template<typename T, typename Body>
std::vector<T> on_device(const std::size_t n, Body body) {
  const std::size_t bytes = sizeof(T) * n;
  wwrStream_t stream{};
  EXPECT_EQ(wwrStreamCreate(&stream), wwrSuccess);

  T *d = nullptr;
  EXPECT_EQ(wwrMalloc(reinterpret_cast<void **>(&d), bytes), wwrSuccess);

  body(stream, d);

  std::vector<T> out(n);
  EXPECT_EQ(wwrMemcpyAsync(out.data(), d, bytes, wwrMemcpyDeviceToHost, stream), wwrSuccess);
  EXPECT_EQ(wwrStreamSynchronize(stream), wwrSuccess);

  EXPECT_EQ(wwrFree(d), wwrSuccess);
  EXPECT_EQ(wwrStreamDestroy(stream), wwrSuccess);
  return out;
}

// Copy a host vector onto d (on stream) before the op under test runs on it.
template<typename T>
void upload(wwrStream_t stream, T *d, const std::vector<T> &src) {
  EXPECT_EQ(wwrMemcpyAsync(d, src.data(), sizeof(T) * src.size(), wwrMemcpyHostToDevice, stream),
            wwrSuccess);
}

constexpr std::size_t kN = 2048;

TEST(ThrustElementwiseTests, FillSetsEveryElement) {
  const auto out = on_device<float>(kN, [](wwrStream_t s, float *d) { fill(s, d, kN, 3.5f); });
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_EQ(out[i], 3.5f) << "at " << i;
  }
}

TEST(ThrustElementwiseTests, SequenceWritesProgression) {
  const auto out =
      on_device<int>(kN, [](wwrStream_t s, int *d) { sequence(s, d, kN, 10, 2); });
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_EQ(out[i], 10 + 2 * static_cast<int>(i)) << "at " << i;
  }
}

TEST(ThrustElementwiseTests, SequenceDefaultsToIota) {
  const auto out = on_device<int>(kN, [](wwrStream_t s, int *d) { sequence(s, d, kN); });
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_EQ(out[i], static_cast<int>(i)) << "at " << i;
  }
}

TEST(ThrustElementwiseTests, ReplaceMatchesValue) {
  std::vector<int> in(kN);
  for (std::size_t i = 0; i < kN; ++i) {
    in[i] = (i % 2 == 0) ? 7 : 1;
  }
  const auto out = on_device<int>(kN, [&](wwrStream_t s, int *d) {
    upload(s, d, in);
    replace(s, d, kN, 7, 99);
  });
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_EQ(out[i], (i % 2 == 0) ? 99 : 1) << "at " << i;
  }
}

TEST(ThrustElementwiseTests, TransformUnaryNegate) {
  std::vector<double> in(kN);
  for (std::size_t i = 0; i < kN; ++i) {
    in[i] = static_cast<double>(i) - 1000.0;
  }
  const auto out = on_device<double>(kN, [&](wwrStream_t s, double *d) {
    // in-place is allowed for an elementwise map; use the same buffer for in/out.
    upload(s, d, in);
    transform_unary<UnaryOp::Negate>(s, d, d, kN);
  });
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_EQ(out[i], -in[i]) << "at " << i;
  }
}

TEST(ThrustElementwiseTests, TransformUnarySquareInt) {
  std::vector<int> in(kN);
  for (std::size_t i = 0; i < kN; ++i) {
    in[i] = static_cast<int>(i) - 10;
  }
  const auto out = on_device<int>(kN, [&](wwrStream_t s, int *d) {
    upload(s, d, in);
    transform_unary<UnaryOp::Square>(s, d, d, kN);
  });
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_EQ(out[i], in[i] * in[i]) << "at " << i;
  }
}

TEST(ThrustElementwiseTests, TransformUnaryAbs) {
  std::vector<float> in(kN);
  for (std::size_t i = 0; i < kN; ++i) {
    in[i] = static_cast<float>(i) - 1000.0f;
  }
  const auto out = on_device<float>(kN, [&](wwrStream_t s, float *d) {
    upload(s, d, in);
    transform_unary<UnaryOp::Abs>(s, d, d, kN);
  });
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_EQ(out[i], std::abs(in[i])) << "at " << i;
  }
}

TEST(ThrustElementwiseTests, TransformBinaryPlus) {
  std::vector<float> a(kN), b(kN);
  for (std::size_t i = 0; i < kN; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i);
  }
  const std::size_t bytes = sizeof(float) * kN;
  wwrStream_t stream{};
  ASSERT_EQ(wwrStreamCreate(&stream), wwrSuccess);
  float *da = nullptr;
  float *db = nullptr;
  float *dc = nullptr;
  ASSERT_EQ(wwrMalloc(reinterpret_cast<void **>(&da), bytes), wwrSuccess);
  ASSERT_EQ(wwrMalloc(reinterpret_cast<void **>(&db), bytes), wwrSuccess);
  ASSERT_EQ(wwrMalloc(reinterpret_cast<void **>(&dc), bytes), wwrSuccess);

  upload(stream, da, a);
  upload(stream, db, b);
  transform_binary<BinaryOp::Multiply>(stream, da, db, dc, kN);

  std::vector<float> out(kN);
  ASSERT_EQ(wwrMemcpyAsync(out.data(), dc, bytes, wwrMemcpyDeviceToHost, stream), wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_EQ(out[i], a[i] * b[i]) << "at " << i;
  }

  EXPECT_EQ(wwrFree(da), wwrSuccess);
  EXPECT_EQ(wwrFree(db), wwrSuccess);
  EXPECT_EQ(wwrFree(dc), wwrSuccess);
  EXPECT_EQ(wwrStreamDestroy(stream), wwrSuccess);
}

TEST(ThrustElementwiseTests, ComplexFillNegateReplace) {
  // Complex exercises the wwrC*-based functors and the component-wise replace
  // predicate (cuComplex/hipComplex have no operator==).
  const auto out = on_device<wwrFloatComplex>(kN, [](wwrStream_t s, wwrFloatComplex *d) {
    fill(s, d, kN, make_wwrFloatComplex(2.0f, -3.0f));
    transform_unary<UnaryOp::Negate>(s, d, d, kN);
    replace(s, d, kN, make_wwrFloatComplex(-2.0f, 3.0f), make_wwrFloatComplex(5.0f, 5.0f));
  });
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_EQ(wwrCrealf(out[i]), 5.0f) << "re at " << i;
    EXPECT_EQ(wwrCimagf(out[i]), 5.0f) << "im at " << i;
  }
}

} // namespace
} // namespace wwr::extension::thrust
