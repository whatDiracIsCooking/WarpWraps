// scan_tests.cpp -- the A-scan (issue #240) acceptance: the typed
// wwr.wrappers.thrust.scan wrappers produce the correct prefix sequence AND run
// on the stream they are handed.
//
// The scan work lives inside the module's device library (scan.cu); this is an
// ordinary host CXX TU that imports the module and drives it, exactly as
// test/extension/rand drives init_state/random_normal. REQUIRES_GPU: it
// allocates, launches and synchronizes on a real device.
//
// How "runs on the passed stream" is proven here, extending the F2 idiom
// (test/extension/thrust/stream_tests.cpp): F2 already showed, deterministically,
// that wwr::par_on(stream) binds the policy to exactly the stream passed. These
// scans forward that same stream, so the policy binding is covered there. What
// this suite adds is the BEHAVIOURAL half per algorithm -- each scan is enqueued
// on a non-default stream, the device->host copy is enqueued on the SAME stream
// with no intervening sync, and only that stream is synchronized. A result that
// comes back fully and correctly therefore proves the scan completed, ordered,
// on that stream (had it misrouted to another stream, the same-stream copy would
// not be ordered after it). Correctness and stream-honoring are the one check.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;
import wwr.wrappers.thrust.scan;

namespace wwr::thrust::test {
namespace {

// Allocate n T on device, run `scan` (a lambda taking stream + device ptrs) on a
// fresh non-default stream with the D2H copy ordered on the same stream, and
// return what comes back. The scan and the copy share one stream and only that
// stream is synced -- the behavioural stream-honoring check (see file header).
template<typename T, typename Scan>
std::vector<T> run_on_stream(const std::vector<T> &input, Scan scan) {
  const std::size_t n = input.size();
  const std::size_t bytes = sizeof(T) * n;

  wwrStream_t stream{};
  EXPECT_EQ(wwrStreamCreate(&stream), wwrSuccess);

  T *d_in = nullptr;
  T *d_out = nullptr;
  EXPECT_EQ(wwrMalloc(reinterpret_cast<void **>(&d_in), bytes), wwrSuccess);
  EXPECT_EQ(wwrMalloc(reinterpret_cast<void **>(&d_out), bytes), wwrSuccess);
  EXPECT_EQ(wwrMemcpyAsync(d_in, input.data(), bytes, wwrMemcpyHostToDevice, stream), wwrSuccess);

  scan(stream, d_in, n, d_out);

  std::vector<T> out(n, T{});
  EXPECT_EQ(wwrMemcpyAsync(out.data(), d_out, bytes, wwrMemcpyDeviceToHost, stream), wwrSuccess);
  EXPECT_EQ(wwrStreamSynchronize(stream), wwrSuccess);

  EXPECT_EQ(wwrFree(d_in), wwrSuccess);
  EXPECT_EQ(wwrFree(d_out), wwrSuccess);
  EXPECT_EQ(wwrStreamDestroy(stream), wwrSuccess);
  return out;
}

// 1, 2, 3, ..., n as the input across the numeric tests -- a prefix sum of it is
// the triangular numbers, exactly representable in every type used here at this
// size.
template<typename T>
std::vector<T> iota(const std::size_t n) {
  std::vector<T> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = static_cast<T>(i + 1);
  }
  return v;
}

constexpr std::size_t kN = 2048;

TEST(ThrustScanTests, InclusiveSumFloat) {
  const auto in = iota<float>(kN);
  const auto out = run_on_stream<float>(in, [](wwrStream_t s, const float *di, std::size_t n,
                                               float *dout) { inclusive_scan(s, di, n, dout); });
  // out[i] = sum(1..i+1) = (i+1)(i+2)/2.
  for (std::size_t i = 0; i < kN; ++i) {
    const auto k = static_cast<double>(i + 1);
    EXPECT_FLOAT_EQ(out[i], static_cast<float>(k * (k + 1) / 2.0)) << "at " << i;
  }
}

TEST(ThrustScanTests, InclusiveSumInt) {
  const auto in = iota<int>(kN);
  const auto out = run_on_stream<int>(in, [](wwrStream_t s, const int *di, std::size_t n,
                                             int *dout) { inclusive_scan(s, di, n, dout); });
  int running = 0;
  for (std::size_t i = 0; i < kN; ++i) {
    running += static_cast<int>(i + 1);
    EXPECT_EQ(out[i], running) << "at " << i;
  }
}

TEST(ThrustScanTests, ExclusiveSumInt64) {
  const auto in = iota<std::int64_t>(kN);
  const std::int64_t init = 100;
  const auto out =
      run_on_stream<std::int64_t>(in, [init](wwrStream_t s, const std::int64_t *di, std::size_t n,
                                             std::int64_t *dout) {
        exclusive_scan(s, di, n, dout, init);
      });
  // out[0] = init; out[i] = init + sum(1..i).
  std::int64_t running = init;
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_EQ(out[i], running) << "at " << i;
    running += static_cast<std::int64_t>(i + 1);
  }
}

TEST(ThrustScanTests, InclusiveMaxCustomOp) {
  // The custom-associative-operator acceptance: a running maximum. A sawtooth so
  // the prefix max is a clear staircase, not just a monotone copy of the input.
  std::vector<int> in(kN);
  for (std::size_t i = 0; i < kN; ++i) {
    in[i] = static_cast<int>((i * 7 + 3) % 101);
  }
  const auto out = run_on_stream<int>(in, [](wwrStream_t s, const int *di, std::size_t n,
                                             int *dout) {
    inclusive_scan(s, di, n, dout, Op::max);
  });
  int running = std::numeric_limits<int>::min();
  for (std::size_t i = 0; i < kN; ++i) {
    running = std::max(running, in[i]);
    EXPECT_EQ(out[i], running) << "at " << i;
  }
}

TEST(ThrustScanTests, TransformInclusiveNegateDouble) {
  const auto in = iota<double>(kN);
  const auto out =
      run_on_stream<double>(in, [](wwrStream_t s, const double *di, std::size_t n, double *dout) {
        transform_inclusive_scan_negate(s, di, n, dout);
      });
  // negate then inclusive sum: out[i] = -(1+2+...+(i+1)).
  for (std::size_t i = 0; i < kN; ++i) {
    const auto k = static_cast<double>(i + 1);
    EXPECT_DOUBLE_EQ(out[i], -(k * (k + 1) / 2.0)) << "at " << i;
  }
}

TEST(ThrustScanTests, TransformExclusiveNegateFloat) {
  const auto in = iota<float>(kN);
  const float init = 5.0F;
  const auto out =
      run_on_stream<float>(in, [init](wwrStream_t s, const float *di, std::size_t n, float *dout) {
        transform_exclusive_scan_negate(s, di, n, dout, init);
      });
  // out[0] = init; out[i] = init - (1+2+...+i).
  float running = init;
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(out[i], running) << "at " << i;
    running -= static_cast<float>(i + 1);
  }
}

TEST(ThrustScanTests, InPlaceInclusiveSum) {
  // out may alias in -- Thrust's inclusive_scan supports in-place. Allocate one
  // buffer and scan it onto itself.
  const auto in = iota<int>(kN);
  const std::size_t bytes = sizeof(int) * kN;

  wwrStream_t stream{};
  ASSERT_EQ(wwrStreamCreate(&stream), wwrSuccess);
  int *d = nullptr;
  ASSERT_EQ(wwrMalloc(reinterpret_cast<void **>(&d), bytes), wwrSuccess);
  ASSERT_EQ(wwrMemcpyAsync(d, in.data(), bytes, wwrMemcpyHostToDevice, stream), wwrSuccess);

  inclusive_scan(stream, d, kN, d); // in-place: out == in

  std::vector<int> out(kN, 0);
  ASSERT_EQ(wwrMemcpyAsync(out.data(), d, bytes, wwrMemcpyDeviceToHost, stream), wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);

  int running = 0;
  for (std::size_t i = 0; i < kN; ++i) {
    running += static_cast<int>(i + 1);
    EXPECT_EQ(out[i], running) << "at " << i;
  }
  EXPECT_EQ(wwrFree(d), wwrSuccess);
  EXPECT_EQ(wwrStreamDestroy(stream), wwrSuccess);
}

} // namespace
} // namespace wwr::thrust::test
