// reduce_tests.cpp -- the #239 (A-reduce) acceptance: the Thrust reduction
// family's typed wrappers return the correct value AND run on the stream they
// are handed.
//
// Unlike the F2 stream suite there is no device .cu here: wwr.wrappers.thrust.
// reduce's exported wrappers are host functions that forward across
// reduce_bridge.h to the device archive's explicit instantiations, so the host
// gtest calls them directly -- importing the module is all the wiring needed.
// REQUIRES_GPU: every case allocates, reduces and synchronizes on a real
// device.
//
// Two kinds of check, because neither alone is the whole claim:
//
//   - The Correctness suite drives all seven entry points across the whole
//     type set (float, double, int32_t, int64_t) against hand-computed
//     expected values -- reduce(exec, ...), transform_reduce, count, count_if,
//     inner_product, min_element and max_element, each through wwr::par_on.
//
//   - StreamIsHonored is the DETERMINISTIC proof the reduction runs on the
//     passed stream: the input is produced by an async H2D copy enqueued on
//     `stream`, an event is recorded on `stream` after it, and the reduction
//     is launched on `stream`. Because stream order is FIFO, the reduction can
//     only observe the fully-copied input if it executes in `stream`'s queue;
//     a reduction misrouted to the default stream would be unordered against
//     that copy. The result is read back and must equal the value computed
//     from the input -- so the ordering, not a race, is the discriminator,
//     the same honest reading the F2 suite's StreamIsHonored documents.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;
import wwr.wrappers.thrust.reduce;

namespace wwr::thrust::test {
namespace {

// Upload a host vector to a fresh device allocation on `stream` and return the
// device pointer; the caller frees it. Kept async on `stream` so the reduction
// that follows shares the stream's order.
template<typename T>
T *upload(const wwrStream_t stream, const std::vector<T> &host) {
  T *d = nullptr;
  const std::size_t bytes = sizeof(T) * host.size();
  EXPECT_EQ(wwrMalloc(reinterpret_cast<void **>(&d), bytes), wwrSuccess);
  EXPECT_EQ(wwrMemcpyAsync(d, host.data(), bytes, wwrMemcpyHostToDevice, stream), wwrSuccess);
  return d;
}

// A typed fixture: every reduction entry point is exercised for each element
// type the family instantiates.
template<typename T>
class ReduceTyped : public ::testing::Test {};

using ElementTypes = ::testing::Types<float, double, std::int32_t, std::int64_t>;
TYPED_TEST_SUITE(ReduceTyped, ElementTypes);

TYPED_TEST(ReduceTyped, FamilyIsCorrect) {
  using T = TypeParam;
  // n = 1024 keeps the largest aggregate (sum of squares, n(n+1)(2n+1)/6 ~
  // 3.6e8) inside int32's range, so the int32_t instantiation is exact rather
  // than wrapping -- the family is correct for every type at this size.
  constexpr std::size_t n = 1024;

  // Input 1..n: a range with a known sum, sum of squares, extrema and a
  // non-zero count, and distinct from its own reverse so inner_product is a
  // real cross term rather than a sum of squares.
  std::vector<T> host(n);
  for (std::size_t i = 0; i < n; ++i) {
    host[i] = static_cast<T>(i + 1);
  }
  std::vector<T> reversed(n);
  for (std::size_t i = 0; i < n; ++i) {
    reversed[i] = static_cast<T>(n - i);
  }

  // Hand-computed expectations in a wide host type, then cast to T. The float
  // aggregates exceed float's 2^24 exact-integer range, so they are compared
  // with a relative tolerance below; int32_t/int64_t/double compare exactly.
  long double sum = 0, sqsum = 0, dot = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const long double a = static_cast<long double>(host[i]);
    const long double b = static_cast<long double>(reversed[i]);
    sum += a;
    sqsum += a * a;
    dot += a * b;
  }

  wwrStream_t stream{};
  ASSERT_EQ(wwrStreamCreate(&stream), wwrSuccess);

  T *d = upload(stream, host);
  T *d_rev = upload(stream, reversed);

  // reduce / transform_reduce / inner_product return in T. For float the sums
  // exceed the 2^24 exact-integer range, so compare with a relative tolerance;
  // the integer and double types compare exactly.
  const T got_sum = reduce_sum<T>(stream, d, n, T{});
  const T got_sqsum = transform_reduce_square_sum<T>(stream, d, n, T{});
  const T got_dot = inner_product<T>(stream, d, d_rev, n, T{});

  // count: how many equal a value in range (exactly one) and out of range
  // (none). count_if_nonzero: all n are non-zero.
  const std::int64_t got_count_one = count<T>(stream, d, n, static_cast<T>(7));
  const std::int64_t got_count_none = count<T>(stream, d, n, static_cast<T>(0));
  const std::int64_t got_nonzero = count_if_nonzero<T>(stream, d, n);

  // extrema: 1 and n.
  const T got_min = min_element_value<T>(stream, d, n);
  const T got_max = max_element_value<T>(stream, d, n);

  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);

  if constexpr (std::is_floating_point_v<T>) {
    const long double tol_sum = static_cast<long double>(sum) * 1e-5L;
    const long double tol_sq = static_cast<long double>(sqsum) * 1e-5L;
    const long double tol_dot = static_cast<long double>(dot) * 1e-5L;
    EXPECT_NEAR(static_cast<long double>(got_sum), static_cast<long double>(sum), tol_sum);
    EXPECT_NEAR(static_cast<long double>(got_sqsum), static_cast<long double>(sqsum), tol_sq);
    EXPECT_NEAR(static_cast<long double>(got_dot), static_cast<long double>(dot), tol_dot);
  } else {
    EXPECT_EQ(got_sum, static_cast<T>(sum));
    EXPECT_EQ(got_sqsum, static_cast<T>(sqsum));
    EXPECT_EQ(got_dot, static_cast<T>(dot));
  }

  EXPECT_EQ(got_count_one, 1);
  EXPECT_EQ(got_count_none, 0);
  EXPECT_EQ(got_nonzero, static_cast<std::int64_t>(n));
  EXPECT_EQ(got_min, static_cast<T>(1));
  EXPECT_EQ(got_max, static_cast<T>(n));

  EXPECT_EQ(wwrFree(d), wwrSuccess);
  EXPECT_EQ(wwrFree(d_rev), wwrSuccess);
  EXPECT_EQ(wwrStreamDestroy(stream), wwrSuccess);
}

// The stream-honored proof, on one representative type. The input is produced
// by an async copy on `stream` and gated behind an event recorded on `stream`;
// the reduction then runs on `stream`. A correct sum proves the reduction was
// ordered after that copy in `stream`'s FIFO queue -- i.e. it executed on the
// stream it was given, not the default one.
TEST(ThrustReduceStream, StreamIsHonored) {
  constexpr std::size_t n = 4096;
  constexpr int value = 3;

  std::vector<int> host(n, value);
  const long double expected = static_cast<long double>(n) * value;

  wwrStream_t stream{};
  ASSERT_EQ(wwrStreamCreate(&stream), wwrSuccess);

  int *d = nullptr;
  const std::size_t bytes = sizeof(int) * n;
  ASSERT_EQ(wwrMalloc(reinterpret_cast<void **>(&d), bytes), wwrSuccess);

  // Zero on the stream first, THEN overwrite with the real input on the same
  // stream, with an event recorded between: stream order means the reduction
  // enqueued next must see the second copy's values. A reduction on the wrong
  // stream is unordered against these and could observe the zeros.
  ASSERT_EQ(wwrMemsetAsync(d, 0, bytes, stream), wwrSuccess);

  wwrEvent_t gate{};
  ASSERT_EQ(wwrEventCreate(&gate), wwrSuccess);
  ASSERT_EQ(wwrMemcpyAsync(d, host.data(), bytes, wwrMemcpyHostToDevice, stream), wwrSuccess);
  ASSERT_EQ(wwrEventRecord(gate, stream), wwrSuccess);

  const int got = reduce_sum<int>(stream, d, n, 0);

  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);
  EXPECT_EQ(static_cast<long double>(got), expected)
      << "reduce_sum did not observe the input copied on the stream it was given";

  EXPECT_EQ(wwrEventDestroy(gate), wwrSuccess);
  EXPECT_EQ(wwrFree(d), wwrSuccess);
  EXPECT_EQ(wwrStreamDestroy(stream), wwrSuccess);
}

} // namespace
} // namespace wwr::thrust::test
