// stream_tests.cpp -- the F2 acceptance: wwr::par_on(stream) selects the right
// backend Thrust policy AND the stream it is handed is the one honored.
//
// The device half lives in stream_ops.cu (a device TU that #includes the shim
// and calls thrust::sort(wwr::par_on(stream), ...)); this host TU drives it and
// checks the result, as test/extension/rand drives its device modules.
// REQUIRES_GPU: it allocates, launches and synchronizes on a real device.
//
// Two complementary checks, because neither alone is the whole claim:
//
//   - StreamIsHonored is the DETERMINISTIC proof that the stream is honored:
//     par_on(stream) builds a policy, and Thrust's own get_stream() read back
//     out of it returns exactly the handle passed in. That is precisely what
//     "the work is enqueued on the passed stream" means at the policy level,
//     and it does not depend on any timing or race. (A host-side race test is
//     unreliable here: thrust::sort synchronizes in this CCCL, so a sort
//     misrouted to the default stream still returns a correct result -- the
//     binding, not a race, is the honest discriminator.)
//
//   - SortRunsThroughTheShim is the BEHAVIOURAL half: the literal acceptance
//     call thrust::sort(wwr::par_on(stream), ...) runs on the stream and the
//     data comes back exactly sorted, so the shim's policy is a usable,
//     correct first argument to the algorithm (what README.md's audit promises
//     for every in-scope algorithm).

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;

// After the import: stream_bridge.h names wwrStream_t, which wwr.runtime_api
// exports, so the import must precede it.
#include "stream_bridge.h"

namespace wwr::test {
namespace {

TEST(ThrustStreamTests, StreamIsHonored) {
  wwrStream_t stream{};
  ASSERT_EQ(wwrStreamCreate(&stream), wwrSuccess);

  // par_on(stream) must bind the policy to exactly this stream.
  EXPECT_EQ(policy_bound_stream(stream), stream)
      << "wwr::par_on(stream) did not bind the policy to the stream it was given";

  // A fresh, distinct stream binds to itself, not to the first -- the shim
  // forwards its argument, it does not latch a stream.
  wwrStream_t other{};
  ASSERT_EQ(wwrStreamCreate(&other), wwrSuccess);
  EXPECT_EQ(policy_bound_stream(other), other);
  EXPECT_NE(policy_bound_stream(other), stream);

  EXPECT_EQ(wwrStreamDestroy(other), wwrSuccess);
  EXPECT_EQ(wwrStreamDestroy(stream), wwrSuccess);
}

TEST(ThrustStreamTests, SortRunsThroughTheShim) {
  constexpr int n = 4096;
  const std::size_t count = static_cast<std::size_t>(n);
  const std::size_t bytes = sizeof(int) * count;

  // Descending input; a correct ascending sort makes it 1, 2, 3, ... n.
  std::vector<int> data(count);
  for (int i = 0; i < n; ++i) {
    data[static_cast<std::size_t>(i)] = n - i;
  }

  wwrStream_t stream{};
  ASSERT_EQ(wwrStreamCreate(&stream), wwrSuccess);

  int *d = nullptr;
  ASSERT_EQ(wwrMalloc(reinterpret_cast<void **>(&d), bytes), wwrSuccess);
  ASSERT_EQ(wwrMemcpyAsync(d, data.data(), bytes, wwrMemcpyHostToDevice, stream), wwrSuccess);

  thrust_sort_on_stream(stream, d, count);

  std::vector<int> out(count, 0);
  ASSERT_EQ(wwrMemcpyAsync(out.data(), d, bytes, wwrMemcpyDeviceToHost, stream), wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);

  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(out[static_cast<std::size_t>(i)], i + 1) << "mismatch at index " << i;
  }

  EXPECT_EQ(wwrFree(d), wwrSuccess);
  EXPECT_EQ(wwrStreamDestroy(stream), wwrSuccess);
}

} // namespace
} // namespace wwr::test
