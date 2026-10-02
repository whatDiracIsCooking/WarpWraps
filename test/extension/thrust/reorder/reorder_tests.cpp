// reorder_tests.cpp -- acceptance for wwr.wrappers.thrust.reorder (issue #238).
//
// The module's wrappers (wwr::reorder::sort, unique, partition, remove, copy_if,
// copy_if_stencil, reverse) forward to a device-compiled TU that calls
// thrust::<algo>(wwr::par_on(stream), ...). This host suite drives each through
// the module surface -- no local .cu -- so it exercises the whole path: the
// exported template, its device instantiation, and the F2 stream policy.
// REQUIRES_GPU: every case allocates, launches and synchronizes on a real
// device, all on a caller-created non-default stream.
//
// Two things are proven per algorithm: it reorders CORRECTLY, and it runs on the
// PASSED stream -- the result is read back only after wwrStreamSynchronize(that
// stream), so a correct result after that sync is the behavioural proof the work
// was enqueued there (the deterministic get_stream() proof that par_on binds the
// policy to the stream is F2's stream_tests.cpp, which this builds on).

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;
import wwr.wrappers.thrust.reorder;

namespace wwr::reorder {
namespace {

// A device round-trip helper: upload `in`, run `op` on `stream`, download `count`
// elements, sync. Returns the downloaded prefix. `op` receives the device
// pointer and enqueues its work on `stream`; nothing is read back before the
// sync, so a correct result proves the work ran on `stream`.
template<typename T, typename Op>
std::vector<T> on_stream(const std::vector<T> &in, const std::size_t count, Op op) {
  const std::size_t bytes = sizeof(T) * in.size();

  wwrStream_t stream{};
  EXPECT_EQ(wwrStreamCreate(&stream), wwrSuccess);

  T *d = nullptr;
  EXPECT_EQ(wwrMalloc(reinterpret_cast<void **>(&d), bytes), wwrSuccess);
  EXPECT_EQ(wwrMemcpyAsync(d, in.data(), bytes, wwrMemcpyHostToDevice, stream), wwrSuccess);

  op(stream, d);

  std::vector<T> out(count, T{});
  EXPECT_EQ(wwrMemcpyAsync(out.data(), d, sizeof(T) * count, wwrMemcpyDeviceToHost, stream),
            wwrSuccess);
  EXPECT_EQ(wwrStreamSynchronize(stream), wwrSuccess);

  EXPECT_EQ(wwrFree(d), wwrSuccess);
  EXPECT_EQ(wwrStreamDestroy(stream), wwrSuccess);
  return out;
}

TEST(ThrustReorderTests, SortAscending) {
  std::vector<int> in{5, 1, 4, 2, 3, 0, 9, 7};
  const auto out = on_stream(in, in.size(), [&](wwrStream_t s, int *d) {
    sort(s, d, in.size());
  });
  EXPECT_TRUE(std::ranges::is_sorted(out));
  EXPECT_EQ(out, (std::vector<int>{0, 1, 2, 3, 4, 5, 7, 9}));
}

TEST(ThrustReorderTests, SortDoubles) {
  std::vector<double> in{2.5, -1.0, 0.0, 3.25, 2.5};
  const auto out = on_stream(in, in.size(), [&](wwrStream_t s, double *d) {
    sort(s, d, in.size());
  });
  EXPECT_EQ(out, (std::vector<double>{-1.0, 0.0, 2.5, 2.5, 3.25}));
}

TEST(ThrustReorderTests, UniqueDropsConsecutiveDuplicates) {
  // Sorted input, so unique is a global dedup.
  std::vector<int> in{1, 1, 2, 3, 3, 3, 4};
  std::size_t kept = 0;
  const auto out = on_stream(in, in.size(), [&](wwrStream_t s, int *d) {
    kept = unique(s, d, in.size());
  });
  EXPECT_EQ(kept, 4u);
  EXPECT_EQ(std::vector<int>(out.begin(), out.begin() + 4), (std::vector<int>{1, 2, 3, 4}));
}

TEST(ThrustReorderTests, PartitionNonzeroFirst) {
  std::vector<int> in{0, 1, 0, 2, 3, 0, 4};
  std::size_t point = 0;
  const auto out = on_stream(in, in.size(), [&](wwrStream_t s, int *d) {
    point = partition(s, d, in.size());
  });
  EXPECT_EQ(point, 4u); // four nonzero elements
  for (std::size_t i = 0; i < point; ++i) {
    EXPECT_NE(out[i], 0) << "nonzero expected before the partition point at " << i;
  }
  for (std::size_t i = point; i < out.size(); ++i) {
    EXPECT_EQ(out[i], 0) << "zero expected after the partition point at " << i;
  }
}

TEST(ThrustReorderTests, RemoveValue) {
  std::vector<int> in{7, 0, 7, 1, 7, 2};
  std::size_t kept = 0;
  const auto out = on_stream(in, in.size(), [&](wwrStream_t s, int *d) {
    kept = remove(s, d, in.size(), 7);
  });
  EXPECT_EQ(kept, 3u);
  EXPECT_EQ(std::vector<int>(out.begin(), out.begin() + 3), (std::vector<int>{0, 1, 2}));
}

TEST(ThrustReorderTests, CopyIfNonzero) {
  std::vector<int> src{0, 5, 0, 6, 7, 0};
  const std::size_t bytes = sizeof(int) * src.size();

  wwrStream_t stream{};
  ASSERT_EQ(wwrStreamCreate(&stream), wwrSuccess);
  int *d_src = nullptr;
  int *d_dst = nullptr;
  ASSERT_EQ(wwrMalloc(reinterpret_cast<void **>(&d_src), bytes), wwrSuccess);
  ASSERT_EQ(wwrMalloc(reinterpret_cast<void **>(&d_dst), bytes), wwrSuccess);
  ASSERT_EQ(wwrMemcpyAsync(d_src, src.data(), bytes, wwrMemcpyHostToDevice, stream), wwrSuccess);

  const std::size_t written = copy_if(stream, d_src, src.size(), d_dst);
  EXPECT_EQ(written, 3u);

  std::vector<int> out(3, 0);
  ASSERT_EQ(wwrMemcpyAsync(out.data(), d_dst, sizeof(int) * 3, wwrMemcpyDeviceToHost, stream),
            wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);
  EXPECT_EQ(out, (std::vector<int>{5, 6, 7}));

  EXPECT_EQ(wwrFree(d_src), wwrSuccess);
  EXPECT_EQ(wwrFree(d_dst), wwrSuccess);
  EXPECT_EQ(wwrStreamDestroy(stream), wwrSuccess);
}

TEST(ThrustReorderTests, CopyIfStencil) {
  std::vector<int> src{10, 11, 12, 13};
  std::vector<int> stencil{0, 1, 0, 1};
  const std::size_t bytes = sizeof(int) * src.size();

  wwrStream_t stream{};
  ASSERT_EQ(wwrStreamCreate(&stream), wwrSuccess);
  int *d_src = nullptr;
  int *d_stencil = nullptr;
  int *d_dst = nullptr;
  ASSERT_EQ(wwrMalloc(reinterpret_cast<void **>(&d_src), bytes), wwrSuccess);
  ASSERT_EQ(wwrMalloc(reinterpret_cast<void **>(&d_stencil), bytes), wwrSuccess);
  ASSERT_EQ(wwrMalloc(reinterpret_cast<void **>(&d_dst), bytes), wwrSuccess);
  ASSERT_EQ(wwrMemcpyAsync(d_src, src.data(), bytes, wwrMemcpyHostToDevice, stream), wwrSuccess);
  ASSERT_EQ(wwrMemcpyAsync(d_stencil, stencil.data(), bytes, wwrMemcpyHostToDevice, stream),
            wwrSuccess);

  const std::size_t written = copy_if_stencil(stream, d_src, d_stencil, src.size(), d_dst);
  EXPECT_EQ(written, 2u);

  std::vector<int> out(2, 0);
  ASSERT_EQ(wwrMemcpyAsync(out.data(), d_dst, sizeof(int) * 2, wwrMemcpyDeviceToHost, stream),
            wwrSuccess);
  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);
  EXPECT_EQ(out, (std::vector<int>{11, 13})); // src[i] where stencil[i] != 0

  EXPECT_EQ(wwrFree(d_src), wwrSuccess);
  EXPECT_EQ(wwrFree(d_stencil), wwrSuccess);
  EXPECT_EQ(wwrFree(d_dst), wwrSuccess);
  EXPECT_EQ(wwrStreamDestroy(stream), wwrSuccess);
}

TEST(ThrustReorderTests, Reverse) {
  std::vector<int> in{1, 2, 3, 4, 5};
  const auto out = on_stream(in, in.size(), [&](wwrStream_t s, int *d) {
    reverse(s, d, in.size());
  });
  EXPECT_EQ(out, (std::vector<int>{5, 4, 3, 2, 1}));
}

TEST(ThrustReorderTests, UnsignedAndWideTypes) {
  // Exercises an instantiation beyond int/double: uint64_t sort on a stream.
  std::vector<std::uint64_t> in{1ull << 40, 3, 1ull << 50, 2};
  const auto out = on_stream(in, in.size(), [&](wwrStream_t s, std::uint64_t *d) {
    sort(s, d, in.size());
  });
  EXPECT_TRUE(std::ranges::is_sorted(out));
}

} // namespace
} // namespace wwr::reorder
