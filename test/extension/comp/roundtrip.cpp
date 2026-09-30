// roundtrip.cpp - Device round-trip tests for wwr.comp
//
// Unlike the other wwr* modules, wwr.comp's exported functions are hand-written
// forwarding shims over two version-skewed vendor libraries (nvCOMP 5.3 /
// hipCOMP 2.2), so a WWR_SAME_FUNCTION address check cannot prove them (there is
// no single backend function they equal). The proof is behavioural: compress
// then decompress a buffer on the device through the neutral API and assert it
// round-trips, for each of the three algorithms both backends implement.
//
// A plain TU: GTest::gtest_main supplies main(); the tests share no state. Every
// suite is REQUIRES_GPU -- the batched LLIF runs on a real device.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;
import wwr.comp;

namespace wwr::test {

// One batch of one chunk: the smallest shape that exercises the whole batched
// LLIF (chunk-pointer arrays, temp buffer, per-chunk size read-back). The six
// neutral entry points share one signature across algorithms, so the round-trip
// is written once and instantiated per algorithm with its own opts type.
template <class Opts>
void round_trip(
    const std::vector<unsigned char> &input, Opts opts,
    wwrcompStatus_t (*compress_temp)(std::size_t, std::size_t, Opts, std::size_t *),
    wwrcompStatus_t (*max_out)(std::size_t, Opts, std::size_t *),
    wwrcompStatus_t (*compress)(const void *const *, const std::size_t *, std::size_t, std::size_t,
                                void *, std::size_t, void *const *, std::size_t *, Opts,
                                wwrcompStream_t),
    wwrcompStatus_t (*decompress_temp)(std::size_t, std::size_t, std::size_t *),
    wwrcompStatus_t (*decompress)(const void *const *, const std::size_t *, const std::size_t *,
                                  std::size_t *, std::size_t, void *, std::size_t, void *const *,
                                  wwrcompStatus_t *, wwrcompStream_t)) {
  const std::size_t n = input.size();
  const std::size_t batch = 1;

  wwrStream_t stream{};
  ASSERT_EQ(wwrStreamCreate(&stream), wwrSuccess);

  // Uncompressed data + its chunk-pointer / chunk-size arrays, all on device.
  void *d_input = nullptr;
  ASSERT_EQ(wwrMalloc(&d_input, n), wwrSuccess);
  ASSERT_EQ(wwrMemcpy(d_input, input.data(), n, wwrMemcpyHostToDevice), wwrSuccess);

  void *d_unc_ptrs = nullptr;
  ASSERT_EQ(wwrMalloc(&d_unc_ptrs, sizeof(void *)), wwrSuccess);
  {
    void *h[1] = {d_input};
    ASSERT_EQ(wwrMemcpy(d_unc_ptrs, h, sizeof h, wwrMemcpyHostToDevice), wwrSuccess);
  }
  void *d_unc_bytes = nullptr;
  ASSERT_EQ(wwrMalloc(&d_unc_bytes, sizeof(std::size_t)), wwrSuccess);
  {
    std::size_t h[1] = {n};
    ASSERT_EQ(wwrMemcpy(d_unc_bytes, h, sizeof h, wwrMemcpyHostToDevice), wwrSuccess);
  }

  std::size_t max_comp = 0;
  ASSERT_EQ(max_out(n, opts, &max_comp), WWRCOMP_SUCCESS);
  std::size_t comp_temp_bytes = 0;
  ASSERT_EQ(compress_temp(batch, n, opts, &comp_temp_bytes), WWRCOMP_SUCCESS);

  void *d_comp = nullptr;
  ASSERT_EQ(wwrMalloc(&d_comp, max_comp), wwrSuccess);
  void *d_comp_ptrs = nullptr;
  ASSERT_EQ(wwrMalloc(&d_comp_ptrs, sizeof(void *)), wwrSuccess);
  {
    void *h[1] = {d_comp};
    ASSERT_EQ(wwrMemcpy(d_comp_ptrs, h, sizeof h, wwrMemcpyHostToDevice), wwrSuccess);
  }
  void *d_comp_bytes = nullptr;
  ASSERT_EQ(wwrMalloc(&d_comp_bytes, sizeof(std::size_t)), wwrSuccess);
  void *d_comp_temp = nullptr;
  if (comp_temp_bytes)
    ASSERT_EQ(wwrMalloc(&d_comp_temp, comp_temp_bytes), wwrSuccess);

  ASSERT_EQ(compress(static_cast<const void *const *>(d_unc_ptrs),
                     static_cast<const std::size_t *>(d_unc_bytes), n, batch, d_comp_temp,
                     comp_temp_bytes, static_cast<void *const *>(d_comp_ptrs),
                     static_cast<std::size_t *>(d_comp_bytes), opts, stream),
            WWRCOMP_SUCCESS);
  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);

  std::size_t comp_size = 0;
  ASSERT_EQ(wwrMemcpy(&comp_size, d_comp_bytes, sizeof comp_size, wwrMemcpyDeviceToHost),
            wwrSuccess);
  EXPECT_GT(comp_size, 0u);
  EXPECT_LE(comp_size, max_comp);

  // Decompress back into a fresh buffer; the compressed sizes stay on device.
  std::size_t decomp_temp_bytes = 0;
  ASSERT_EQ(decompress_temp(batch, n, &decomp_temp_bytes), WWRCOMP_SUCCESS);
  void *d_decomp_temp = nullptr;
  if (decomp_temp_bytes)
    ASSERT_EQ(wwrMalloc(&d_decomp_temp, decomp_temp_bytes), wwrSuccess);

  void *d_out = nullptr;
  ASSERT_EQ(wwrMalloc(&d_out, n), wwrSuccess);
  void *d_out_ptrs = nullptr;
  ASSERT_EQ(wwrMalloc(&d_out_ptrs, sizeof(void *)), wwrSuccess);
  {
    void *h[1] = {d_out};
    ASSERT_EQ(wwrMemcpy(d_out_ptrs, h, sizeof h, wwrMemcpyHostToDevice), wwrSuccess);
  }
  void *d_actual = nullptr;
  ASSERT_EQ(wwrMalloc(&d_actual, sizeof(std::size_t)), wwrSuccess);
  void *d_status = nullptr;
  ASSERT_EQ(wwrMalloc(&d_status, sizeof(wwrcompStatus_t)), wwrSuccess);

  ASSERT_EQ(decompress(static_cast<const void *const *>(d_comp_ptrs),
                       static_cast<const std::size_t *>(d_comp_bytes),
                       static_cast<const std::size_t *>(d_unc_bytes),
                       static_cast<std::size_t *>(d_actual), batch, d_decomp_temp, decomp_temp_bytes,
                       static_cast<void *const *>(d_out_ptrs),
                       static_cast<wwrcompStatus_t *>(d_status), stream),
            WWRCOMP_SUCCESS);
  ASSERT_EQ(wwrStreamSynchronize(stream), wwrSuccess);

  std::size_t actual = 0;
  ASSERT_EQ(wwrMemcpy(&actual, d_actual, sizeof actual, wwrMemcpyDeviceToHost), wwrSuccess);
  EXPECT_EQ(actual, n);

  std::vector<unsigned char> out(n, 0);
  ASSERT_EQ(wwrMemcpy(out.data(), d_out, n, wwrMemcpyDeviceToHost), wwrSuccess);
  EXPECT_EQ(out, input);

  wwrFree(d_input);
  wwrFree(d_unc_ptrs);
  wwrFree(d_unc_bytes);
  wwrFree(d_comp);
  wwrFree(d_comp_ptrs);
  wwrFree(d_comp_bytes);
  if (d_comp_temp)
    wwrFree(d_comp_temp);
  if (d_decomp_temp)
    wwrFree(d_decomp_temp);
  wwrFree(d_out);
  wwrFree(d_out_ptrs);
  wwrFree(d_actual);
  wwrFree(d_status);
  wwrStreamDestroy(stream);
}

// A compressible payload: runs of identical bytes so every algorithm shrinks it.
std::vector<unsigned char> payload() {
  std::vector<unsigned char> v(4096);
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = static_cast<unsigned char>((i / 64) & 0xff);
  return v;
}

TEST(CompRoundTrip, LZ4) {
  round_trip<wwrcompBatchedLZ4Opts>(
      payload(), {WWRCOMP_TYPE_CHAR}, &wwrcompBatchedLZ4CompressGetTempSize,
      &wwrcompBatchedLZ4CompressGetMaxOutputChunkSize, &wwrcompBatchedLZ4CompressAsync,
      &wwrcompBatchedLZ4DecompressGetTempSize, &wwrcompBatchedLZ4DecompressAsync);
}

TEST(CompRoundTrip, Snappy) {
  round_trip<wwrcompBatchedSnappyOpts>(
      payload(), {}, &wwrcompBatchedSnappyCompressGetTempSize,
      &wwrcompBatchedSnappyCompressGetMaxOutputChunkSize, &wwrcompBatchedSnappyCompressAsync,
      &wwrcompBatchedSnappyDecompressGetTempSize, &wwrcompBatchedSnappyDecompressAsync);
}

TEST(CompRoundTrip, Cascaded) {
  round_trip<wwrcompBatchedCascadedOpts>(
      payload(), {4096, WWRCOMP_TYPE_CHAR, 2, 1, 1}, &wwrcompBatchedCascadedCompressGetTempSize,
      &wwrcompBatchedCascadedCompressGetMaxOutputChunkSize, &wwrcompBatchedCascadedCompressAsync,
      &wwrcompBatchedCascadedDecompressGetTempSize, &wwrcompBatchedCascadedDecompressAsync);
}

} // namespace wwr::test
