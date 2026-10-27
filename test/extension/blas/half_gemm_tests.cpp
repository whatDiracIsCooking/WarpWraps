// half_gemm_tests.cpp - wwrblasHgemm / wwrblasGemmEx numerics against a double reference
//
// The compile-time half (test/gpu/blas.cppm) pins the neutral signatures; this
// runs them. On HIP the Hgemm family is a hand-written forwarder that
// reinterprets wwrHalf pointers as hipblasHalf, and on CUDA GemmEx is a typed
// reference picking one of an overload pair -- a wrong cast or a wrong overload
// still compiles, so only a product checked against a reference shows the
// binding is right.
//
// Inputs are drawn in [-1, 1] and rounded to half, and the reference is the
// exact (double) product of those rounded values, so the only error measured is
// the GEMM's own. Each entry must sit inside the worst-case summation bound
// k * u * sum_l |a_il b_lj| for the accumulation precision u (plus one rounding
// of the result when C is half). Hgemm is held to the fp16 bound: cuBLAS really
// accumulates in fp16, hipBLAS in fp32, and both must pass. A per-entry bound
// alone would pass an all-zero C wherever the reference is small, so each case
// also bounds the relative Frobenius error.
//
// Runtime, device-requiring. Backend-neutral -- built and run for either
// WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;
import wwr.blas;
import wwr.fp16;

namespace wwr::extension::test {
namespace {

constexpr int kM = 64;
constexpr int kN = 48;
constexpr int kK = 96;
constexpr double kUnitRoundoff16 = 0x1p-11;
// fp32 accumulation, with headroom for tensor cores, which align products by
// truncation rather than rounding before they add.
constexpr double kUnitRoundoff32 = 16 * 0x1p-24;

template<typename T>
class DeviceArray {
 public:
  explicit DeviceArray(std::size_t n) : n_(n) {
    EXPECT_EQ(wwrMalloc(reinterpret_cast<void **>(&ptr_), n * sizeof(T)), wwrSuccess);
  }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  ~DeviceArray() { (void)wwrFree(ptr_); }

  void upload(const std::vector<T> &host) {
    ASSERT_EQ(host.size(), n_);
    ASSERT_EQ(wwrMemcpy(ptr_, host.data(), n_ * sizeof(T), wwrMemcpyHostToDevice), wwrSuccess);
  }
  std::vector<T> download() const {
    std::vector<T> host(n_);
    EXPECT_EQ(wwrMemcpy(host.data(), ptr_, n_ * sizeof(T), wwrMemcpyDeviceToHost), wwrSuccess);
    return host;
  }
  T *get() const { return ptr_; }

 private:
  T *ptr_ = nullptr;
  std::size_t n_;
};

// Column-major half operands, the doubles they round to, and their exact product.
struct Problem {
  std::vector<wwrHalf> a, b;
  std::vector<double> exact, magnitude; // sum_l a_il b_lj and sum_l |a_il b_lj|

  explicit Problem(int batch = 1) {
    std::mt19937 rng{20261027};
    std::uniform_real_distribution<float> dist{-1.0F, 1.0F};
    a.resize(std::size_t(batch) * kM * kK);
    b.resize(std::size_t(batch) * kK * kN);
    for (auto &x : a) x = wwrFloat2Half(dist(rng));
    for (auto &x : b) x = wwrFloat2Half(dist(rng));
    exact.assign(std::size_t(batch) * kM * kN, 0.0);
    magnitude.assign(exact.size(), 0.0);
    for (int p = 0; p < batch; ++p) {
      for (int j = 0; j < kN; ++j) {
        for (int i = 0; i < kM; ++i) {
          double sum = 0.0, abs_sum = 0.0;
          for (int l = 0; l < kK; ++l) {
            const double prod = double(wwrHalf2Float(a[std::size_t(p) * kM * kK + l * kM + i])) *
                                double(wwrHalf2Float(b[std::size_t(p) * kK * kN + j * kK + l]));
            sum += prod;
            abs_sum += std::abs(prod);
          }
          exact[std::size_t(p) * kM * kN + j * kM + i] = sum;
          magnitude[std::size_t(p) * kM * kN + j * kM + i] = abs_sum;
        }
      }
    }
  }

  // Every entry within k * u_acc * magnitude (+ u_out * |exact| for a half C),
  // and the whole product within frob_tol relative Frobenius error.
  void expect_within(const std::vector<double> &c, double u_acc, double u_out,
                     double frob_tol) const {
    ASSERT_EQ(c.size(), exact.size());
    double err2 = 0.0, ref2 = 0.0;
    int bad = 0;
    for (std::size_t e = 0; e < c.size(); ++e) {
      const double err = std::abs(c[e] - exact[e]);
      const double bound = kK * u_acc * magnitude[e] + u_out * std::abs(exact[e]);
      if (!(err <= bound) && bad++ < 5) {
        ADD_FAILURE() << "entry " << e << ": got " << c[e] << ", exact " << exact[e]
                      << ", error " << err << " > bound " << bound;
      }
      err2 += err * err;
      ref2 += exact[e] * exact[e];
    }
    EXPECT_EQ(bad, 0) << "entries outside the summation bound";
    EXPECT_LT(std::sqrt(err2 / ref2), frob_tol);
  }
};

std::vector<double> widen(const std::vector<wwrHalf> &c) {
  std::vector<double> out(c.size());
  std::ranges::transform(c, out.begin(), [](wwrHalf x) { return double(wwrHalf2Float(x)); });
  return out;
}

std::vector<double> widen(const std::vector<float> &c) { return {c.begin(), c.end()}; }

class HalfGemmTests : public ::testing::Test {
 protected:
  void SetUp() override { ASSERT_EQ(wwrblasCreate(&handle_), WWRBLAS_STATUS_SUCCESS); }
  void TearDown() override { EXPECT_EQ(wwrblasDestroy(handle_), WWRBLAS_STATUS_SUCCESS); }
  wwrblasHandle_t handle_ = nullptr;
};

TEST_F(HalfGemmTests, HgemmMatchesReference) {
  const Problem prob;
  DeviceArray<wwrHalf> a(prob.a.size()), b(prob.b.size()), c(prob.exact.size());
  a.upload(prob.a);
  b.upload(prob.b);
  const wwrHalf one = wwrFloat2Half(1.0F), zero = wwrFloat2Half(0.0F);

  ASSERT_EQ(wwrblasHgemm(handle_, WWRBLAS_OP_N, WWRBLAS_OP_N, kM, kN, kK, &one, a.get(), kM,
                         b.get(), kK, &zero, c.get(), kM),
            WWRBLAS_STATUS_SUCCESS);
  prob.expect_within(widen(c.download()), kUnitRoundoff16, kUnitRoundoff16, 2e-2);
}

TEST_F(HalfGemmTests, HgemmStridedBatchedMatchesReference) {
  constexpr int kBatch = 3;
  const Problem prob{kBatch};
  DeviceArray<wwrHalf> a(prob.a.size()), b(prob.b.size()), c(prob.exact.size());
  a.upload(prob.a);
  b.upload(prob.b);
  const wwrHalf one = wwrFloat2Half(1.0F), zero = wwrFloat2Half(0.0F);

  ASSERT_EQ(wwrblasHgemmStridedBatched(handle_, WWRBLAS_OP_N, WWRBLAS_OP_N, kM, kN, kK, &one,
                                       a.get(), kM, kM * kK, b.get(), kK, kK * kN, &zero,
                                       c.get(), kM, kM * kN, kBatch),
            WWRBLAS_STATUS_SUCCESS);
  prob.expect_within(widen(c.download()), kUnitRoundoff16, kUnitRoundoff16, 2e-2);
}

TEST_F(HalfGemmTests, GemmExHalfInFloatOutMatchesReference) {
  const Problem prob;
  DeviceArray<wwrHalf> a(prob.a.size()), b(prob.b.size());
  DeviceArray<float> c(prob.exact.size());
  a.upload(prob.a);
  b.upload(prob.b);
  const float one = 1.0F, zero = 0.0F; // COMPUTE_32F takes float scalars

  ASSERT_EQ(wwrblasGemmEx(handle_, WWRBLAS_OP_N, WWRBLAS_OP_N, kM, kN, kK, &one, a.get(),
                          WWRBLAS_R_16F, kM, b.get(), WWRBLAS_R_16F, kK, &zero, c.get(),
                          WWRBLAS_R_32F, kM, WWRBLAS_COMPUTE_32F, WWRBLAS_GEMM_DEFAULT),
            WWRBLAS_STATUS_SUCCESS);
  prob.expect_within(widen(c.download()), kUnitRoundoff32, 0.0, 1e-5);
}

TEST_F(HalfGemmTests, GemmExHalfInHalfOutMatchesReference) {
  const Problem prob;
  DeviceArray<wwrHalf> a(prob.a.size()), b(prob.b.size()), c(prob.exact.size());
  a.upload(prob.a);
  b.upload(prob.b);
  const float one = 1.0F, zero = 0.0F;

  ASSERT_EQ(wwrblasGemmEx(handle_, WWRBLAS_OP_N, WWRBLAS_OP_N, kM, kN, kK, &one, a.get(),
                          WWRBLAS_R_16F, kM, b.get(), WWRBLAS_R_16F, kK, &zero, c.get(),
                          WWRBLAS_R_16F, kM, WWRBLAS_COMPUTE_32F, WWRBLAS_GEMM_DEFAULT),
            WWRBLAS_STATUS_SUCCESS);
  // fp32 accumulation, then one rounding to half: the Frobenius error is the
  // output rounding's, about u16.
  prob.expect_within(widen(c.download()), kUnitRoundoff32, kUnitRoundoff16, 2e-3);
}

} // namespace
} // namespace wwr::extension::test
