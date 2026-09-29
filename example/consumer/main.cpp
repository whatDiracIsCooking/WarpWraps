// main.cpp -- what using an installed wwr actually looks like.
//
// This always consumes the core wwr package: the backend-neutral gpu* layer
// (wwr.runtime_api, wwr.blas) and the dispatch wrappers (wwr.wrappers.*).
//
// The extension layer (wwr.extension.*, the RAII handle / buffer / error
// abstractions plus the untyped tools-extension guard wwr.extension.tx) is
// OPTIONAL in the package -- it ships only from a build
// configured with -DWWR_INSTALL_EXTENSION=ON. This file consumes it too, guarded
// by WWR_CONSUMER_HAS_EXTENSION, which the consumer's CMakeLists.txt defines from
// the package's WWR_HAS_EXTENSION variable. So the one source builds against a
// core-only install (managing its own device memory and handles, as
// multiply_square does) and against a full one, and install-check.sh --extension
// is what compiles this half.
//
// Two things are proved, and they fail differently:
//
//   * COMPILE + LINK proves the install. `import` resolving means the module
//     SOURCES were installed and the export set re-attaches them (a C++23
//     module package ships sources -- a BMI is not portable -- and this build
//     compiles them). Each wrapper's units #include "wrappers/.../dispatch_*.h"
//     from their global module fragment, so those headers had to travel next to
//     the sources; wwr.wrappers.sparse also needs the WWR_GPU_BACKEND_*
//     define at install-compile time. Taking the address of one instantiation
//     per wrapper (solver/fft/sparse below) forces each to resolve and link
//     without running a kernel -- so this half needs no GPU.
//   * RUN proves it works. multiply_square does a real gemm on the device:
//     allocate, copy up, dispatch gemm<float,int> (cublasSgemm on CUDA,
//     hipblasSgemm on HIP -- named nowhere here, which is the point), copy down,
//     check. This half needs a device; with none it reports 77 (ctest's skip)
//     and the compile-and-link result still stands.

// stderr is a FILE*, which `import std;` does not give you -- the std module
// exports the std:: names, not the C library's macros and objects. An ordinary
// #include beside an import is fine in a plain translation unit like this one;
// it is only inside a MODULE unit's global module fragment that this project's
// headers and `import std;` would collide.
#include <cstdio>

import std;

import wwr.runtime_api; // wwrMalloc, wwrMemcpy, wwrGetDevice, wwrSuccess
import wwr.blas;        // wwrblasHandle_t, wwrblasCreate, WWRBLAS_OP_N
import wwr.wrappers.blas;
import wwr.wrappers.solver;
import wwr.wrappers.fft;
import wwr.wrappers.sparse;

#if defined(WWR_CONSUMER_HAS_EXTENSION)
import wwr.extension.common;        // success_code/error_name/error_string, error_policy concepts
import wwr.extension.runtime;       // StreamWrapper and the rest of the RAII runtime
import wwr.extension.random_normal; // random_normal<T>, backed by a device archive
import wwr.extension.tx;            // wwr::extension::ScopedRange
#endif

// The gpu* names (wwrSuccess, wwrMalloc, WWRBLAS_OP_N, ...) and the wrappers
// (gemm, potri, ...) are all exported in namespace wwr. A consumer is not
// inside it, so unlike this project's own tests it has to say so.
using namespace wwr;

namespace {

// C = A * B for square column-major matrices, via the backend's BLAS.
//
// A is the identity and B is 1, 2, 3, ..., so A * B == B and the check needs no
// tolerance. Device memory and the handle are managed by hand here -- the RAII
// wrappers that would do this live in the (uninstalled) extension layer.
bool multiply_square(const int n) {
  const std::size_t count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

  std::vector<float> host_a(count, 0.0f);
  std::vector<float> host_b(count);
  for (std::size_t i = 0; i < count; ++i) {
    host_b[i] = static_cast<float>(i + 1);
  }
  for (int i = 0; i < n; ++i) {
    host_a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
           static_cast<std::size_t>(i)] = 1.0f;
  }

  const std::size_t bytes = count * sizeof(float);
  float *a = nullptr;
  float *b = nullptr;
  float *c = nullptr;
  if (wwrMalloc(reinterpret_cast<void **>(&a), bytes) != wwrSuccess ||
      wwrMalloc(reinterpret_cast<void **>(&b), bytes) != wwrSuccess ||
      wwrMalloc(reinterpret_cast<void **>(&c), bytes) != wwrSuccess) {
    std::println(stderr, "device allocation failed");
    return false;
  }

  // One cleanup path for every early return below. wwrFree's status is
  // discarded on purpose -- this is best-effort teardown -- and the casts are
  // load-bearing: hipFree is [[nodiscard]] where cudaFree is not, so without
  // them the example warns under HIP and is clean under CUDA.
  const auto teardown = [&] {
    (void)wwrFree(a);
    (void)wwrFree(b);
    (void)wwrFree(c);
  };

  if (wwrMemcpy(a, host_a.data(), bytes, wwrMemcpyHostToDevice) != wwrSuccess ||
      wwrMemcpy(b, host_b.data(), bytes, wwrMemcpyHostToDevice) != wwrSuccess) {
    std::println(stderr, "host -> device copy failed");
    teardown();
    return false;
  }

  wwrblasHandle_t handle{};
  if (wwrblasCreate(&handle) != WWRBLAS_STATUS_SUCCESS) {
    std::println(stderr, "wwrblasCreate failed");
    teardown();
    return false;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const auto status = gemm<float, int>(handle, WWRBLAS_OP_N, WWRBLAS_OP_N, n, n, n, &alpha, a, n, b,
                                       n, &beta, c, n);
  wwrblasDestroy(handle);
  if (status != WWRBLAS_STATUS_SUCCESS) {
    std::println(stderr, "gemm failed with status {}", static_cast<int>(status));
    teardown();
    return false;
  }

  std::vector<float> host_c(count);
  const bool copied = wwrMemcpy(host_c.data(), c, bytes, wwrMemcpyDeviceToHost) == wwrSuccess &&
                      wwrDeviceSynchronize() == wwrSuccess;
  teardown();
  if (!copied) {
    std::println(stderr, "device -> host copy failed");
    return false;
  }

  for (std::size_t i = 0; i < count; ++i) {
    if (host_c[i] != host_b[i]) {
      std::println(stderr, "gemm mismatch at {}: got {}, want {}", i, host_c[i], host_b[i]);
      return false;
    }
  }

  std::println("gemm   : {0}x{0} identity * B == B, {1} elements checked", n, count);
  return true;
}

// solver / fft / sparse are proved at compile and link time only: taking the
// address of one instantiation each forces its wrapper to resolve and its
// archive to link, without needing a device. The volatile sink keeps the
// compiler from folding the reads away.
bool wrappers_link() {
  static const void *volatile sink[] = {
      reinterpret_cast<const void *>(&potri<float>),    // wwr.wrappers.solver
      reinterpret_cast<const void *>(&exec_c2c<float>), // wwr.wrappers.fft
      reinterpret_cast<const void *>(&bsrmv<float>),    // wwr.wrappers.sparse
  };
  for (const void *volatile p : sink) {
    if (p == nullptr)
      return false;
  }
  std::println("link   : solver/fft/sparse wrappers resolved and linked");
  return true;
}

#if defined(WWR_CONSUMER_HAS_EXTENSION)
// The extension layer, proved at compile and link time only -- no device
// needed, same as wrappers_link above. Naming extension::StreamWrapper proves
// wwr.extension.runtime's module sources installed and compiled here; taking the
// address of random_normal<float> forces its WHOLE_ARCHIVE device archive
// (wwr.extension.random_normal.device) to resolve and link, which is the
// fragile part of the extension install -- the archive has to survive the export
// set and re-attach in a find_package consumer.
// The library ships no error policy -- a consumer brings its own. This is this
// consumer's: print to stderr and abort on failure. It only names the wrapper's
// policy arguments below; nothing here runs it.
template<typename T>
struct AbortPolicy {
  using error_type = T;
  void handle_error(const T error, std::source_location loc) noexcept {
    if (error != extension::success_code<T>()) {
      std::println(stderr, "GPU error at {}:{} in {}: {} ({})", loc.file_name(), loc.line(),
                   loc.function_name(), extension::error_name(error), extension::error_string(error));
      std::abort();
    }
  }
};

bool extension_link() {
  static_assert(sizeof(extension::StreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>,
                                                   AbortPolicy<wwrError_t>>) > 0);
  static const void *volatile sink[] = {
      reinterpret_cast<const void *>(&extension::random_normal<float>),
  };
  for (const void *volatile p : sink) {
    if (p == nullptr)
      return false;
  }
  std::println("ext    : extension runtime + random_normal resolved and linked");
  return true;
}
#endif

} // namespace

int main() {
#if defined(WWR_CONSUMER_HAS_EXTENSION)
  // wwr.extension.tx is host-side profiler annotation -- it needs no device, so
  // constructing a ScopedRange here proves the tx chain (wwr.extension.tx ->
  // wwr.tx -> the backend's NVTX/rocTX module) installs, compiles, links AND
  // runs, regardless of whether a GPU is present. It guards the rest of main, so
  // it pops on every return path below. Guarded by WWR_CONSUMER_HAS_EXTENSION
  // because tx now ships with the optional extension layer, not the wrappers.
  const extension::ScopedRange session{"wwr install-check"};
  std::println("tx     : scoped range resolved and linked");
#endif

  // Compile-and-link proof first: no device needed, and it is what proves the
  // install regardless of whether a GPU is present to run the gemm.
  if (!wrappers_link())
    return 1;

#if defined(WWR_CONSUMER_HAS_EXTENSION)
  if (!extension_link())
    return 1;
#endif

  int device = 0;
  if (wwrGetDevice(&device) != wwrSuccess) {
    std::println("wwr : installed package consumed and linked; no GPU to run the gemm");
    return 77; // ctest's conventional "skipped"
  }

  if (!multiply_square(64))
    return 1;

  std::println("wwr : consumed from an installed package, all checks passed");
  return 0;
}
