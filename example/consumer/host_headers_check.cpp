// host_headers_check.cpp -- consuming the fp16 / bf16 / rand / blas / solver /
// blaslt / rtc / fft / sparse surfaces the non-module way, by #include rather than import.
//
// The runtime_host_check.cpp twin for the modules rolled onto the
// header-consumption path after runtime_api. main.cpp reaches these surfaces by
// `import wwr.fp16;` etc. (indirectly, through the layers it uses); this TU
// reaches the SAME surfaces the other way a consumer can -- #include "wwr/fp16.h",
// "wwr/bf16.h", "wwr/rand.h", "wwr/blas.h", "wwr/solver.h", "wwr/blaslt.h",
// "wwr/rtc.h", "wwr/fft.h", "wwr/sparse.h", no import at all. It is the
// install-check for those paths: that the wwr/*.h headers, the detail/*_names.h
// fragments they pull in, and the wwr::fp16::host / wwr::bf16::host /
// wwr::rand::host / wwr::blas::host / wwr::solver::host / wwr::blaslt::host /
// wwr::rtc::host / wwr::fft::host / wwr::sparse::host targets all travel in the
// package and re-attach in a
// find_package consumer.
//
// wwr/tensor.h (wwr::tensor::host) rides the same path but is OPTIONAL in the
// package: tensor ships only from a build configured with -DWWR_WITH_TENSOR=ON,
// so it is reached under WWR_CONSUMER_HAS_TENSOR, which the consumer's
// CMakeLists sets from WWR_HAS_TENSOR -- the same optional-component shape the
// wrappers / extension layers use.
//
// A SEPARATE translation unit on purpose, for the reason runtime_host_check.cpp
// documents: a TU that both imports a module and #includes its header twin would
// declare the same wwr* names twice. Across TUs linked together it is fine.
#include "wwr/bf16.h"
#include "wwr/blas.h"
#include "wwr/blaslt.h"
#include "wwr/fft.h"
#include "wwr/fp16.h"
#include "wwr/rand.h"
#include "wwr/rtc.h"
#include "wwr/solver.h"
#include "wwr/sparse.h"
#if defined(WWR_CONSUMER_HAS_TENSOR)
#include "wwr/tensor.h"
#endif

// Proof is COMPILE + LINK, no device needed. Compile: the wwr* names exist from a
// pure #include. Link: calling the conversion forwarders and taking the address
// of a real RNG / BLAS entry point bound through the header forces the symbols to
// resolve.
bool host_headers_check() {
  const wwr::wwrHalf half = wwr::wwrFloat2Half(1.5f);
  const wwr::wwrBfloat16 bf = wwr::wwrFloat2Bfloat16(1.5f);

  wwr::wwrrandStatus_t (*create)(wwr::wwrrandGenerator_t *, wwr::wwrrandRngType_t) =
      &wwr::wwrrandCreateGenerator;

  wwr::wwrblasStatus_t (*blas_create)(wwr::wwrblasHandle_t *) = &wwr::wwrblasCreate;

  wwr::wwrsolverStatus_t (*solver_create)(wwr::wwrsolverDnHandle_t *) = &wwr::wwrsolverDnCreate;

  wwr::wwrblasLtStatus_t (*blaslt_create)(wwr::wwrblasLtHandle_t *) = &wwr::wwrblasLtCreate;

  wwr::wwrrtcResult (*rtc_version)(int *, int *) = &wwr::wwrrtcVersion;

  wwr::wwrfftResult_t (*fft_create)(wwr::wwrfftHandle *) = &wwr::wwrfftCreate;

  wwr::wwrsparseStatus_t (*sparse_create)(wwr::wwrsparseHandle_t *) = &wwr::wwrsparseCreate;

  bool ok = wwr::wwrHalf2Float(half) == 1.5f && wwr::wwrBfloat162Float(bf) == 1.5f &&
            create != nullptr && blas_create != nullptr && solver_create != nullptr &&
            blaslt_create != nullptr && rtc_version != nullptr && fft_create != nullptr &&
            sparse_create != nullptr;

#if defined(WWR_CONSUMER_HAS_TENSOR)
  wwr::wwrtensorStatus_t (*tensor_create)(wwr::wwrtensorHandle_t *) = &wwr::wwrtensorCreate;
  ok = ok && tensor_create != nullptr;
#endif

  return ok;
}
