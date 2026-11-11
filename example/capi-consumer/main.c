/* A PURE C consumer of an INSTALLED wwr -- it stands in for any non-C++ FFI
 * consumer (Rust, Python ctypes, ...). It sees wwr only through the installed
 * <wwr/wwr_capi.h> and links the installed libwwr_capi.so; it knows nothing of
 * wwr's module core, its source tree, or the C++23 toolchain the rest of the
 * package needs. That is the point: if this compiles, links and prints the
 * right numbers against the install prefix alone, the capi install rules (the
 * .so + soname, the header into the wwr include root, wwr::capi in the export
 * set, no PRIVATE-visibility leak of the capi's wwr deps) are all correct.
 *
 * src/capi/driver.c is the in-tree twin of this file, run as a ctest against
 * the IN-TREE library; this one is the find_package half that only
 * devtools/install-check.sh can prove. Returns non-zero on any wrong answer, so
 * the script sees a failure; 77 when no device is reachable, which the script
 * reports as a skip rather than a failure (same convention as example/consumer). */
#include <wwr/wwr_capi.h>

#include <math.h>
#include <stdio.h>

static int near_f(float got, float want) {
  return fabsf(got - want) < 1e-4f;
}

int main(void) {
  int ok = 1;

  /* The host arithmetic surface -- deterministic, no device needed, identical
   * on both backends. wwr_cadd64 is the one the issue names; exercise a couple
   * more so a mis-installed .so that resolves one symbol but not the rest also
   * fails here. */
  wwr_complex64 a = {3.0f, 4.0f};
  wwr_complex64 b = {1.0f, 2.0f};
  wwr_complex64 s = wwr_cadd64(a, b);
  float m = wwr_cabs64(a);

  printf("add = %.4f + %.4fi  (expect 4.0000 + 6.0000i)\n", s.re, s.im);
  printf("abs = %.4f            (expect 5.0000)\n", m);

  ok &= near_f(s.re, 4.0f) && near_f(s.im, 6.0f);
  ok &= near_f(m, 5.0f);

  /* The status surface's structural guarantee -- these enum values come from
   * the installed header and are frozen across backends (architecture.md §6),
   * so this asserts the header travelled correctly without touching a GPU. */
  ok &= (WWR_STATUS_SUCCESS == 0);
  ok &= (WWR_STATUS_SUCCESS != WWR_STATUS_UNKNOWN);

  if (!ok) {
    fprintf(stderr, "FAIL: a computed value was wrong\n");
    return 1;
  }

  /* The runtime version query. On a healthy toolkit it returns a positive
   * version and WWR_STATUS_SUCCESS with no device present; on a box with no
   * reachable runtime at all it reports non-success, which we treat as ctest's
   * "skipped" (77) rather than a failure -- install-check.sh builds the consumer
   * everywhere but only a GPU box can run this leg. */
  int ver = 0;
  wwr_status vs = wwr_runtime_get_version(&ver);
  printf("runtime version = %d  status = %d  (expect status 0, version > 0)\n",
         ver, (int)vs);
  if (vs != WWR_STATUS_SUCCESS || ver <= 0) {
    fprintf(stderr, "no reachable runtime -- skipping the version assertion\n");
    return 77;
  }

  return 0;
}
