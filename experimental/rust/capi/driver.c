/* A PURE C consumer — no C++, no CUDA header, no vendor name. It sees only
 * wwr_capi.h. This is exactly what the generated ffi.rs gives Rust: opaque PODs
 * and plain-C prototypes over libwwr.capi.so. If this links and prints the right
 * numbers, so does the Rust. Returns non-zero on any wrong answer, so ctest sees
 * a failure. */
#include "wwr_capi.h"
#include <math.h>
#include <stdio.h>

static int near_f(float got, float want) {
  return fabsf(got - want) < 1e-4f;
}

static int near_d(double got, double want) {
  return fabs(got - want) < 1e-9;
}

int main(void) {
  int ok = 1;

  wwr_complex64 a = {3.0f, 4.0f};
  wwr_complex64 b = {1.0f, 2.0f};

  wwr_complex64 s = wwr_cadd64(a, b);
  wwr_complex64 p = wwr_cmul64(a, b);
  float m = wwr_cabs64(a);
  wwr_complex64 q = wwr_cdiv64(a, b);

  printf("add  = %.4f + %.4fi   (expect 4.0000 + 6.0000i)\n", s.re, s.im);
  printf("mul  = %.4f + %.4fi   (expect -5.0000 + 10.0000i)\n", p.re, p.im);
  printf("abs  = %.4f             (expect 5.0000)\n", m);
  printf("div  = %.4f + %.4fi   (expect 2.2000 - 0.4000i)\n", q.re, q.im);

  ok &= near_f(s.re, 4.0f) && near_f(s.im, 6.0f);
  ok &= near_f(p.re, -5.0f) && near_f(p.im, 10.0f);
  ok &= near_f(m, 5.0f);
  ok &= near_f(q.re, 2.2f) && near_f(q.im, -0.4f);

  wwr_complex128 a2 = {3.0, 4.0};
  wwr_complex128 b2 = {1.0, 2.0};

  wwr_complex128 s2 = wwr_cadd128(a2, b2);
  wwr_complex128 p2 = wwr_cmul128(a2, b2);
  double m2 = wwr_cabs128(a2);
  wwr_complex128 q2 = wwr_cdiv128(a2, b2);

  printf("add128 = %.4f + %.4fi (expect 4.0000 + 6.0000i)\n", s2.re, s2.im);
  printf("mul128 = %.4f + %.4fi (expect -5.0000 + 10.0000i)\n", p2.re, p2.im);
  printf("abs128 = %.4f           (expect 5.0000)\n", m2);
  printf("div128 = %.4f + %.4fi (expect 2.2000 - 0.4000i)\n", q2.re, q2.im);

  ok &= near_d(s2.re, 4.0) && near_d(s2.im, 6.0);
  ok &= near_d(p2.re, -5.0) && near_d(p2.im, 10.0);
  ok &= near_d(m2, 5.0);
  ok &= near_d(q2.re, 2.2) && near_d(q2.im, -0.4);

  /* The status surface (#329 increment 2). These exercise the stable wwr_status
   * enum the thunk maps the backend status onto (architecture.md §6), so Rust
   * sees one mapping whichever backend libwwr.capi links. Every assertion below
   * is safe on a no-GPU CI runner and on both backends -- see each note. */

  /* (1) Structural stability proof. These values come from the GENERATED header
   * and are identical regardless of backend -- that fixed layout IS the §6
   * stability guarantee, proven without running anything on a GPU. */
  ok &= (WWR_STATUS_SUCCESS == 0);
  ok &= (WWR_STATUS_INVALID_VALUE != WWR_STATUS_INVALID_DEVICE);
  ok &= (WWR_STATUS_SUCCESS != WWR_STATUS_UNKNOWN);

  /* (2) Success path -- deterministic with or without a GPU, both backends. The
   * runtime/driver version query needs no device; a healthy toolkit returns a
   * positive version and wwrSuccess, which maps to WWR_STATUS_SUCCESS (== 0). */
  int ver = 0;
  wwr_status vs = wwr_runtime_get_version(&ver);
  printf("runtime version = %d  status = %d  (expect status 0, version > 0)\n",
         ver, (int)vs);
  ok &= (vs == WWR_STATUS_SUCCESS);
  ok &= (ver > 0);

  /* (3) Error path -- a NON-success status, asserted structurally, not by a
   * specific code. wwr_set_device(-1) is argument validation: an out-of-range
   * ordinal is rejected independently of GPU presence, so this is non-success on
   * a no-GPU runner too. We deliberately DO NOT hard-assert a specific
   * enumerator: the backend code for a bad ordinal vs no-device is not verified
   * identical across backends and runners, and the stable-value proof above
   * already covers §6. Non-success is the safe, backend- and GPU-agnostic claim. */
  wwr_status bad = wwr_set_device(-1);
  printf("set_device(-1) status = %d  (expect non-zero)\n", (int)bad);
  ok &= (bad != WWR_STATUS_SUCCESS);

  if (!ok) {
    fprintf(stderr, "FAIL: a computed value was wrong\n");
    return 1;
  }
  return 0;
}
