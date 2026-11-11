/* A PURE C consumer — no C++, no CUDA header, no vendor name. It sees only
 * wwr_capi.h. This is exactly what the generated ffi.rs gives Rust: opaque PODs
 * and plain-C prototypes over libwwr_capi.so. If this links and prints the right
 * numbers, so does the Rust. */
#include "wwr_capi.h"
#include <stdio.h>

int main(void) {
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
  return 0;
}
