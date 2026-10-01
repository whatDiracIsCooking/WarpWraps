// Compile-time test for the portable common atomics: the set both CUDA and HIP
// spell identically in the global namespace (atomicAdd, atomicCAS, atomicExch,
// atomicMin/Max, atomicAnd/Or/Xor, atomicSub, atomicInc/Dec).
//
// There is no src atomics header to test -- the atomic builtins ride the
// vendor runtime header, which runtime.h's device section switches
// (<cuda_runtime.h> vs <hip/hip_runtime.h>), so any device TU that includes it
// has them for free, under one identical spelling on both backends. wwr does
// NOT wrap them: like cooperative_groups.h (see src's
// README), a forwarding function per name would only rename each name to
// itself, and the one real divergence -- AMD FP-atomic codegen, gated on
// -munsafe-fp-atomics -- is a compile flag a source wrapper cannot touch.
//
// What this file locks down is the portability contract that lets extension
// code call these bare: that every operation below resolves for the common
// widths under BOTH front ends. nvcc is the permissive one, so a HIP-only gap
// (a missing overload on some future ROCm) is exactly the #18-class regression
// this catches -- building the TU under -x hip IS the assertion. No ctest
// entry, no launch, matching cooperative_groups.cu.
//
// FP atomics are exercised on the default (safe) codegen: -munsafe-fp-atomics
// is opt-in and not passed here, so atomicAdd on float/double lowers to AMD's
// correct CAS-loop fallback. See docs/architecture.md section 15.
#include "runtime.h"

namespace {

// Integer atomics: the widths every operation supports on both backends.
__global__ void wwr_atomics_integral(int *i, unsigned *u, unsigned long long *ull) {
  atomicAdd(i, *i);
  atomicAdd(u, *u);
  atomicAdd(ull, *ull);

  atomicSub(i, *i);
  atomicSub(u, *u);

  atomicExch(i, *i);
  atomicExch(u, *u);
  atomicExch(ull, *ull);

  atomicMin(i, *i);
  atomicMin(u, *u);
  atomicMin(ull, *ull);

  atomicMax(i, *i);
  atomicMax(u, *u);
  atomicMax(ull, *ull);

  atomicAnd(i, *i);
  atomicAnd(u, *u);
  atomicAnd(ull, *ull);

  atomicOr(i, *i);
  atomicOr(u, *u);
  atomicOr(ull, *ull);

  atomicXor(i, *i);
  atomicXor(u, *u);
  atomicXor(ull, *ull);

  atomicCAS(i, *i, *i);
  atomicCAS(u, *u, *u);
  atomicCAS(ull, *ull, *ull);

  // inc/dec: unsigned-only, with a wraparound semantic, on both backends.
  atomicInc(u, *u);
  atomicDec(u, *u);
}

// Floating-point atomics: add and exch, on the default safe codegen (see the
// file header). double atomicAdd needs sm_60+, met by the compile-time
// preset's sm_70.
__global__ void wwr_atomics_floating(float *f, double *d) {
  atomicAdd(f, *f);
  atomicAdd(d, *d);
  atomicExch(f, *f);
}

} // namespace
