// main.cpp - Shared test runner for HIP compile-time module tests

import std;
import wwr.test.hip.hip_runtime_api;
import wwr.test.hip.rocm_smi;
import wwr.test.hip.amd_smi;
import wwr.test.hip.roctracer;
import wwr.test.hip.hip_complex;
import wwr.test.hip.hip_fp16;
import wwr.test.hip.hip_bf16;
import wwr.test.hip.hip_fp8;
// wwr.test.hip.hip_fp4 / hip_fp6 deliberately not imported -- see the
// BLOCKED comment in src/hip/CMakeLists.txt and src/hip/README.md.
import wwr.test.hip.hiprtc;
import wwr.test.hip.hipblas;
import wwr.test.hip.hipblaslt;
import wwr.test.hip.hipsolver;
import wwr.test.hip.hipsparse;
import wwr.test.hip.hipfft;
import wwr.test.hip.hipfftXt;
import wwr.test.hip.hiprand;

int main() {
  // Nothing to check here at run time: the tests in this binary are the
  // static_asserts in each imported module, proved when it compiled. What this
  // adds is the link -- reaching main means every module above compiled and
  // the executable linked.
  std::println("HIP compilation-time and link-time tests passed!");

  // Skip global/static teardown. ROCm's amd_smi and rocm_smi (both linked here,
  // via wwr.hip.amd_smi / wwr.hip.rocm_smi) each register an atexit finalizer
  // that destroys the same std::map<amd::smi::DevInfoTypes, ...> -- a latent
  // double-free in the vendor libraries (rocm-7.2.4, confirmed under valgrind:
  // every frame is in librocm_smi64/libamd_smi __cxa_finalize, none in wwr).
  // It is normally benign, but the finalization order once libhipcomp is a
  // DT_NEEDED of this binary (wwr.hip.hipcomp's link checks) lands the second
  // free on a reused block -> `malloc_consolidate(): invalid chunk size` at
  // exit, AFTER every test above has already passed. This runner's contract is
  // "it compiled, linked, and reached main"; vendor global-destructor teardown
  // is out of scope, so leave without running it. _Exit does not flush, so flush
  // the pass line first -- fflush(nullptr) flushes every open stream and needs
  // no <cstdio> `stdout` macro (import std does not export macros).
  std::fflush(nullptr);
  std::_Exit(0);
}
