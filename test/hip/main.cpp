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
}
