// main.cpp - Shared test runner for HIP compile-time module tests

import std;
import gpumod.test.hip.hip_runtime_api;
import gpumod.test.hip.rocm_smi;
import gpumod.test.hip.amd_smi;
import gpumod.test.hip.roctracer;
import gpumod.test.hip.hip_complex;
import gpumod.test.hip.hip_fp16;
import gpumod.test.hip.hip_bf16;
import gpumod.test.hip.hip_fp8;
// gpumod.test.hip.hip_fp4 / hip_fp6 deliberately not imported -- see the
// BLOCKED comment in src/hip/CMakeLists.txt and src/hip/README.md.
import gpumod.test.hip.hiprtc;
import gpumod.test.hip.hipblas;
import gpumod.test.hip.hipblaslt;
import gpumod.test.hip.hipsolver;
import gpumod.test.hip.hipsparse;
import gpumod.test.hip.hipfft;
import gpumod.test.hip.hipfftXt;
import gpumod.test.hip.hiprand;

int main() {
  // Nothing to check here at run time: the tests in this binary are the
  // static_asserts in each imported module, proved when it compiled. What this
  // adds is the link -- reaching main means every module above compiled and
  // the executable linked.
  std::println("HIP compilation-time and link-time tests passed!");
}
