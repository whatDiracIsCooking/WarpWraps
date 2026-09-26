// main.cpp - Shared test runner for compile-time module tests

import std;
import gpumod.test.cuda.cuda_runtime_api;
import gpumod.test.cuda.cuComplex;
import gpumod.test.cuda.cuda_fp16;
import gpumod.test.cuda.cuda_bf16;
import gpumod.test.cuda.cublasLt;
import gpumod.test.cuda.cublasXt;
import gpumod.test.cuda.cufft;
import gpumod.test.cuda.cufftXt;
import gpumod.test.cuda.cusparse;
import gpumod.test.cuda.cuda_fp4;
import gpumod.test.cuda.cuda_fp6;
import gpumod.test.cuda.cuda_fp8;
import gpumod.test.cuda.cuda_h;
import gpumod.test.cuda.cusolverMg;
import gpumod.test.cuda.cupti;
import gpumod.test.cuda.cuda_profiler_api;
import gpumod.test.cuda.cusolverSp;
import gpumod.test.cuda.nvFatbin;
import gpumod.test.cuda.cufile;
import gpumod.test.cuda.nvJitLink;
import gpumod.test.cuda.nvjpeg;
import gpumod.test.cuda.nvml;
import gpumod.test.cuda.nvrtc;

int main() {
  // Nothing to check here at run time: the tests in this binary are the
  // static_asserts in each imported module, proved when it compiled. What this
  // adds is the link -- reaching main means every module above compiled and
  // the executable linked.
  std::println("Compilation-time and link-time tests passed!");
}
