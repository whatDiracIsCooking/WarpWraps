// main.cpp - Shared test runner for compile-time module tests

import std;
import wwr.test.cuda.cuda_runtime_api;
import wwr.test.cuda.cuComplex;
import wwr.test.cuda.cuda_fp16;
import wwr.test.cuda.cuda_bf16;
import wwr.test.cuda.cublasLt;
import wwr.test.cuda.cublasXt;
import wwr.test.cuda.cufft;
import wwr.test.cuda.cufftXt;
import wwr.test.cuda.cusparse;
import wwr.test.cuda.cuda_fp4;
import wwr.test.cuda.cuda_fp6;
import wwr.test.cuda.cuda_fp8;
import wwr.test.cuda.cuda_h;
import wwr.test.cuda.cusolverMg;
import wwr.test.cuda.cupti;
import wwr.test.cuda.cuda_profiler_api;
import wwr.test.cuda.cusolverSp;
import wwr.test.cuda.nvFatbin;
import wwr.test.cuda.cufile;
import wwr.test.cuda.nvJitLink;
import wwr.test.cuda.nvjpeg;
import wwr.test.cuda.nvml;
import wwr.test.cuda.nvrtc;

int main() {
  // Nothing to check here at run time: the tests in this binary are the
  // static_asserts in each imported module, proved when it compiled. What this
  // adds is the link -- reaching main means every module above compiled and
  // the executable linked.
  std::println("Compilation-time and link-time tests passed!");
}
