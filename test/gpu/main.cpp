// main.cpp - Shared test runner for the src compile-time tests

import std;
import gpumod.test.gpu.runtime_api;
import gpumod.test.gpu.complex;
import gpumod.test.gpu.fp16;
import gpumod.test.gpu.bf16;
import gpumod.test.gpu.blas;
import gpumod.test.gpu.solver;
import gpumod.test.gpu.sparse;
import gpumod.test.gpu.rand;

int main() {
  // Nothing to check here at run time: the tests in this binary are the
  // static_asserts in each imported module, proved when it compiled. What this
  // adds is the link -- reaching main means every module above compiled and
  // the executable linked.
  std::println("Compilation-time and link-time tests passed!");
}
