// main.cpp - Shared test runner for the src compile-time tests

import std;
import wwr.test.gpu.runtime_api;
import wwr.test.gpu.complex;
import wwr.test.gpu.fp16;
import wwr.test.gpu.bf16;
import wwr.test.gpu.blas;
import wwr.test.gpu.solver;
import wwr.test.gpu.sparse;
import wwr.test.gpu.rand;

int main() {
  // Nothing to check here at run time: the tests in this binary are the
  // static_asserts in each imported module, proved when it compiled. What this
  // adds is the link -- reaching main means every module above compiled and
  // the executable linked.
  std::println("Compilation-time and link-time tests passed!");
}
