// tensor_link_check.cpp - link-only main() for the wwr.tensor compile-time test
//
// This executable is BUILT but never RUN (NO_RUN in test/gpu/CMakeLists.txt).
// wwr.test.gpu.tensor is whole-archived in, so the WWR_LINK_CHECK entries in
// test/gpu/tensor.cppm force every wwrtensor* symbol to resolve against
// libcutensor / libhiptensor at link time -- a successful link is the assertion.
// It is not launched because libhiptensor.so aborts in a global constructor on
// any host without a supported AMD GPU (see the comment in test/gpu/CMakeLists.txt).
int main() {}
