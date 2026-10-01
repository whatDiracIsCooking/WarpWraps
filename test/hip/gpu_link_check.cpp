// gpu_link_check.cpp - link-only main() for the hiptensor + rccl compile-time tests
//
// This executable is BUILT but never RUN (NO_RUN in test/hip/CMakeLists.txt).
// wwr.test.hip.hiptensor and wwr.test.hip.rccl are whole-archived in, so the
// WWR_LINK_CHECK entries in those modules force every re-exported symbol to
// resolve against libhiptensor / librccl at link time -- a successful link is the
// assertion. It is not launched because both libraries' load-time initialisation
// aborts on a host without a GPU (see the comment in test/hip/CMakeLists.txt),
// matching test/gpu/tensor_link_check.cpp.
int main() {}
