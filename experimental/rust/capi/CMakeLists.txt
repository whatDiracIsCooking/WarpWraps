# wwr.capi — the experimental extern "C" ABI over wwr, for Rust FFI
#
# A shared library of the generated extern "C" thunks (gen/wwr_capi.cpp), which
# `import wwr.complex;` and forward into the wwr::wwrC* functions -- so the
# backend is chosen only by which wwr library this links, not by anything in the
# source. driver.c is a pure-C consumer run as a ctest: it checks the symbols
# link and the numbers are right (host-only, no GPU). See the directory README
# and docs/architecture.md §25 for why the thunk calls into wwr rather than
# re-deriving it.
#
# The soname is versioned only to exercise the packaging shape -- the ABI is
# experimental and NOT yet a stable contract (issue #329); it is not part of the
# installed wwr package (wwr_install_package() sweeps src/ only).

# The thunk is a plain shared library, not a module (it exports no module, it
# imports one). CXX_MODULE_STD / CXX_SCAN_FOR_MODULES let a non-module TU
# consume the wwr.complex module, which `import std;`s; the real dotted name on
# the add_library, the :: alias by hand.
add_library(wwr.capi SHARED gen/wwr_capi.cpp)
add_library(wwr::capi ALIAS wwr.capi)

target_link_libraries(wwr.capi PRIVATE wwr::complex)
target_include_directories(wwr.capi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gen)
set_target_properties(
  wwr.capi
  PROPERTIES CXX_MODULE_STD 1
             CXX_SCAN_FOR_MODULES ON
             CUDA_SEPARABLE_COMPILATION OFF
             CUDA_RESOLVE_DEVICE_SYMBOLS OFF
             VERSION 0.1.0
             SOVERSION 0
)

if(WWR_BUILD_TESTS)
  # driver.c is pure C -- the one C TU in the tree -- so enable the C language
  # here (the project() call enables only CXX, plus CUDA on a CUDA build). It
  # stands in for Rust's C ABI exactly; keeping it C, not C++, is the point.
  enable_language(C)

  # A pure-C consumer that links the symbols and verifies the arithmetic. Not a
  # module TU and not C++, so it uses add_executable directly rather than a
  # wwr_* test macro. Host-only: NOT REQUIRES_GPU, so it runs on no-GPU CI.
  add_executable(wwr.capi.driver driver.c)
  target_link_libraries(wwr.capi.driver PRIVATE wwr::capi)
  target_include_directories(
    wwr.capi.driver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gen
  )
  add_test(NAME wwr.capi.driver COMMAND wwr.capi.driver)
  set_tests_properties(wwr.capi.driver PROPERTIES TIMEOUT 60)
endif()
