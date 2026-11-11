# wwr.capi — the extern "C" ABI over wwr, for Rust FFI
#
# A shared library of the generated extern "C" thunks (gen/wwr_capi.cpp), which
# `import wwr.complex;` / `import wwr.runtime_api;` and forward into the
# wwr::wwrC* and wwr::wwr* runtime functions -- so the backend is chosen only by
# which wwr library this links, not by anything in the source. The status
# surface maps the neutral wwr status onto a stable wwr_status C enum below the
# extern "C" line (architecture.md section 6). driver.c is a pure-C consumer run
# as a ctest: it checks the symbols link, the complex numbers are right, and the
# status surface returns the stable enum (host-only, no GPU). See the directory
# README and docs/architecture.md sections 6/25 for why the thunk calls into wwr
# and maps the status.
#
# Unlike the rest of wwr (which ships as module SOURCE a consumer compiles -- a
# BMI is not portable), the capi is a prebuilt libwwr_capi.so a non-C++ consumer
# links. It therefore does NOT ride the src/-sweep in wwr_install.cmake; it is
# installed explicitly there under WWR_INSTALL_CAPI (the sweep's non-recursive
# src/ collection never reaches this subdirectory). The target builds
# UNCONDITIONALLY with the rest of src/ -- only the install is gated.
#
# VERSION / SOVERSION below are the first declared contract version (1.0.0 /
# SOVERSION 1): the wwr_status enumerator values are frozen and SOVERSION bumps
# only on an ABI break. See the directory README's "Stability policy" and
# architecture.md section 21.

# The thunk is a plain shared library, not a module (it exports no module, it
# imports one). CXX_MODULE_STD / CXX_SCAN_FOR_MODULES let a non-module TU use
# the wwr.complex / wwr.runtime_api modules, which `import std;`; the real
# dotted name on the add_library, the :: alias by hand.
add_library(wwr.capi SHARED gen/wwr_capi.cpp)
add_library(wwr::capi ALIAS wwr.capi)

# The OUTPUT file name is wwr_capi (-> libwwr_capi.so), NOT the dotted target
# name: the target keeps the convention's dotted wwr.capi + wwr::capi alias, but
# rustc's `-l wwr_capi` / ffi.rs's `#[link(name = "wwr_capi")]` do a literal
# libwwr_capi.so lookup (and underscore matches the C ABI's wwr_ symbol prefix).
# driver.c links via the target so it never saw the file name; the Rust driver
# does. generate.py emits the matching link name -- keep the two in step.
set_target_properties(wwr.capi PROPERTIES OUTPUT_NAME wwr_capi)

# wwr::complex for the POD-complex thunks; wwr::runtime_api for the status
# surface (its module BMI for the import) AND wwr::runtime_api::host for the
# runtime host library (CUDA::cudart / hip::host) that resolves the
# wwrRuntimeGetVersion / wwrGetDeviceCount / wwrSetDevice symbols at link.
# wwr::blas / wwr::fp16 for the blas surface: each is the module BMI for its
# import AND carries the vendor library (CUDA::cublas / roc::hipblas)
# transitively (LINK_PUBLIC on wwr.blas), which resolves the cublas* / hipblas*
# symbols at link. The blas thunks run on the DEVICE at call time; CI
# link-checks them (no GPU), a GPU box exercises them (README, architecture.md).
target_link_libraries(
  wwr.capi
  PRIVATE wwr::complex
          wwr::runtime_api
          wwr::runtime_api::host
          wwr::blas
          wwr::fp16
)
target_include_directories(wwr.capi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gen)
set_target_properties(
  wwr.capi
  PROPERTIES CXX_MODULE_STD 1
             CXX_SCAN_FOR_MODULES ON
             CUDA_SEPARABLE_COMPILATION OFF
             CUDA_RESOLVE_DEVICE_SYMBOLS OFF
             VERSION 1.0.0
             SOVERSION 1
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

  # The Rust FFI test (#329 increment 3). rustc-direct, no cargo -- it compiles
  # the ONE driver.rs, which include!s the generated gen/ffi.rs, and links the
  # wwr.capi shared library. This is the real thing driver.c stands in for.
  #
  # Gated on rustc being FOUND, not merely on WWR_BUILD_TESTS, and that gate is
  # load-bearing for CI: ci.yml's cpp jobs run inside the image main already
  # published (container: is resolved before any step, so a PR that edits
  # docker/ is tested against the OLD, rust-less image -- see ci.yml /
  # images.yml headers). With no rustc there, this registers nothing and prints
  # a visible skip, like the repo's GPU-gated tests, so this PR's CI is green.
  # Once images.yml republishes the rust-enabled base on main, every later cpp
  # run finds rustc and RUNS it. Proven locally against a pinned rustc on both
  # backends before shipping -- it is not a test that only ever skips.
  find_program(RUSTC rustc)
  if(RUSTC)
    # The output FILE is driver_rs (not the target name), so it cannot collide
    # with the add_custom_target below -- Ninja rejects a file and a target that
    # share a name. The ctest name is set on add_test, independent of both.
    set(_rs_bin ${CMAKE_CURRENT_BINARY_DIR}/driver_rs)
    set(_rs_src ${CMAKE_CURRENT_SOURCE_DIR}/driver.rs)
    set(_rs_gen ${CMAKE_CURRENT_SOURCE_DIR}/gen)
    # Link against the built library: -L its output dir, -l wwr_capi (its
    # OUTPUT_NAME), and an rpath to that dir so the binary finds libwwr_capi.so
    # at run time without an LD_LIBRARY_PATH on the ctest. WWR_CAPI_FFI points
    # driver.rs's include!(concat!(env!(...))) at the committed gen/ dir.
    add_custom_command(
      OUTPUT ${_rs_bin}
      COMMAND ${CMAKE_COMMAND} -E env WWR_CAPI_FFI=${_rs_gen}
              ${RUSTC} --edition 2021 -O
              -L $<TARGET_FILE_DIR:wwr.capi>
              -l wwr_capi
              -C link-arg=-Wl,-rpath,$<TARGET_FILE_DIR:wwr.capi>
              -o ${_rs_bin} ${_rs_src}
      DEPENDS ${_rs_src} ${_rs_gen}/ffi.rs wwr.capi
      COMMENT "Compiling wwr.capi.driver_rs (rustc)"
      VERBATIM
    )
    add_custom_target(
      wwr.capi.driver_rs ALL
      DEPENDS ${_rs_bin}
      COMMENT "Building the Rust FFI driver (wwr.capi.driver_rs)"
    )
    add_test(NAME wwr.capi.driver_rs COMMAND ${_rs_bin})
    set_tests_properties(wwr.capi.driver_rs PROPERTIES TIMEOUT 60)
  else()
    message(STATUS "wwr.capi: rustc not found -- skipping the Rust ABI test "
                   "(wwr.capi.driver_rs); the C driver stands in for it")
  endif()
endif()
