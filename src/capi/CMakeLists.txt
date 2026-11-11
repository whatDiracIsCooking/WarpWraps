# wwr.capi — the extern "C" ABI over wwr, for Rust FFI, split PER UNIT
#
# The generated extern "C" thunks (gen/<unit>/wwr_capi_<unit>.cpp) each `import`
# only the wwr module(s) their surface needs and forward into the wwr::wwrC* /
# wwr::wwr* / wwr::wwrblas* functions -- so the backend is chosen only by which
# wwr library a unit links, not by anything in the source. The status surface
# maps the neutral wwr status onto a stable wwr_status C enum below the extern
# "C" line (architecture.md section 6). driver.c is a pure-C consumer run as a
# ctest: it checks the symbols link, the complex numbers are right, and the
# status surface returns the stable enum (host-only, no GPU). See the directory
# README and docs/architecture.md sections 6/25 for why the thunk calls wwr.
#
# WHY PER UNIT. One libwwr_capi.so that linked every surface's wwr target also
# NEEDed every surface's vendor library -- so a consumer of the pure-host
# `complex` surface still dragged in libcublas via the `blas` surface. That
# collapses the project's own per-target dependency isolation (wwr.complex links
# nothing extra; wwr.blas links cuBLAS). So each surface gets its OWN shared
# object linking only its surface's vendor dep, and an UMBRELLA libwwr_capi.so
# that contains all unit symbols keeps every existing -lwwr_capi consumer (the
# C/Rust drivers via the combined gen/ffi.rs, example/capi-consumer) unchanged.
# The wwr-sys Rust crate goes further: it is split per surface into Cargo
# features over the per-unit .so's (src/capi/README.md). Proof: `readelf -d
# libwwr_capi_complex.so` shows NO cublas; `libwwr_capi_blas.so` DOES.
#
# Each unit's thunk compiles ONCE into an OBJECT library (wwr.capi.<unit>.obj)
# linked into both its own per-unit .so AND the umbrella .so -- so the umbrella
# carries every unit's symbols without recompiling. The object library carries
# the module-scan properties and its wwr deps PUBLIC, so both consumers inherit
# the right link line (and thus the right NEEDED set).
#
# Unlike the rest of wwr (which ships as module SOURCE a consumer compiles -- a
# BMI is not portable), the capi ships as prebuilt shared objects a non-C++
# consumer links. They do NOT ride the src/-sweep in wwr_install.cmake; they are
# installed explicitly there under WWR_INSTALL_CAPI (the sweep's non-recursive
# src/ collection never reaches this subdirectory). The targets build
# UNCONDITIONALLY with the rest of src/ -- only the install is gated.
#
# VERSION / SOVERSION below are the first declared contract version (1.0.0 /
# SOVERSION 1): the wwr_status enumerator values are frozen and SOVERSION bumps
# only on an ABI break. See the directory README's "Stability policy" and
# architecture.md section 21.

# One per-unit shared library over an object library compiled once. `unit` is
# the surface (complex / runtime_api / blas); the object library
# (wwr.capi.<unit>.obj) compiles gen/<unit>/wwr_capi_<unit>.cpp and carries its
# wwr deps PUBLIC, so both the unit .so here and the umbrella below inherit them
# (and so the right NEEDED set). The .so file is wwr_capi_<unit> (->
# libwwr_capi_<unit>.so), while the target keeps the convention's dotted
# wwr.capi.<unit> + wwr::capi::<unit> alias. A thunk is a plain TU, not a module
# (it exports none, it imports one), so CXX_MODULE_STD / CXX_SCAN_FOR_MODULES
# let it use the wwr modules, which import std.
function(_wwr_add_capi_unit unit)
  set(_obj wwr.capi.${unit}.obj)
  set(_lib wwr.capi.${unit})

  add_library(${_obj} OBJECT gen/${unit}/wwr_capi_${unit}.cpp)
  target_link_libraries(${_obj} PUBLIC ${ARGN})
  target_include_directories(${_obj} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gen)
  set_target_properties(
    ${_obj}
    PROPERTIES CXX_MODULE_STD 1
               CXX_SCAN_FOR_MODULES ON
               CUDA_SEPARABLE_COMPILATION OFF
               CUDA_RESOLVE_DEVICE_SYMBOLS OFF
               POSITION_INDEPENDENT_CODE ON
  )

  add_library(${_lib} SHARED $<TARGET_OBJECTS:${_obj}>)
  add_library(wwr::capi::${unit} ALIAS ${_lib})
  target_link_libraries(${_lib} PRIVATE ${_obj})
  set_target_properties(
    ${_lib}
    PROPERTIES OUTPUT_NAME wwr_capi_${unit}
               VERSION 1.0.0
               SOVERSION 1
  )
endfunction()

# complex -> no vendor math lib (wwr::complex links nothing extra); runtime_api
# -> the runtime host library (CUDA::cudart / hip::host) resolving the wwr*
# symbols via wwr::runtime_api::host; blas -> cuBLAS / hipBLAS, carried
# transitively by wwr::blas (LINK_PUBLIC on wwr.blas) with wwr::fp16 for the
# half forwarders. THIS is where the isolation lives: the blas unit alone NEEDs
# the vendor BLAS. The blas thunks run on the DEVICE at call time; CI
# link-checks them (no GPU), a GPU box exercises them (README, architecture.md).
_wwr_add_capi_unit(complex wwr::complex)
_wwr_add_capi_unit(runtime_api wwr::runtime_api wwr::runtime_api::host)
_wwr_add_capi_unit(blas wwr::blas wwr::fp16)
# The milestone-#6 library surfaces. Each imports ONLY its own wwr::<x> module,
# which carries its vendor library PUBLIC (cuBLASLt / cuSOLVER / cuRAND / nvRTC
# and the HIP equivalents) -- so each unit's .so NEEDs only that one vendor lib,
# and none of the others (verify: `readelf -d libwwr_capi_solver.so | grep
# NEEDED` shows cusolver, not cublasLt / curand / nvrtc). rtc is host-only (no
# device at call time); the other three execute on the device, so CI link-checks
# them and a GPU box exercises them, exactly as blas.
_wwr_add_capi_unit(blaslt wwr::blaslt)
_wwr_add_capi_unit(solver wwr::solver)
_wwr_add_capi_unit(rand wwr::rand)
_wwr_add_capi_unit(rtc wwr::rtc)
# The other milestone-#6 half (this PR): sparse / fft build unconditionally like
# blas (they import wwr.sparse + wwr.complex / wwr.fft + wwr.complex, each
# carrying only its own vendor lib -- cuSPARSE / cuFFT -- PUBLIC). tensor / comp
# are GATED on WWR_WITH_TENSOR / WWR_WITH_COMP, matching their wwr.tensor /
# wwr.comp modules in src/CMakeLists.txt (which exist only under those gates),
# so their unit .so and their umbrella object below are conditional. comp is the
# one hand-written library surface (its functions are wwr-authored shims, like
# complex); the rest are fragment-sourced.
_wwr_add_capi_unit(sparse wwr::sparse wwr::complex)
_wwr_add_capi_unit(fft wwr::fft wwr::complex)
# tx (NVTX / rocTX) is UNCONDITIONAL, like blas: wwr.tx builds in every
# configuration (no WWR_WITH_* gate). Its thunk imports only wwr.tx, which
# carries the tx backend PUBLIC -- nothing on CUDA (NVTX is header-only static
# inline, its forwarders compiled INTO wwr.tx), libroctx64 on HIP. Host-only (a
# profiler marker never touches the device), so it runs fully on no-GPU CI.
_wwr_add_capi_unit(tx wwr::tx)
if(WWR_WITH_TENSOR)
  _wwr_add_capi_unit(tensor wwr::tensor)
endif()
if(WWR_WITH_COMP)
  _wwr_add_capi_unit(comp wwr::comp)
endif()
# ccl (NCCL / RCCL) is GATED on WWR_WITH_CCL, matching its wwr.ccl module in
# src/CMakeLists.txt (which exists only under that gate) -- so its unit .so and
# its umbrella object below are conditional, exactly like tensor / comp. Its
# thunk imports only wwr.ccl, which carries the vendor collectives library
# PUBLIC (nccl / rccl). Every ccl call runs across the fabric, so CI link-checks
# it and a multi-GPU box exercises it.
if(WWR_WITH_CCL)
  _wwr_add_capi_unit(ccl wwr::ccl)
endif()

# The milestone-#6 numeric TYPE surfaces (#360 fp16+bf16, #361 vector_types,
# #362 fp8). Each imports ONLY its own wwr::<x> module, which carries just the
# vendor's HEADER (the half / bfloat16 / vector-type / fp8 headers are
# header-only -- no vendor math library), so each unit's .so NEEDs no extra
# vendor lib beyond the backend runtime (verify: `readelf -d
# libwwr_capi_fp16.so | grep NEEDED` shows no cublas / cufft / etc.). All four
# are HOST surfaces (type conversions / constructors), so they run on no-GPU CI.
# fp4 / fp6 are intentionally ABSENT: their wwr.fp4 / wwr.fp6 modules are
# BLOCKED on HIP and unregistered in src/CMakeLists.txt, so a unit importing
# them would fail the HIP compile gate -- they flip on the day those modules do
# (generate.py reports them as explicit skips). See src/capi/spec/fp8.py.
_wwr_add_capi_unit(fp16 wwr::fp16)
_wwr_add_capi_unit(bf16 wwr::bf16)
_wwr_add_capi_unit(vector_types wwr::vector_types)
_wwr_add_capi_unit(fp8 wwr::fp8)

# The umbrella libwwr_capi.so CONTAINS every unit's object symbols, so an
# existing -lwwr_capi consumer resolves the whole ABI from one .so. The OUTPUT
# file name is wwr_capi (-> libwwr_capi.so), NOT the dotted target name: rustc's
# `-l wwr_capi` / ffi.rs's `#[link(name = "wwr_capi")]` do a literal
# libwwr_capi.so lookup (and the underscore matches the C ABI's wwr_ symbol
# prefix). driver.c links via the wwr::capi target so it never saw the file
# name; the Rust driver does. generate.py emits the matching link name -- keep
# them in step. Linking the object libraries (not the per-unit .so's) gives the
# umbrella every unit's symbols AND, since each object library carries its wwr
# deps PUBLIC, the union of the unit NEEDED sets -- so the umbrella stays a
# drop-in for the old single .so.
# The gated units (tensor / comp) are appended only when built, so the umbrella
# aggregates exactly the units that exist. The unconditional ones are listed
# literally; _capi_gated_objs / _capi_gated_obj_targets carry the gated extras.
set(_capi_gated_objs "")
set(_capi_gated_obj_targets "")
if(WWR_WITH_TENSOR)
  list(APPEND _capi_gated_objs $<TARGET_OBJECTS:wwr.capi.tensor.obj>)
  list(APPEND _capi_gated_obj_targets wwr.capi.tensor.obj)
endif()
if(WWR_WITH_COMP)
  list(APPEND _capi_gated_objs $<TARGET_OBJECTS:wwr.capi.comp.obj>)
  list(APPEND _capi_gated_obj_targets wwr.capi.comp.obj)
endif()
if(WWR_WITH_CCL)
  list(APPEND _capi_gated_objs $<TARGET_OBJECTS:wwr.capi.ccl.obj>)
  list(APPEND _capi_gated_obj_targets wwr.capi.ccl.obj)
endif()

add_library(
  wwr.capi SHARED
  $<TARGET_OBJECTS:wwr.capi.complex.obj>
  $<TARGET_OBJECTS:wwr.capi.runtime_api.obj>
  $<TARGET_OBJECTS:wwr.capi.blas.obj>
  $<TARGET_OBJECTS:wwr.capi.blaslt.obj>
  $<TARGET_OBJECTS:wwr.capi.solver.obj>
  $<TARGET_OBJECTS:wwr.capi.rand.obj>
  $<TARGET_OBJECTS:wwr.capi.rtc.obj>
  $<TARGET_OBJECTS:wwr.capi.sparse.obj>
  $<TARGET_OBJECTS:wwr.capi.fft.obj>
  $<TARGET_OBJECTS:wwr.capi.tx.obj>
  $<TARGET_OBJECTS:wwr.capi.fp16.obj>
  $<TARGET_OBJECTS:wwr.capi.bf16.obj>
  $<TARGET_OBJECTS:wwr.capi.vector_types.obj>
  $<TARGET_OBJECTS:wwr.capi.fp8.obj>
  ${_capi_gated_objs}
)
add_library(wwr::capi ALIAS wwr.capi)
target_link_libraries(
  wwr.capi
  PRIVATE wwr.capi.complex.obj wwr.capi.runtime_api.obj wwr.capi.blas.obj
          wwr.capi.blaslt.obj wwr.capi.solver.obj wwr.capi.rand.obj
          wwr.capi.rtc.obj wwr.capi.sparse.obj wwr.capi.fft.obj
          wwr.capi.tx.obj
          wwr.capi.fp16.obj wwr.capi.bf16.obj wwr.capi.vector_types.obj
          wwr.capi.fp8.obj
          ${_capi_gated_obj_targets}
)
set_target_properties(
  wwr.capi
  PROPERTIES OUTPUT_NAME wwr_capi
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
  #
  # It links ONLY the two units it exercises -- complex (host math) and
  # runtime_api (host status) -- NOT the umbrella wwr::capi. The umbrella NEEDs
  # every unit's vendor lib, including hiptensor, whose on-load library init
  # PROBES for a GPU and aborts on a GPU-less HIP runner ("no ROCm-capable
  # device is detected"). So merely RUNNING an umbrella-linked binary fails on
  # no-GPU HIP CI (cuTENSOR does not probe on load, so CUDA was unaffected).
  # This driver calls no device-library symbol, so the two host units suffice.
  add_executable(wwr.capi.driver driver.c)
  target_link_libraries(
    wwr.capi.driver PRIVATE wwr::capi::complex wwr::capi::runtime_api
  )
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
    # Link ONLY the two units driver.rs actually CALLS -- complex + runtime_api
    # -- NOT the umbrella wwr_capi, for the same reason the C driver does (the
    # umbrella NEEDs hiptensor, which probes a GPU on load and aborts on no-GPU
    # HIP CI). The combined gen/ffi.rs still DECLARES every surface's externs,
    # but an unreferenced extern needs no `.so` at link time, so the two host
    # units suffice. -L the units' output dir (every per-unit .so lands there),
    # -l each by its OUTPUT_NAME, and an rpath so the ctest finds them without
    # LD_LIBRARY_PATH. WWR_CAPI_FFI points driver.rs's include! at the gen/ dir.
    add_custom_command(
      OUTPUT ${_rs_bin}
      COMMAND ${CMAKE_COMMAND} -E env WWR_CAPI_FFI=${_rs_gen}
              ${RUSTC} --edition 2021 -O
              -L $<TARGET_FILE_DIR:wwr.capi.complex>
              -l wwr_capi_complex
              -l wwr_capi_runtime_api
              -C link-arg=-Wl,-rpath,$<TARGET_FILE_DIR:wwr.capi.complex>
              -o ${_rs_bin} ${_rs_src}
      DEPENDS ${_rs_src} ${_rs_gen}/ffi.rs wwr.capi.complex wwr.capi.runtime_api
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
