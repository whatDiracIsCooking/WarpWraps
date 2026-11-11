# wwr_add_gpu_device_library — a STATIC library of device-compiled `.cu`
# sources, built for whichever backend the build selected.
#
# A `.cu` here means "device pass", not "nvcc". Keeping device sources in their
# own library is what keeps their language and `-x hip` flags off the host CXX
# module compiles. The call sites are src/extension/init_state and
# src/extension/random_normal.
#
# See cmake/README.md for the interface, what it deliberately omits, and what it
# replaced; docs/architecture.md §17 for how a `.cu` is compiled under HIP.
#
# wwr_add_gpu_device_library(
#   NAME    <target, by convention <module>.device>
#   SOURCES <file.cu ...>
#   [LINK_PRIVATE <lib ...>])
macro(wwr_add_gpu_device_library)
  cmake_parse_arguments(
    _GDL
    ""
    "NAME"
    "SOURCES;LINK_PRIVATE"
    ${ARGN}
  )

  _wwr_require_args("wwr_add_gpu_device_library" _GDL NAME SOURCES)

  add_library(${_GDL_NAME} STATIC ${_GDL_SOURCES})

  if(_GDL_LINK_PRIVATE)
    target_link_libraries(${_GDL_NAME} PRIVATE ${_GDL_LINK_PRIVATE})
  endif()

  if(WWR_GPU_BACKEND STREQUAL "CUDA")
    # .cu maps to CUDA by extension; nothing to set for the language. Separable
    # compilation OFF, device symbols unresolved to match —
    # docs/architecture.md §17.
    set_target_properties(
      ${_GDL_NAME}
      PROPERTIES CUDA_STANDARD 20
                 CUDA_STANDARD_REQUIRED ON
                 POSITION_INDEPENDENT_CODE ON
    )
    _wwr_disable_cuda_device_linking(${_GDL_NAME})
  else()
    # A HIP build enables no CUDA language, so .cu must be forced back to CXX;
    # that is also what lets hip::device's $<COMPILE_LANGUAGE:CXX>-gated
    # `-x hip` reach these sources. See docs/architecture.md §17. (A macro opens
    # no directory scope, so these relative names resolve in the caller's
    # directory, where set_source_files_properties looks.)
    set_source_files_properties(${_GDL_SOURCES} PROPERTIES LANGUAGE CXX)

    # hip::device PRIVATE: its `-x hip` must reach this library's own sources
    # and nothing else — src/hip/README.md, "Link hip::host, never
    # hip::device".
    target_link_libraries(${_GDL_NAME} PRIVATE hip::device)

    # NO `-B<rocm>/llvm/bin` HERE ANY MORE, and that is the fix rather than an
    # omission: the device link's `ld.lld` now comes from the image itself
    # (docker/Dockerfile.base installs lld-${LLVM_VERSION}), and borrowing
    # ROCm's also handed its older clang-linker-wrapper the bitcode THIS clang
    # produced. No -Wno-linker-warnings either -- the data-layout warning it hid
    # is gone. Both, with the measurements: docs/architecture.md §17.
  endif()

endmacro()
