# wwr.wrappers.thrust.transform -- A-transform elementwise family (issue
# #241): typed, backend-neutral fill / sequence / replace / transform over the
# portable Thrust subset. Structured like src/extension/random_normal: a device
# library holds the thrust::<algo> calls (a device TU cannot be a module unit
# -- architecture.md §8), the host module forwards to it across a bridge header
# (§14), and the device archive is forced whole into the module so its
# lazily-resolved members survive the link and export. The functor-taking half
# (algorithms.cuh) is a header template with no target of its own -- a
# consumer's own device TU #includes it.

# The device-kernel half: transform.cu, device-compiled for the selected
# backend. wwr::thrust carries the backend's Thrust include dirs, the src/
# include root (so "wrappers/thrust/execution_policy.cuh", the
# transform_bridge.h beside this file, "runtime.h" and "complex.h" resolve) and
# wwr::device's runtime -- the one link this .cu needs, PRIVATE as every
# device-library link is.
wwr_add_gpu_device_library(
  NAME
  wwr.wrappers.thrust.transform.device
  SOURCES
  transform.cu
  LINK_PRIVATE
  wwr::thrust
)

# The module itself: ordinary host C++23, no device code, no backend flags.
# interface.cppm includes transform_bridge.h in its global module fragment,
# which pulls runtime.h (for wwrStream_t) -- a GMF cannot `import`, so the src/
# include root (INCLUDE_DIRS_PRIVATE) and wwr::backend (the WWR_GPU_BACKEND_*
# define behind runtime.h, since there is no device pass to read
# __CUDACC__/__HIP__) are both required, PRIVATE, as in
# wwr.extension.random_normal. wwr::complex is PUBLIC: the complex element types
# appear in the exported wrapper signatures.
wwr_add_cxx_module_library(
  NAME
  wwr.wrappers.thrust.transform
  PRIMARY_INTERFACE
  interface.cppm
  LINK_PUBLIC
  wwr::runtime_api
  wwr::complex
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  IMPORT_STD
)

# Force the device archive member in, or the linker drops the explicit transform
# instantiations and consumers fail to link. The WHOLE_ARCHIVE genex and the
# BUILD/INSTALL spelling split are both needed for this to survive export -- see
# cmake/README.md, "Forcing a device archive into an exported target", same as
# wwr.extension.random_normal.
set(_wa_build
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,wwr.wrappers.thrust.transform.device>"
)
set(_wa_install
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,wwr::wrappers::thrust::transform::device>"
)
target_link_libraries(
  wwr.wrappers.thrust.transform
  PUBLIC "$<BUILD_INTERFACE:${_wa_build}>" "$<INSTALL_INTERFACE:${_wa_install}>"
)
unset(_wa_build)
unset(_wa_install)

# interface.cppm's two compile requirements (the src/ include root for
# transform_bridge.h, and wwr_backend for the WWR_GPU_BACKEND_* define behind
# runtime.h), granted to installed consumers under $<INSTALL_INTERFACE:> because
# they are the ones compiling it -- see random_normal's CMakeLists.txt and the
# comment at the end of src/CMakeLists.txt. PRIVATE for the in-tree build.
target_include_directories(
  wwr.wrappers.thrust.transform
  PUBLIC $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
  wwr.wrappers.thrust.transform PUBLIC $<INSTALL_INTERFACE:wwr_backend>
)
