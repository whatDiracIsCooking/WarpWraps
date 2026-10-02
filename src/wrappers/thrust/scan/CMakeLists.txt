# wwr.wrappers.thrust.scan -- A-scan prefix-sum family (issue #240): typed,
# backend-neutral inclusive/exclusive/transform scans over the portable Thrust
# subset. Structured like src/extension/init_state: a device library holds the
# thrust::*_scan calls (a device TU cannot be a module unit -- architecture.md
# §8), the host module forwards to it across a bridge header (§14), and the
# device archive is forced whole into the module so its lazily-resolved members
# survive the link and export.

# The device-kernel half: scan.cu, device-compiled for the selected backend.
# wwr::thrust carries the backend's Thrust include dirs, the src/ include root
# (so "wrappers/thrust/execution_policy.cuh", the scan_bridge.h beside this
# file and "runtime.h" resolve) and wwr::device's runtime -- the one link this
# .cu needs, PRIVATE as every device-library link is.
wwr_add_gpu_device_library(
  NAME
  wwr.wrappers.thrust.scan.device
  SOURCES
  scan.cu
  LINK_PRIVATE
  wwr::thrust
)

# The module itself: ordinary host C++23, no device code, no backend flags.
# interface.cppm includes scan_bridge.h in its global module fragment, which
# pulls runtime.h (for wwrStream_t) -- a GMF cannot `import`, so the src/
# include root (INCLUDE_DIRS_PRIVATE) and wwr::backend (the WWR_GPU_BACKEND_*
# define behind runtime.h, since there is no device pass to read
# __CUDACC__/__HIP__) are both required, PRIVATE, as wwr.extension.init_state.
wwr_add_cxx_module_library(
  NAME
  wwr.wrappers.thrust.scan
  PRIMARY_INTERFACE
  interface.cppm
  LINK_PUBLIC
  wwr::runtime_api
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  IMPORT_STD
)

# Force the device archive member in, or the linker drops the explicit scan
# instantiations and consumers fail to link. The WHOLE_ARCHIVE genex and the
# BUILD/INSTALL spelling split are both needed for this to survive export -- see
# cmake/README.md, "Forcing a device archive into an exported target", same as
# wwr.extension.init_state.
set(_wa_build
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,wwr.wrappers.thrust.scan.device>"
)
set(_wa_install
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,wwr::wrappers::thrust::scan::device>"
)
target_link_libraries(
  wwr.wrappers.thrust.scan
  PUBLIC "$<BUILD_INTERFACE:${_wa_build}>" "$<INSTALL_INTERFACE:${_wa_install}>"
)
unset(_wa_build)
unset(_wa_install)

# interface.cppm's two compile requirements (the src/ include root for
# scan_bridge.h, and wwr_backend for the WWR_GPU_BACKEND_* define behind
# runtime.h), granted to installed consumers under $<INSTALL_INTERFACE:> because
# they are the ones compiling it -- see init_state's CMakeLists.txt and the
# comment at the end of src/CMakeLists.txt. PRIVATE for the in-tree build.
target_include_directories(
  wwr.wrappers.thrust.scan
  PUBLIC $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
  wwr.wrappers.thrust.scan PUBLIC $<INSTALL_INTERFACE:wwr_backend>
)
