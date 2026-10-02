# wwr.wrappers.thrust.reduce -- portable, typed wrappers for the Thrust
# reduction family (reduce, transform_reduce, count/count_if, inner_product,
# min_element/max_element), each dispatched through wwr::par_on(stream).
#
# Split like src/extension/init_state: a device-kernel archive (reduce.cu,
# device-compiled for the selected backend, where the thrust:: calls live) plus
# an ordinary host C++23 module (interface.cppm) that forwards to it across the
# reduce_bridge.h boundary. The module's extern-template declarations bind to
# the archive's explicit instantiations; the archive is forced in whole so the
# linker keeps those object members. See init_state for the canonical form.

# The device-kernel half. wwr::thrust carries the backend's Thrust include dirs,
# the src/ include root (so "wrappers/thrust/execution_policy.cuh",
# "wrappers/thrust/reduce/reduce_bridge.h" and "runtime.h" resolve) and
# wwr::device's runtime. The link is PRIVATE, as every device-library link is.
wwr_add_gpu_device_library(
  NAME
  wwr.wrappers.thrust.reduce.device
  SOURCES
  reduce.cu
  LINK_PRIVATE
  wwr::thrust
)

# The module itself: ordinary host C++23, no device code. reduce_bridge.h is
# #included in interface.cppm's global module fragment and pulls the src/-root
# header runtime.h for wwrStream_t; runtime.h reaches selected_backend.h, which
# in a host compile reads WWR_GPU_BACKEND_* -- so this module links wwr::backend
# PRIVATE for that define and carries the src/ include root, exactly as
# init_state does for the same GMF-#include reason. Both are compile
# requirements, so both are re-granted under $<INSTALL_INTERFACE:> below (an
# installed consumer compiles this .cppm itself).
wwr_add_cxx_module_library(
  NAME
  wwr.wrappers.thrust.reduce
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

# Force the device archive member in, or the linker drops the explicit
# instantiations the module's extern-template declarations resolve against and
# consumers fail to link. The WHOLE_ARCHIVE genex and the BUILD/INSTALL spelling
# split both survive export -- see cmake/README.md, "Forcing a device archive
# into an exported target", and the identical block in src/extension/init_state.
set(_wa_build
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,wwr.wrappers.thrust.reduce.device>"
)
set(_wa_install
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,wwr::wrappers::thrust::reduce::device>"
)
target_link_libraries(
  wwr.wrappers.thrust.reduce
  PUBLIC "$<BUILD_INTERFACE:${_wa_build}>"
         "$<INSTALL_INTERFACE:${_wa_install}>"
)
unset(_wa_build)
unset(_wa_install)

# interface.cppm's two compile requirements (the src/ include root for
# reduce_bridge.h -> runtime.h, and wwr_backend for the WWR_GPU_BACKEND_* define
# behind selected_backend.h), re-granted to installed consumers under
# $<INSTALL_INTERFACE:> because they compile it -- the same grant init_state
# makes. PRIVATE for the in-tree build, unchanged.
target_include_directories(
  wwr.wrappers.thrust.reduce
  PUBLIC $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
  wwr.wrappers.thrust.reduce PUBLIC $<INSTALL_INTERFACE:wwr_backend>
)
