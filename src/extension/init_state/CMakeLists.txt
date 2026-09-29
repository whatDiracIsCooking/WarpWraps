# Backend-neutral parallel initialization of cuRAND/hipRAND DEVICE-API
# generator states -- one state per element, advanced later inside a kernel by
# wwr.extension.random_normal. See README.md.
#
# Split, with wwr.extension.random_normal, out of the single
# wwr.extension.rand module (two partitions, one device library). Nothing
# here couples the two: they share only wwrrandState, which comes from
# wwr.rand.

# The device-kernel library (wwr_add_gpu_device_library handles the
# per-backend compilation; see cmake/). The link line is what is local: PRIVATE,
# as every device-library link is. wwr.extension.parallel_for carries
# ${CMAKE_SOURCE_DIR}/src (so init_state_bridge.h and rand.cuh resolve) and
# wwr.device; wwr.rand.device adds the cuRAND/hipRAND library.
wwr_add_gpu_device_library(
  NAME
  wwr.extension.init_state.device
  SOURCES
  init_state.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
  wwr.rand.device
)

# The module itself: ordinary host C++23, no device code, no backend flags.
# init_state is a single non-template entry point, so there is no partition and
# no implementation unit -- the wrapper is defined in interface.cppm and there
# are no explicit template instantiations to keep in step (unlike
# wwr.extension.random_normal).
#
# INCLUDE_DIRS_PRIVATE is for init_state_bridge.h, which interface.cppm includes
# in its global module fragment -- the only way a module unit can declare a
# function that a plain TU defines (see init_state_bridge.h's own header).
#
# LINK_PRIVATE wwr_backend is for the two bridges init_state_bridge.h pulls
# in there (extension/bridge/stream_bridge.h,
# extension/bridge/rand_state_bridge.h): a GMF cannot `import`, so wwrStream_t
# and wwrrandState have to arrive by #include, and selected_backend.h behind
# them reads WWR_GPU_BACKEND_{CUDA,HIP} when there is no device-compile pass
# to read __CUDACC__/__HIP__ from. PRIVATE, so the define reaches this module
# unit and nothing that imports it.
wwr_add_cxx_module_library(
  NAME
  wwr.extension.init_state
  PRIMARY_INTERFACE
  interface.cppm
  LINK_PUBLIC
  wwr.runtime_api
  wwr.rand
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  IMPORT_STD
)

# Force the device archive member in, or the linker drops device::init_state's
# lazily-resolved object and consumers fail to link. The WHOLE_ARCHIVE genex and
# the BUILD/INSTALL spelling split are both needed for this to survive export --
# see cmake/README.md, "Forcing a device archive into an exported target".
set(_wa_build
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,wwr.extension.init_state.device>"
)
set(_wa_install
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,wwr::wwr.extension.init_state.device>"
)
target_link_libraries(
  wwr.extension.init_state
  PUBLIC "$<BUILD_INTERFACE:${_wa_build}>"
         "$<INSTALL_INTERFACE:${_wa_install}>"
)
unset(_wa_build)
unset(_wa_install)

# interface.cppm's two compile requirements (the src/ include root for
# init_state_bridge.h, and wwr_backend for the WWR_GPU_BACKEND_* define
# behind the bridges), granted to installed consumers under
# $<INSTALL_INTERFACE:> because they are the ones compiling it -- see the
# comment at the end of src/CMakeLists.txt. PRIVATE for the in-tree build,
# unchanged.
target_include_directories(
  wwr.extension.init_state
  PUBLIC $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
  wwr.extension.init_state PUBLIC $<INSTALL_INTERFACE:wwr_backend>
)
