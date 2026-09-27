# Backend-neutral parallel generation of standard-normal values into a typed
# device array, drawing from cuRAND/hipRAND DEVICE-API states initialized by
# gpumod.extension.init_state. See README.md.
#
# Split, with gpumod.extension.init_state, out of the single
# gpumod.extension.rand module (two partitions, one device library). Nothing
# here couples the two: they share only gpurandState, which comes from
# gpumod.rand.

# The device-kernel library (wwr_add_gpu_device_library handles the
# per-backend compilation; see cmake/). The link line is what is local: PRIVATE,
# as every device-library link is. gpumod.extension.parallel_for carries
# ${CMAKE_SOURCE_DIR}/src (so random_normal_bridge.h and rand.cuh resolve) and
# gpumod.device; gpumod.rand.device adds the cuRAND/hipRAND library.
wwr_add_gpu_device_library(
  NAME
  gpumod.extension.random_normal.device
  SOURCES
  random_normal.cu
  LINK_PRIVATE
  gpumod.extension.parallel_for
  gpumod.rand.device
)

# The module itself: ordinary host C++23, no device code, no backend flags.
#
# INCLUDE_DIRS_PRIVATE is for random_normal_bridge.h, which interface.cppm
# includes in its global module fragment -- the only way a module unit can
# declare a function that a plain TU defines (see random_normal_bridge.h's own
# header).
#
# LINK_PRIVATE gpumod_backend is for the two bridges random_normal_bridge.h
# pulls in there (extension/bridge/gpu_stream_bridge.h,
# extension/bridge/rand_state_bridge.h): a GMF cannot `import`, so gpuStream_t
# and gpurandState have to arrive by #include, and selected_backend.h behind
# them reads WWR_GPU_BACKEND_{CUDA,HIP} when there is no device-compile pass
# to read __CUDACC__/__HIP__ from. PRIVATE, so the define reaches these module
# units and nothing that imports them.
wwr_add_cxx_module_library(
  NAME
  gpumod.extension.random_normal
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  gpumod.runtime_api
  gpumod.rand
  gpumod.complex
  gpumod.fp16
  gpumod.bf16
  LINK_PRIVATE
  gpumod_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  IMPORT_STD
)

# Force every member of the device archive in, or the explicit
# device::random_normal<T> instantiations in random_normal.cu are dropped (the
# module only names them through lazily-resolved calls) and consumers fail to
# link. The WHOLE_ARCHIVE genex and the BUILD/INSTALL spelling split are both
# needed for this to survive export -- see cmake/README.md, "Forcing a device
# archive into an exported target".
set(_wa_build
  "$<LINK_LIBRARY:WHOLE_ARCHIVE,gpumod.extension.random_normal.device>"
)
set(_wa_install
  "$<LINK_LIBRARY:WHOLE_ARCHIVE,gpumod::gpumod.extension.random_normal.device>"
)
target_link_libraries(
  gpumod.extension.random_normal
  PUBLIC "$<BUILD_INTERFACE:${_wa_build}>"
         "$<INSTALL_INTERFACE:${_wa_install}>"
)
unset(_wa_build)
unset(_wa_install)

# interface.cppm and instantiations.cpp's two compile requirements (the src/
# include root for random_normal_bridge.h, and gpumod_backend for the
# WWR_GPU_BACKEND_* define behind the bridges), granted to installed
# consumers under $<INSTALL_INTERFACE:> because they are the ones compiling them
# -- see the comment at the end of src/CMakeLists.txt. PRIVATE for the in-tree
# build.
target_include_directories(
  gpumod.extension.random_normal
  PUBLIC $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
  gpumod.extension.random_normal PUBLIC $<INSTALL_INTERFACE:gpumod_backend>
)
