# wwr.wrappers.thrust.reorder -- A-sort reorder family (issue #238): portable,
# typed sort/unique/partition/remove/copy_if/reverse over the Thrust subset.
# Structured like scan/ and src/extension/random_normal: a device library holds
# the thrust::<algo> calls (a device TU cannot be a module unit --
# architecture.md §8), the host module forwards to it across a bridge header
# (§14), and the device archive is forced whole into the module so its
# lazily-resolved members survive the link and export. Unlike scan, the module
# exports TEMPLATE wrappers declared `extern template`; instantiations.cpp is
# their one instantiation point (reorder.cu instantiates the device half).

# Device half. wwr::thrust carries the backend's Thrust include dirs, the src/
# include root (so "wrappers/thrust/execution_policy.cuh", the reorder_bridge.h
# beside this file and "runtime.h" resolve) and wwr::device's runtime.
wwr_add_gpu_device_library(
  NAME
  wwr.wrappers.thrust.reorder.device
  SOURCES
  reorder.cu
  LINK_PRIVATE
  wwr::thrust
)

# The module: ordinary host C++23, no device code. INCLUDE_DIRS_PRIVATE is for
# reorder_bridge.h, which interface.cppm includes in its global module fragment;
# LINK_PRIVATE wwr::backend is for the WWR_GPU_BACKEND_* define behind the
# runtime.h that bridge pulls in, and wwr::device for the backend runtime's own
# include path (runtime.h reaches cuda_runtime_api.h / hip_runtime_api.h) and
# libraries. Both PRIVATE: a GMF cannot import, so wwrStream_t arrives by
# #include, and neither requirement reaches an importer.
wwr_add_cxx_module_library(
  NAME
  wwr.wrappers.thrust.reorder
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::runtime_api
  LINK_PRIVATE
  wwr::backend
  wwr::device
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  IMPORT_STD
)

# Force every member of the device archive in, or reorder.cu's explicit
# device::* instantiations are dropped (the module only names them through
# lazily-resolved calls) and consumers fail to link. The WHOLE_ARCHIVE genex and
# the BUILD/INSTALL spelling split both survive export -- see cmake/README.md,
# "Forcing a device archive into an exported target".
set(_wa_build
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,wwr.wrappers.thrust.reorder.device>"
)
set(_wa_install
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,wwr::wrappers::thrust::reorder::device>"
)
target_link_libraries(
  wwr.wrappers.thrust.reorder
  PUBLIC "$<BUILD_INTERFACE:${_wa_build}>"
         "$<INSTALL_INTERFACE:${_wa_install}>"
)
unset(_wa_build)
unset(_wa_install)

# interface.cppm and instantiations.cpp's two compile requirements (the src/
# include root for reorder_bridge.h, and wwr_backend for the WWR_GPU_BACKEND_*
# define behind runtime.h), granted to installed consumers under
# $<INSTALL_INTERFACE:> because they are the ones compiling them -- see the
# comment at the end of src/CMakeLists.txt. PRIVATE for the in-tree build.
target_include_directories(
  wwr.wrappers.thrust.reorder
  PUBLIC $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
  wwr.wrappers.thrust.reorder PUBLIC $<INSTALL_INTERFACE:wwr_backend>
)
