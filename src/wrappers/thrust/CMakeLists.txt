# wwr.thrust -- backend-neutral Thrust include dirs and link deps behind one
# name. Thrust is header-only vendor templates, so this is a bare INTERFACE
# target with no sources, no module and no manifest (nothing to harvest, §21):
# it re-exports the backend's Thrust package and carries the one wwr* header
# written against it, execution_policy.cuh. CUDA resolves CCCL::Thrust (CCCL,
# which ships with the toolkit); HIP resolves rocThrust + rocPRIM. Both packages
# are found in the top-level CMakeLists.txt, with the version floors.
add_library(wwr.thrust INTERFACE)
# In-tree :: alias, spelled by hand because the dot-less rule that turns dots
# into :: has nothing to turn here -- same as wwr::module_flags. Dependants link
# the guarded wwr::thrust spelling (a :: name resolving to nothing is a hard
# configure error; a bare one silently degrades to -lname).
add_library(wwr::thrust ALIAS wwr.thrust)

if(WWR_GPU_BACKEND STREQUAL "CUDA")
  target_link_libraries(wwr.thrust INTERFACE CCCL::Thrust)
else()
  target_link_libraries(wwr.thrust INTERFACE roc::rocthrust roc::rocprim)
endif()

# execution_policy.cuh is #included as "wrappers/thrust/execution_policy.cuh"
# and pulls the src/-root headers "device_guard.h" and "runtime.h" alongside it,
# so the include root is src/ itself (the installed tree mirrors src/ under
# include/wwr/, so the spelling survives install -- same as
# wwr.extension.parallel_for). wwr::device carries runtime.h's own include path
# and the runtime-only libs those two headers need.
target_include_directories(
  wwr.thrust
  INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)
target_link_libraries(wwr.thrust INTERFACE wwr::device)

# ============================================================================
# A-sort family (issue #238): wwr.wrappers.thrust.reorder
#
# Self-contained block -- the four A-families each append one like it; keep the
# additions here and in test/extension/thrust so the coordinator's merge is
# trivially additive.
#
# The reorder family mirrors wwr.extension.random_normal: a device archive
# (reorder.cu, thrust::<algo>(wwr::par_on(stream), ...) with explicit per-type
# instantiations) forced whole into a host module (reorder.cppm) whose exported
# wrappers forward to it. reorder_bridge.h carries the typed declarations across
# that boundary; it is #included in the module's GMF, so its include root and
# backend define are both compile requirements of the module.
# ============================================================================

# Device half. wwr::thrust carries the backend's Thrust include dirs, the src/
# include root (so "wrappers/thrust/execution_policy.cuh", "reorder_bridge.h"
# and "runtime.h" resolve) and wwr::device's runtime.
wwr_add_gpu_device_library(
  NAME
  wwr.wrappers.thrust.reorder.device
  SOURCES
  reorder.cu
  LINK_PRIVATE
  wwr::thrust
)

# The module: ordinary host C++23, no device code. INCLUDE_DIRS_PRIVATE is for
# reorder_bridge.h, which reorder.cppm includes in its global module fragment;
# LINK_PRIVATE wwr::backend is for the WWR_GPU_BACKEND_* define behind the
# runtime.h that bridge pulls in, and wwr::device for the backend runtime's own
# include path (runtime.h reaches cuda_runtime_api.h / hip_runtime_api.h) and
# libraries. Both PRIVATE: a GMF cannot import, so wwrStream_t arrives by
# #include, and neither requirement reaches an importer.
wwr_add_cxx_module_library(
  NAME
  wwr.wrappers.thrust.reorder
  PRIMARY_INTERFACE
  reorder.cppm
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

# reorder.cppm and instantiations.cpp's two compile requirements (the src/
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

# A-family algorithm wrappers (issues #238-#241), each a self-contained
# subdirectory built on wwr::thrust and F2's execution_policy.cuh. One
# add_subdirectory per family; keep them one-per-line so the four parallel
# family branches merge additively.
add_subdirectory(scan)
