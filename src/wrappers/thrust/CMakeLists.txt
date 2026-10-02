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

# ===========================================================================
# A-transform: elementwise family (issue #241)
# ---------------------------------------------------------------------------
# Self-contained block -- sibling A-families (sort/reduce/scan) append their own
# below; keep each additive and separated so the serialized merge is trivial.
#
# Follows the src/wrappers/blas + src/extension/random_normal template: a host
# C++23 module (elementwise.cppm) whose exported wrappers forward to a device
# archive (elementwise.cu) holding the actual
# thrust::<algo>(wwr::par_on(stream), ...) calls, bound at the link boundary by
# extern template
# (§8/§14). elementwise_bridge.h carries the device:: declarations across that
# boundary; algorithms.cuh is the header-template, caller-functor half (no CMake
# target of its own -- it is #included into a consumer's device TU).

# The device-kernel archive: elementwise.cu, device-compiled for the backend.
# wwr::thrust carries the Thrust include dirs, the src/ root (so the
# "wrappers/thrust/*.cuh" / "wrappers/thrust/elementwise_bridge.h" / "runtime.h"
# / "complex.h" spellings resolve) and wwr::device's runtime. PRIVATE, as every
# device-library link is.
wwr_add_gpu_device_library(
  NAME
  wwr.wrappers.thrust.elementwise.device
  SOURCES
  elementwise.cu
  LINK_PRIVATE
  wwr::thrust
)

# The module itself: ordinary host C++23, no device code. INCLUDE_DIRS_PRIVATE
# is the src/ root for elementwise_bridge.h (included in the global module
# fragment -- the only way a module unit declares a function a plain TU
# defines). LINK_PRIVATE wwr::backend is for runtime.h behind that bridge
# header: a GMF cannot import, so wwrStream_t arrives by #include and
# selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP} (no device
# pass here to read __CUDACC__).
wwr_add_cxx_module_library(
  NAME
  wwr.wrappers.thrust.elementwise
  PRIMARY_INTERFACE
  elementwise.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::runtime_api
  wwr::complex
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  IMPORT_STD
)

# Force every member of the device archive in, or the explicit
# device::* instantiations in elementwise.cu are dropped (the module names them
# only through lazily-resolved calls) and consumers fail to link. The
# WHOLE_ARCHIVE genex and the BUILD/INSTALL spelling split both survive export
# -- see cmake/README.md, "Forcing a device archive into an exported target",
# and src/extension/random_normal for the same pattern.
set(_wa_build
  "$<LINK_LIBRARY:WHOLE_ARCHIVE,wwr.wrappers.thrust.elementwise.device>"
)
set(_wa_install
  "$<LINK_LIBRARY:WHOLE_ARCHIVE,wwr::wrappers::thrust::elementwise::device>"
)
target_link_libraries(
  wwr.wrappers.thrust.elementwise
  PUBLIC "$<BUILD_INTERFACE:${_wa_build}>"
         "$<INSTALL_INTERFACE:${_wa_install}>"
)
unset(_wa_build)
unset(_wa_install)

# The src/ include root and wwr_backend compile requirement, granted to
# installed consumers under $<INSTALL_INTERFACE:> because they compile the
# module units -- same shape as src/extension/random_normal.
target_include_directories(
  wwr.wrappers.thrust.elementwise
  PUBLIC $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
  wwr.wrappers.thrust.elementwise PUBLIC $<INSTALL_INTERFACE:wwr_backend>
)
# ===========================================================================
