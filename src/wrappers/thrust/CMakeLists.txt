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
# Thrust A-families -- one per-family subdirectory, each self-contained (its own
# module, device archive, bridge header and CMakeLists). One add_subdirectory
# line per family, one per line, so parallel family branches union-merge
# cleanly. Keep this block alphabetical.
# ============================================================================
add_subdirectory(reorder)
add_subdirectory(scan)
add_subdirectory(reduce)
add_subdirectory(transform)
