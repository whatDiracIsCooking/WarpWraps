# wwr.thrust -- backend-neutral Thrust include dirs and link deps behind one
# name. Thrust is header-only vendor templates, so this is a bare INTERFACE
# target with no sources, no module and no manifest (nothing to harvest): it
# re-exports the backend's Thrust package and carries the wwr* headers written
# against it (execution_policy.cuh plus the per-header algorithm wrappers).
# CUDA resolves CCCL::Thrust (CCCL, which ships with the toolkit); HIP resolves
# rocThrust + rocPRIM. Both packages are found in the top-level CMakeLists.txt,
# with the version floors.
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

# The wwr* headers here are #included as "thrust/<name>" and pull the src/-root
# headers "device_guard.h" and "runtime.h" alongside them, so the include root
# is src/ itself (the installed tree mirrors src/ under include/wwr/, so the
# spelling survives install -- same as wwr.extension.parallel_for). wwr::device
# carries runtime.h's own include path and the runtime-only libs those headers
# need.
target_include_directories(
  wwr.thrust
  INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)
target_link_libraries(wwr.thrust INTERFACE wwr::device)
