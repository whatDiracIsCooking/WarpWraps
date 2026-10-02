# wwr.thrust -- backend-neutral Thrust include dirs and link deps behind one
# name. Thrust is header-only vendor templates, so this is a bare INTERFACE
# target with no sources, no module and no manifest (nothing to harvest, §21):
# it only re-exports the backend's Thrust package. CUDA resolves CCCL::Thrust
# (CCCL, which ships with the toolkit); HIP resolves rocThrust + rocPRIM. Both
# packages are found in the top-level CMakeLists.txt, with the version floors.
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
