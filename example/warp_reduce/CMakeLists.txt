# The warp-reduction example: a hand-written warp-level kernel on
# cooperative_groups.cuh, and the host code that runs it.
#
# This is the in-tree consumer that keeps src/cooperative_groups.cuh honest.
# test/gpu/cooperative_groups.cu compiles every portable entity the header
# reaches; nothing launched one until this, so the shfl_down ladder's answer was
# never checked on a device. Deliberately an example rather than a
# gpumod.extension.* module: see example/README.md.
#
# Two targets because a .cu and a C++23 module TU cannot share one. The device
# source needs a language decision and, under HIP, `-x hip --offload-arch=...`;
# main.cpp is an ordinary host CXX compile that imports modules and must get
# neither. cmake/gpumod_add_gpu_device_library.cmake is where that split is
# spelled out in full.

# gpumod.device carries cooperative_groups.cuh and runtime.cuh (and the
# GPUMOD_WARP_SIZE the tile size is built from); gpumod.extension.bridge carries
# "extension/bridge/gpu_stream_bridge.h", which warp_reduce_bridge.h includes.
# PRIVATE, as every device-library link is: nothing that links this should
# inherit a device-code include path.
gpumod_add_gpu_device_library(
  NAME
  gpumod.example.warp_reduce.device
  SOURCES
  warp_reduce.cu
  LINK_PRIVATE
  gpumod.device
  gpumod.extension.bridge
)

# The host half. No ctest entry: it needs a real device, and the runtime tier
# currently has nothing device-dependent in it (see CLAUDE.md, "Running
# tests") -- registering this would turn every driverless box red. Run it by
# hand from the build directory.
add_executable(gpumod.example.warp_reduce main.cpp)

# gpumod.extension.bridge for gpu_stream_bridge.h, and gpumod_backend for the
# GPUMOD_GPU_BACKEND_* define that selected_backend.h behind it reads when
# there is no device-compile pass to supply __CUDACC__ / __HIP__ -- the same
# pair, for the same reason, that src/extension/rand's module links.
target_link_libraries(
  gpumod.example.warp_reduce
  PRIVATE gpumod.runtime_api
          gpumod.extension.runtime
          gpumod.extension.memory_buffer
          gpumod.extension.bridge
          gpumod_backend
          gpumod.example.warp_reduce.device
)

set_target_properties(
  gpumod.example.warp_reduce
  PROPERTIES CXX_MODULE_STD 1 CUDA_SEPARABLE_COMPILATION OFF
             CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
