# Parallel For Infrastructure - backend-neutral, header-only device
# kernel launcher
#
# Header-only library providing parallel_for(stream, count, functor): a
# hand-written index-per-thread __global__ kernel launch (no grid-stride loop,
# 4 warps per block -- 128 threads at the default WWR_WARP_SIZE of 32,
# 256 on a CDNA build that sets 64). Unlike everything else under
# src/extension, this is a device-code header (parallel_for.cuh), not a C++20
# module -- consumers #include it directly into a .cu (CUDA) or
# device-compiled (HIP) translation unit. See parallel_for.cuh's own header
# comment for why gpu_backend.h's module-name-switch technique does not apply
# here, and how backend selection is done instead (src/runtime.cuh,
# keyed off the compiler's own __CUDACC__ / __HIP__ / __HIPCC__ device-compile
# macros).
#
# NOTE: this replaced src/cuda/extension/parallel_for's CUDA-only,
# Thrust-based `parallel_for` target (removed; last at tag `pre-split`). It is
# named wwr.extension.parallel_for, not bare `parallel_for`, because
# the two targets once coexisted in one CMake run and CMake target names are
# global.

add_library(wwr.extension.parallel_for INTERFACE)

# The include root is src/ itself, not this directory: parallel_for.cuh is
# reached as "extension/parallel_for/parallel_for.cuh" and pulls in the
# src/-root header "runtime.cuh" alongside it. The installed tree mirrors src/
# under include/wwr/ for exactly that reason, so the spelling resolves
# unchanged
# after install.
target_include_directories(
  wwr.extension.parallel_for
  INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)

# wwr.device (src/CMakeLists.txt) carries runtime.cuh's
# include path plus the runtime-only libs (CUDA::cudart / hip::host, never
# hip::device -- see that target's own comment). A real device-KERNEL
# consumer of THIS library still links hip::device itself, scoped to a small
# dedicated library holding only its own device-kernel source -- see
# cmake/wwr_add_gpu_device_library.cmake, which is that pattern, and its
# call sites in src/extension/init_state and src/extension/random_normal.
target_link_libraries(wwr.extension.parallel_for INTERFACE wwr.device)

# wwr.extension.bridge carries the same src/ include root, named explicitly
# because parallel_for.cuh includes "extension/bridge/gpu_stream_bridge.h": the
# dependency on the bridge headers is real and should be declared, not left to
# ride wwr.device's include path.
target_link_libraries(
  wwr.extension.parallel_for INTERFACE wwr.extension.bridge
)
