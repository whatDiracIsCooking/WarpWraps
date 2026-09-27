# Bridge headers -- carry wwrStream_t / wwrrandState across the host/device
# boundary by #include, for translation units that cannot `import`.
#
# gpu_stream_bridge.h and rand_state_bridge.h each forward-declare (or minimally
# include) one vendor type and alias it under a gpu* name, picked by
# src/selected_backend.h. A .cpp, a .cppm's global module fragment, a .cu or a
# .cuh reaches them; see each header's own comment.
#
# Header-only, so an INTERFACE library carrying only an include path -- and the
# backend selection behind these headers is NOT its concern: on a device pass
# selected_backend.h reads __CUDACC__/__HIP__, and on a host compile the
# consumer links wwr_backend PRIVATE for the WWR_GPU_BACKEND_* define
# (src/extension/init_state and src/extension/random_normal do exactly this).
# This target grants neither the define nor a licence to #if on it.
add_library(wwr.extension.bridge INTERFACE)

# The include root is src/ itself, not this directory: the headers are reached
# as "extension/bridge/gpu_stream_bridge.h" and themselves include the src/-root
# header "selected_backend.h". The installed tree mirrors src/ under
# include/wwr/ (cmake/wwr_install.cmake stages this directory there), so
# both spellings resolve unchanged after install.
target_include_directories(
  wwr.extension.bridge
  INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)
