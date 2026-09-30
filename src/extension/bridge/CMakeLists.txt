# Bridge header -- carries wwrrandState across the host/device boundary by
# #include, for translation units that cannot `import`. wwrStream_t's
# counterpart moved to the src/-root shared-type header runtime.h (a
# companion to complex.h) when stream_bridge.h was retired;
# rand_state_bridge.h stays because it forward-declares its struct rather
# than #including the heavy curand_kernel.h / hiprand_kernel.h, which a
# src/-root type header could not do.
#
# rand_state_bridge.h forward-declares one vendor struct and aliases it under a
# wwr* name, picked by src/selected_backend.h. A .cpp, a .cppm's global module
# fragment, a .cu or a .cuh reaches it; see the header's own comment.
#
# Header-only, so an INTERFACE library carrying only an include path -- and the
# backend selection behind the header is NOT its concern: on a device pass
# selected_backend.h reads __CUDACC__/__HIP__, and on a host compile the
# consumer links wwr_backend PRIVATE for the WWR_GPU_BACKEND_* define
# (src/extension/init_state and src/extension/random_normal do exactly this).
# This target grants neither the define nor a licence to #if on it.
add_library(wwr.extension.bridge INTERFACE)

# The include root is src/ itself, not this directory: the header is reached
# as "extension/bridge/rand_state_bridge.h" and itself includes the src/-root
# header "selected_backend.h". The installed tree mirrors src/ under
# include/wwr/ (cmake/wwr_install.cmake stages this directory there), so
# both spellings resolve unchanged after install.
target_include_directories(
  wwr.extension.bridge
  INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
)
