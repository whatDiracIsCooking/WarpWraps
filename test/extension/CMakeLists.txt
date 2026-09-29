# Backend-neutral extension tests -- src/extension's tests, written against the
# wwr* names and built for either WWR_GPU_BACKEND.
#
# The build-time halves (static_asserts) must be configured under
# WWR_COMPILE_TIME_ONLY too, so their directories are added before the early
# return; the GoogleTest runtime suites come after it.
add_subdirectory(build_time)
add_subdirectory(common)

# Everything below is a GoogleTest runtime test.
if(WWR_COMPILE_TIME_ONLY)
  return()
endif()

add_subdirectory(runtime)
add_subdirectory(handle)
add_subdirectory(memory_buffer)
# The RNG extension pair (init_state + random_normal) has no build-time half --
# the cuRAND/hipRAND device functions it wraps cannot be called from host code,
# so running the kernels IS the test. One integration suite covers both.
add_subdirectory(rand)

# The library-handle layers (blas / solver / sparse / fft). Their wrappers are
# RAII over a vendor handle whose create/destroy needs a live GPU context, so
# these are runtime only -- the destroy-exactly-once contract is the whole test.
add_subdirectory(blas)
add_subdirectory(solver)
add_subdirectory(sparse)
add_subdirectory(fft)

# The texture / surface object wrappers. RAII over a vendor handle whose
# create/destroy needs a live GPU context (and a backing CUDA array), so these
# are runtime only -- the destroy-exactly-once contract is the whole test.
add_subdirectory(texture)
add_subdirectory(surface)
