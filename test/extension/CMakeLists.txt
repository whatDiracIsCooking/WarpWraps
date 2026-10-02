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

# The wwr* Thrust execution-policy shim (wwr::par_on). Like rand it has no
# build-time half: the acceptance is that a device TU's thrust::sort runs on the
# stream par_on bound, so running the sort on a real device IS the test.
add_subdirectory(thrust)

# Backend-neutral batched compression (wwr.comp). Not a src/extension wrapper --
# it lives here because a compress/decompress round-trip needs a live device.
if(WWR_WITH_COMP)
  add_subdirectory(comp)
endif()

# The library-handle layers (blas / solver / sparse / fft). Their wrappers are
# RAII over a vendor handle whose create/destroy needs a live GPU context, so
# these are runtime only -- the destroy-exactly-once contract is the whole test.
add_subdirectory(blas)
# blaslt: one host-only error suite (runs in CI) plus a REQUIRES_GPU lifetime
# suite for the handle + descriptor/layout/preference wrappers.
add_subdirectory(blaslt)
add_subdirectory(solver)
add_subdirectory(sparse)
add_subdirectory(fft)
# tensor: a host-only error-mapping suite (runs in CI) plus a REQUIRES_GPU
# lifetime suite for the handle + descriptor (wwrtensorCreate needs a device).
if(WWR_WITH_TENSOR)
  add_subdirectory(tensor)
endif()

# The NCCL/RCCL communicator wrapper. Its error-policy mapping suite is
# host-only (runs under -LE gpu); its construct/destroy suite is REQUIRES_GPU,
# since wwrcclCommInitAll needs a live device. Gated on WWR_WITH_CCL: the target
# it links (wwr.extension.ccl) does not exist in a build with the wrapper off.
if(WWR_WITH_CCL)
  add_subdirectory(ccl)
endif()

# The texture / surface object wrappers. RAII over a vendor handle whose
# create/destroy needs a live GPU context (and a backing CUDA array), so these
# are runtime only -- the destroy-exactly-once contract is the whole test.
add_subdirectory(texture)
add_subdirectory(surface)
