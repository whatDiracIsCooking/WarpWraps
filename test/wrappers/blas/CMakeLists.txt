# GPU BLAS wrapper tests (backend-neutral: built and run for either
# GPUMOD_GPU_BACKEND)
#
# Build-time only. The wrappers in src/wrappers/blas are token-paste
# forwarders -- the body is `return gpublas<X><name>(__VA_ARGS__)` and nothing
# else -- so the whole of what they can get wrong is which name is pasted, and
# the dispatch check proves that for every instantiation, without a device.

gpumod_add_dispatch_check(
  NAME
  gpumod.test.wrappers.blas_dispatch
  TARGET
  gpumod.wrappers.blas
  TABLE
  blas_dispatch.toml
  GPU_SOURCE
  src/blas.cppm
)
