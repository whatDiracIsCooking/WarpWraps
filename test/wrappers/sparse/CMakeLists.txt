# GPU sparse wrapper tests (backend-neutral: built and run for either
# WWR_GPU_BACKEND)
#
# Build-time only. The wrappers in src/wrappers/sparse are token-paste
# forwarders -- the body is `return gpusparse<X><name>(__VA_ARGS__)` and nothing
# else -- so the whole of what they can get wrong is which name is pasted, and
# the dispatch check proves that for every instantiation, without a device.

wwr_add_dispatch_check(
  NAME
  gpumod.test.wrappers.sparse_dispatch
  TARGET
  gpumod.wrappers.sparse
  TABLE
  sparse_dispatch.toml
  GPU_SOURCE
  src/sparse.cppm
)
