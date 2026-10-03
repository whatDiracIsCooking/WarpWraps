# GPU BLAS wrapper tests (backend-neutral: built and run for either
# WWR_GPU_BACKEND)
#
# Build-time only. The wrappers in src/wrappers/blas are token-paste
# forwarders -- the body is `return wwrblas<X><name>(__VA_ARGS__)` and nothing
# else -- so the whole of what they can get wrong is which name is pasted, and
# the dispatch check proves that for every instantiation, without a device.

# GPU_SOURCE is the file carrying the WWR_FUNCTION_RAW(wwrblas...) alias lines
# the check maps vendor symbols back through -- the shared fragment, not
# blas.cppm, since the surface moved there (blas.cppm now just #includes it, and
# dispatch.py scans one file's text, not its includes).
wwr_add_dispatch_check(
  NAME
  wwr.test.wrappers.blas_dispatch
  TARGET
  wwr.wrappers.blas
  TABLE
  blas_dispatch.toml
  GPU_SOURCE
  src/detail/blas_names.h
)
