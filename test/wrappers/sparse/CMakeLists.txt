# GPU sparse wrapper tests (backend-neutral: built and run for either
# WWR_GPU_BACKEND)
#
# Build-time only. The wrappers in src/wrappers/sparse are token-paste
# forwarders -- the body is `return wwrsparse<X><name>(__VA_ARGS__)` and nothing
# else -- so the whole of what they can get wrong is which name is pasted, and
# the dispatch check proves that for every instantiation, without a device.

# GPU_SOURCE is the file carrying the WWR_FUNCTION_RAW(wwrsparse...) alias lines
# the check maps vendor symbols back through -- the shared fragment, not
# sparse.cppm, since the surface moved there (sparse.cppm now just #includes it,
# and dispatch.py scans one file's text, not its includes).
wwr_add_dispatch_check(
  NAME
  wwr.test.wrappers.sparse_dispatch
  TARGET
  wwr.wrappers.sparse
  TABLE
  sparse_dispatch.toml
  GPU_SOURCE
  src/detail/sparse_names.h
)
