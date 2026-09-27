# GPU tools-extension wrapper tests (backend-neutral: built for either
# WWR_GPU_BACKEND)
#
# Build-time only, and deliberately no runtime half -- like the fft and solver
# wrapper tests. The tx wrapper is untyped: it has no dispatch table to check in
# the objects and no runtime data-type value to check, so what it can get wrong
# is its API shape (free-function signatures, ScopedRange's ownership contract),
# which api.cppm pins with static_asserts. Its annotations have no observable
# effect with no profiler tool attached, so a runtime suite could only assert
# "did not crash" -- and would add an entry to the `ctest -LE gpu` count for it.
wwr_add_cxx_module_library(
  NAME
  wwr.test.wrappers.tx_api
  IMPORT_STD
  NO_CUDA_DEVICE_LINKING
  PRIMARY_INTERFACE
  api.cppm
  LINK_PUBLIC
  wwr.tx
  wwr.wrappers.tx
)

add_dependencies(compile_time_tests wwr.test.wrappers.tx_api)
