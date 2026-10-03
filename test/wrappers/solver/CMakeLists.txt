# GPU dense solver wrapper tests (backend-neutral: built and run for either
# WWR_GPU_BACKEND)
#
# Build-time only, and the two tests split the module's type dispatch between
# them. The legacy typed API picks its entry point by token-pasting a type
# letter, which no static_assert can see, so solver_dispatch.toml checks it in
# the compiled objects. The modern (X-prefixed) API passes
# get_wwrsolver_type<T>() as a runtime value instead, which no disassembly can
# see, so type_traits.cppm checks that with static_asserts. Between them they
# cover every instantiation: the wrappers are token-paste forwarders with no
# body of their own, so a pasted name and a mapped data type are the whole of
# what they can get wrong.

wwr_add_cxx_module_library(
  NAME
  wwr.test.wrappers.solver_type_traits
  IMPORT_STD
  NO_CUDA_DEVICE_LINKING
  PRIMARY_INTERFACE
  type_traits.cppm
  LINK_PUBLIC
  wwr::solver
  wwr::complex
  wwr::wrappers::solver
)

add_dependencies(
  compile_time_tests wwr.test.wrappers.solver_type_traits
)

# GPU_SOURCE is the file carrying the WWR_FUNCTION_RAW(wwrsolver...) alias lines
# the check maps vendor symbols back through -- the shared fragment, not
# solver.cppm, since the surface moved there (solver.cppm now just #includes it,
# and dispatch.py scans one file's text, not its includes).
wwr_add_dispatch_check(
  NAME
  wwr.test.wrappers.solver_dispatch
  TARGET
  wwr.wrappers.solver
  TABLE
  solver_dispatch.toml
  GPU_SOURCE
  src/detail/solver_names.h
)
