# C++ Module Library
#
# The extension layer's backend-neutral error-handling primitives, split out of
# wwr.extension.common so the error+check machinery is one self-contained
# unit. Imports only std -- no gpu* names, no runtime API.
# wwr.extension.common re-exports it (and adds the gpuError_t specializations
# in :gpu_error), so consumers reach these names through
# `import wwr.extension.common;`.
wwr_add_cxx_module_library(
  NAME
  wwr.extension.common.error_handling
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  error_code.cppm
  error_policy.cppm
  default_error_policy.cppm
  gpu_check.cppm
  IMPORT_STD
)

# Optional custom default error policy. Point WWR_DEFAULT_ERROR_POLICY_IMPL at a
# header defining wwr::extension::DefaultErrorPolicyImpl<T> and this module's
# DefaultErrorPolicy derives from it instead of the built-in body (see
# default_error_policy.cppm). PRIVATE: the macro only has to reach the
# compilation of default_error_policy.cppm, which happens in this target.
set(WWR_DEFAULT_ERROR_POLICY_IMPL
    ""
    CACHE FILEPATH
    "Header providing wwr::extension::DefaultErrorPolicyImpl<T>"
)
if(WWR_DEFAULT_ERROR_POLICY_IMPL)
  cmake_path(GET WWR_DEFAULT_ERROR_POLICY_IMPL PARENT_PATH _wwr_depol_dir)
  cmake_path(GET WWR_DEFAULT_ERROR_POLICY_IMPL FILENAME _wwr_depol_name)
  target_include_directories(
    wwr.extension.common.error_handling PRIVATE "${_wwr_depol_dir}"
  )
  target_compile_definitions(
    wwr.extension.common.error_handling PRIVATE
    "WWR_DEFAULT_ERROR_POLICY_IMPL=\"${_wwr_depol_name}\""
  )
endif()
