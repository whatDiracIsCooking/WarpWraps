# C++ Module Library
#
# The extension layer's backend-neutral error-handling primitives, split out of
# wwr.extension.common so the error+check machinery is one self-contained
# unit. Imports only std -- no gpu* names, no runtime API.
# wwr.extension.common re-exports it (and adds the wwrError_t specializations
# in :gpu_error), so consumers reach these names through
# `import wwr.extension.common;`.
#
# The :default_error_policy partition is a customization point. By default it is
# the built-in "print and abort" body in default_error_policy.cppm. A build may
# replace the WHOLE partition file: point WWR_DEFAULT_ERROR_POLICY_MODULE at a
# .cppm that declares this module's :default_error_policy partition and exports
# its own DefaultErrorPolicy<T>. Being a real module unit, the replacement may
# `import` std, the sibling :error_code partition, or a module of your own --
# name any such module target in WWR_DEFAULT_ERROR_POLICY_LINK so it is linked
# and its BMI is reachable when this module is built. See
# default_error_policy.cppm and example/custom_default_error_policy.
set(WWR_DEFAULT_ERROR_POLICY_MODULE
    ""
    CACHE FILEPATH
    "Replacement .cppm for :default_error_policy (empty = built-in)"
)
set(WWR_DEFAULT_ERROR_POLICY_LINK
    ""
    CACHE STRING
    "Module target(s) the replacement :default_error_policy partition imports"
)

# The built-in partition is a normal, same-directory source; an override lives
# elsewhere in the tree, so it is added after the macro call with its own
# FILE_SET base directory (a CXX_MODULES file set otherwise requires every file
# under the calling directory). Leaving it out of PARTITIONS here also keeps the
# built-in default_error_policy.cppm from being compiled alongside the
# replacement -- two files cannot both declare the same partition.
set(_wwr_default_error_policy default_error_policy.cppm)
if(WWR_DEFAULT_ERROR_POLICY_MODULE)
  set(_wwr_default_error_policy "")
endif()

wwr_add_cxx_module_library(
  NAME
  wwr.extension.common.error_handling
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  error_code.cppm
  error_policy.cppm
  ${_wwr_default_error_policy}
  gpu_check.cppm
  LINK_PUBLIC
  ${WWR_DEFAULT_ERROR_POLICY_LINK}
  IMPORT_STD
)

if(WWR_DEFAULT_ERROR_POLICY_MODULE)
  cmake_path(GET WWR_DEFAULT_ERROR_POLICY_MODULE PARENT_PATH _wwr_depol_dir)
  target_sources(
    wwr.extension.common.error_handling PUBLIC
    FILE_SET CXX_MODULES
    BASE_DIRS "${_wwr_depol_dir}"
    FILES "${WWR_DEFAULT_ERROR_POLICY_MODULE}"
  )
endif()
