# The custom-default-error-policy example.
#
# Demonstrates the partition-swap customization point. Configure the tree with
#   cmake --preset default \
#     -DWWR_DEFAULT_ERROR_POLICY_MODULE=\
#         ${CMAKE_CURRENT_SOURCE_DIR}/custom_default_error_policy.cppm \
#     -DWWR_DEFAULT_ERROR_POLICY_LINK=wwr.example.error_logger
# and wwr::extension::DefaultErrorPolicy is replaced build-wide by
# custom_default_error_policy.cppm's body (see that file and
# src/extension/common/error_handling). The swap happens in the error-handling
# module's own build; this executable just imports the extension layer and
# inherits whichever default policy the tree was compiled with, so it builds
# unchanged either way.
#
# No ctest entry -- and this is why: with the default policy the program aborts
# by design (it triggers an error to show the policy running), so registering it
# would turn a default build's test run red. It needs no device; run it by hand,
# both ways, to see the difference (see example/README.md).

# The user module the replacement partition imports. Defined unconditionally so
# it compiles on both backends (cross-backend-check) and its target exists for
# WWR_DEFAULT_ERROR_POLICY_LINK to name; it is only actually linked into the
# error-handling module when the tree is configured with the swap above.
wwr_add_cxx_module_library(
  NAME
  wwr.example.error_logger
  PRIMARY_INTERFACE
  example_logger.cppm
  IMPORT_STD
)

add_executable(wwr.example.custom_default_error_policy main.cpp)

# wwr.runtime_api for wwrErrorInvalidValue/wwrSuccess; wwr.extension.common for
# gpu_check and DefaultErrorPolicy. Both are backend-neutral, so this compiles
# on either backend -- which is what devtools/cross-backend-check.sh checks.
target_link_libraries(
  wwr.example.custom_default_error_policy
  PRIVATE wwr.runtime_api
          wwr.extension.common
)

set_target_properties(
  wwr.example.custom_default_error_policy PROPERTIES CXX_MODULE_STD 1
)
