# Shared test-support modules
#
# Helpers imported by more than one test target. Unlike the header/script
# helpers alongside it (link_check.h, dispatch.py), DeviceHandle is a compiled
# C++ module: the device_handle the extension suites allocate against, kept out
# of the shipped library (see test/shared/device_handle.cppm). It is a pure
# module interface with no GoogleTest or GPU dependency, so it builds under
# WWR_COMPILE_TIME_ONLY too, which the build-time memory_buffer test needs.
wwr_add_cxx_module_library(
  NAME
  wwr.test.shared.device_handle
  IMPORT_STD
  NO_CUDA_DEVICE_LINKING
  PRIMARY_INTERFACE
  device_handle.cppm
  LINK_PUBLIC
  wwr.runtime_api
  wwr.extension.common
  wwr.extension.handle
  wwr.extension.runtime
)
