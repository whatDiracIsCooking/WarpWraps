# Shared test-support modules
#
# Helpers imported by more than one test target. Unlike the header/script
# helpers alongside it (link_check.h, dispatch.py), these are compiled C++
# modules. Both are pure module interfaces with no GoogleTest or GPU dependency,
# so they build under WWR_COMPILE_TIME_ONLY too, which the build-time
# memory_buffer test needs.

# AbortPolicy is the abort-on-failure error policy the suites name explicitly
# now that the shipped library ships no default (see abort_policy.cppm). Every
# test that instantiates a wrapper or calls gpu_check pulls its policy here.
wwr_add_cxx_module_library(
  NAME
  wwr.test.shared.abort_policy
  IMPORT_STD
  NO_CUDA_DEVICE_LINKING
  PRIMARY_INTERFACE
  abort_policy.cppm
  LINK_PUBLIC
  wwr.extension.common
)

# RecordingPolicy is the non-throwing record-and-continue counterpart to
# AbortPolicy (see recording_policy.cppm): the suites that exercise recoverable
# construction -- a create() that fails without terminating -- name it so the
# wrapper is left un-owned and BaseHandle::valid() becomes observable.
wwr_add_cxx_module_library(
  NAME
  wwr.test.shared.recording_policy
  IMPORT_STD
  NO_CUDA_DEVICE_LINKING
  PRIMARY_INTERFACE
  recording_policy.cppm
  LINK_PUBLIC
  wwr.extension.common
)

# DeviceHandle is the device_handle the extension suites allocate against, kept
# out of the shipped library (see test/shared/device_handle.cppm).
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
  wwr.test.shared.abort_policy
)
