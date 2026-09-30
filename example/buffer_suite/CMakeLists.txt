# The buffer-suite example: the MAP entry point of the :suite partition of
# wwr.extension.memory_buffer (buffer_suite / device_buffer_suite), plus the
# views the suite pairs to their buffers. Unlike warp_reduce there is no kernel
# and no .cu -- this exercises only the host-callable RAII + copy surface, so it
# is one ordinary host CXX module translation unit and needs neither a device
# library nor the src/ include root.

add_executable(wwr.example.buffer_suite main.cpp)

# wwr::extension::common in its own right (success_code/error_name/error_string
# and the error-policy concepts main.cpp names), wwr::extension::runtime for the
# StreamWrapper the handle owns, wwr::extension::memory_buffer for the buffers,
# the suite and copy(). wwr::runtime_api for wwrGetDevice/wwrSuccess.
target_link_libraries(
  wwr.example.buffer_suite
  PRIVATE wwr::runtime_api
          wwr::extension::common
          wwr::extension::runtime
          wwr::extension::memory_buffer
)

set_target_properties(wwr.example.buffer_suite PROPERTIES CXX_MODULE_STD 1)
