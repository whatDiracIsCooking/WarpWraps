# The device-handle example: kit::DeviceHandle, the ready-made pool-tier handle
# from wwr.extension.runtime -- owns a stream and a memory pool on one device,
# so a buffer backed by it allocates from that pool (wwrMallocFromPoolAsync)
# where warp_reduce/buffer_suite back theirs on a bare StreamWrapper (the stream
# tier). Like buffer_suite there is no kernel and no .cu: it exercises only the
# host-callable RAII + copy surface, so it is one ordinary host CXX module
# translation unit and needs neither a device library nor the src/ include root.

add_executable(wwr.example.device_handle main.cpp)

# wwr::extension::runtime carries kit::DeviceHandle (and the StreamWrapper /
# MemPoolWrapper it composes). wwr::extension::common for kit::AbortPolicy,
# wwr::extension::memory_buffer for the buffers, the suite and copy().
# wwr::runtime_api for wwrGetDevice/wwrSuccess and the wwrDeviceProp that
# props() returns.
target_link_libraries(
  wwr.example.device_handle
  PRIVATE wwr::runtime_api
          wwr::extension::common
          wwr::extension::runtime
          wwr::extension::memory_buffer
)

set_target_properties(wwr.example.device_handle PROPERTIES CXX_MODULE_STD 1)
