/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.handle
 *
 * The RAII GPU-handle layer, split out of wwr.extension.common so the error
 * foundation (error_code/gpu_error/gpu_check/error_policy) stays free of the
 * handle machinery. Built on that foundation -- it imports wwr.extension.common
 * for the error policies and NonCopyable, and wwr.runtime_api for the device
 * queries -- and aggregates:
 * - :handle - RAII wrapper base class for GPU handles
 * - :device_bound_handle - CRTP layer recording a handle's owning device
 * - :stream_bound_handle - CRTP layer binding, and outlive-anchoring, a work stream
 * - :handle_view - Non-owning, copyable view over a GPU handle
 * - :device_bound_handle_view - Non-owning view carrying its handle's device
 * - :device_handle - the device_handle capability ladder DeviceBuffer selects on
 *
 * Usage:
 *   import wwr.extension.handle;
 *   using namespace wwr::extension;
 */

export module wwr.extension.handle;

export import :handle;
export import :device_bound_handle;
export import :stream_bound_handle;
export import :handle_view;
export import :device_bound_handle_view;
export import :device_handle;
