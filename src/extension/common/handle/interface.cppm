/**
 * @file interface.cppm
 * @brief Primary interface for gpumod.extension.common.handle
 *
 * The RAII GPU-handle layer, split out of gpumod.extension.common so the error
 * foundation (error_code/gpu_error/gpu_check/error_policy) stays free of the
 * handle machinery. Built on that foundation -- it imports gpumod.extension.common
 * for the error policies and NonCopyable, and gpumod.runtime_api for the device
 * queries -- and aggregates:
 * - :gpu_handle - RAII wrapper base class for GPU handles
 * - :device_bound_handle - CRTP layer recording a handle's owning device
 * - :gpu_handle_view - Non-owning, copyable view over a GPU handle
 * - :device_bound_handle_view - Non-owning view carrying its handle's device
 *
 * Usage:
 *   import gpumod.extension.common.handle;
 *   using namespace gpumod::extension;
 */

export module gpumod.extension.common.handle;

export import :gpu_handle;
export import :device_bound_handle;
export import :gpu_handle_view;
export import :device_bound_handle_view;
