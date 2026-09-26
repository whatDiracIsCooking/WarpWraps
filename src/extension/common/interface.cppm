/**
 * @file interface.cppm
 * @brief Primary interface for gpumod.extension.common
 *
 * This module provides common, backend-neutral utilities for GPU error handling and extension functionality.
 * It aggregates:
 * - :error_code - Success codes and error string utilities
 * - :gpu_error - gpuError_t specializations of the error_code utilities
 * - :gpu_check - GPU error checking utilities
 * - :error_policy - Base error policy class template
 * - :default_error_policy - Default error policy implementation
 * - :gpu_handle - RAII wrapper base class for GPU handles
 * - :device_bound_handle - CRTP layer recording a handle's owning device
 * - :gpu_handle_view - Non-owning, copyable view over a GPU handle
 * - :device_bound_handle_view - Non-owning view carrying its handle's device
 * - :noncopyable - Mixin deleting copy operations while allowing moves
 *
 * Usage:
 *   import gpumod.extension.common;
 *   using namespace gpumod::extension;
 */

export module gpumod.extension.common;

import std;

export import :error_code;
export import :gpu_error;
export import :gpu_check;
export import :error_policy;
export import :default_error_policy;
export import :gpu_handle;
export import :device_bound_handle;
export import :gpu_handle_view;
export import :device_bound_handle_view;
export import :noncopyable;
