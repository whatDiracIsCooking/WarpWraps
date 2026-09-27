/**
 * @file interface.cppm
 * @brief Primary interface for gpumod.extension.common
 *
 * This module provides common, backend-neutral utilities for GPU error handling and extension functionality.
 * It aggregates:
 * - gpumod.extension.common.error_handling - the backend-neutral error+check
 *     primitives (error_code, error_policy, default_error_policy, gpu_check),
 *     re-exported so they stay reachable through this umbrella module
 * - :gpu_error - gpuError_t specializations of the error_code utilities
 * - :gpu_handle - RAII wrapper base class for GPU handles
 * - :device_bound_handle - CRTP layer recording a handle's owning device
 * - :gpu_handle_view - Non-owning, copyable view over a GPU handle
 * - :device_bound_handle_view - Non-owning view carrying its handle's device
 * - :device_scope - RAII guard that makes a device current and restores the previous one
 * - :noncopyable - Mixin deleting copy operations while allowing moves
 *
 * Usage:
 *   import gpumod.extension.common;
 *   using namespace gpumod::extension;
 */

export module gpumod.extension.common;

import std;

export import gpumod.extension.common.error_handling;
export import :gpu_error;
export import :gpu_handle;
export import :device_bound_handle;
export import :gpu_handle_view;
export import :device_bound_handle_view;
export import :device_scope;
export import :noncopyable;
