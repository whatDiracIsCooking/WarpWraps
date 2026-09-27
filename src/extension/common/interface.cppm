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
 * - :device_scope - RAII guard that makes a device current and restores the previous one
 * - :noncopyable - Mixin deleting copy operations while allowing moves
 *
 * The RAII GPU-handle layer that once lived here (BaseHandle, DeviceBoundHandle
 * and their views) is now its own module, gpumod.extension.handle, which
 * builds on this one.
 *
 * Usage:
 *   import gpumod.extension.common;
 *   using namespace gpumod::extension;
 */

export module gpumod.extension.common;

import std;

export import gpumod.extension.common.error_handling;
export import :gpu_error;
export import :device_scope;
export import :noncopyable;
