/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.common
 *
 * This module provides common, backend-neutral utilities for GPU error handling and extension functionality.
 * It aggregates:
 * - wwr.extension.error_handling - the backend-neutral error+check
 *     primitives (error_code, error_policy, gpu_check),
 *     re-exported so they stay reachable through this umbrella module
 * - :error - wwrError_t specializations of the error_code utilities
 * - :scoped_device_index - RAII guard that makes a device current and restores the previous one
 * - :noncopyable - Mixin deleting copy operations while allowing moves
 * - :policy_slot - Storage that collapses a same-type empty policy slot to nothing
 *
 * The RAII GPU-handle layer that once lived here (BaseHandle, DeviceBoundHandle
 * and their views) is now its own module, wwr.extension.handle, which
 * builds on this one.
 *
 * Usage:
 *   import wwr.extension.common;
 *   using namespace wwr::extension;
 */

export module wwr.extension.common;

import std;

export import wwr.extension.error_handling;
export import :error;
export import :scoped_device_index;
export import :noncopyable;
export import :policy_slot;
