/**
 * @file interface.cppm
 * @brief Primary interface for wwr.wrappers.rand
 *
 * This module provides type-safe C++ wrappers for GPU random-number generation
 * (cuRAND or hipRAND, per WWR_GPU_BACKEND). It aggregates:
 * - :type_traits - The real_fp concept (re-exported from wwr.wrappers.common)
 * - :exec - Type-safe host generation wrappers (generate_uniform / _normal /
 *   _lognormal)
 *
 * Usage:
 *   import wwr.wrappers.rand;
 *   using namespace wwr;
 */

module;

export module wwr.wrappers.rand;

import std;

export import :type_traits;
export import :exec;
