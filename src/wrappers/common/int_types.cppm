/**
 * @file int_types.cppm
 * @brief Integer type concepts for GPU library operations
 *
 * This module provides C++20 concepts for constraining template parameters
 * to integer types commonly used in GPU BLAS/solver libraries.
 *
 * Usage:
 *   import gpumod.wrappers.common;
 *   using namespace wwr;
 */

export module gpumod.wrappers.common:int_types;

import std;

export namespace wwr {

// ========================================================================
// Integer Types
// ========================================================================

// ========================================================================
// Re-export standard integer types
// ========================================================================
using std::int64_t;
using std::size_t;

/**
 * @brief Concept constraining type T to the exact index widths the wrappers dispatch on
 *
 * Exactly int or int64_t, and nothing else -- these are the only two types the
 * *_DISPATCH_64 macros discriminate (`if constexpr (is_same_v<IntT, int>) ...
 * else if (is_same_v<IntT, int64_t>) ...`, with no else branch). A wider domain
 * would let a type through the concept that then matches neither branch, leaving
 * the wrapper body empty -- a non-void function with no return.
 *
 * This is deliberately not `std::convertible_to<T, int>`: that admitted double,
 * float, bool and char (all convertible to int), and even the integral types it
 * was meant to allow -- size_t, long long -- are dropped by the dispatch, so
 * accepting them here only hid the mismatch. Callers holding such a type must
 * cast to int or int64_t explicitly.
 */
template<typename T>
concept int_type = std::is_same_v<T, int> || std::is_same_v<T, int64_t>;

} // namespace wwr
