/**
 * @file interface.cppm
 * @brief Primary interface for wwr.wrappers.common
 *
 * This module provides the common, backend-neutral type vocabulary the four
 * extensions (blas, solver, sparse, fft) are written against. It aggregates:
 * - :fp_types - Floating-point type concepts and real/complex/half type maps
 * - :int_types - The index-width concept the wrappers constrain IntT with
 *
 * Usage:
 *   import wwr.wrappers.common;
 *   using namespace wwr;
 */

export module wwr.wrappers.common;

import std;

export import :fp_types;
export import :int_types;
