/**
 * @file interface.cppm
 * @brief Primary interface for gpumod.wrappers.blas
 *
 * This module provides type-safe C++ wrappers for GPU BLAS operations (cuBLAS or hipBLAS, per WWR_GPU_BACKEND).
 * It aggregates all BLAS level operations and extensions:
 * - :type_traits - Type system and concepts (internal)
 * - :level_1 - Vector-vector operations
 * - :level_2 - Matrix-vector operations
 * - :level_3 - Matrix-matrix operations
 * - :extension - BLAS-like extension operations
 *
 * Usage:
 *   import gpumod.wrappers.blas;
 *   using namespace wwr;
 */

module;

export module gpumod.wrappers.blas;

import std;

export import :type_traits;
export import :level_1;
export import :level_2;
export import :level_3;
export import :extension;
