/**
 * @file interface.cppm
 * @brief Primary interface for gpumod.wrappers.sparse
 *
 * This module provides type-safe C++ wrappers for the legacy typed (S/D/C/Z)
 * GPU sparse operations cuSPARSE and hipSPARSE have in common (per
 * WWR_GPU_BACKEND). It aggregates all sparse partitions:
 * - :level_2 - BSR matrix-vector multiply
 * - :solvers - Tridiagonal/pentadiagonal batch solvers (gtsv2 / gpsvInterleavedBatch)
 * - :extra - CSR matrix addition (csrgeam2)
 * - :conversion - nnz, gebsr2gebsc, csr2gebsr
 *
 * Usage:
 *   import gpumod.wrappers.sparse;
 *   using namespace wwr;
 */

module;

export module gpumod.wrappers.sparse;

import std;

export import :level_2;
export import :solvers;
export import :extra;
export import :conversion;
