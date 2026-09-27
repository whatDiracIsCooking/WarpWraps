/**
 * @file interface.cppm
 * @brief Primary interface for gpumod.wrappers.solver
 *
 * This module provides type-safe C++ wrappers for GPU dense solver operations
 * (cuSOLVER Dense or hipSOLVER Dense, per WWR_GPU_BACKEND). It
 * aggregates all solver partitions:
 * - :type_traits - Type system and concepts (internal)
 * - :linear_solver_legacy - Legacy (int-based, pre-params) linear solver API
 * - :eigen_solver_legacy - Legacy (int-based, pre-params) eigenvalue/SVD API
 * - :linear_solver - Modern (X-prefixed) linear solver API, 8 shared functions
 *
 * Usage:
 *   import gpumod.wrappers.solver;
 *   using namespace wwr;
 */

module;

export module gpumod.wrappers.solver;

import std;

export import :type_traits;
export import :linear_solver_legacy;
export import :eigen_solver_legacy;
export import :linear_solver;
