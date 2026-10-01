/**
 * @file exec.cppm
 * @brief Type-safe GPU RNG host generation wrappers
 *
 * The three host generation families each split into a float entry point and a
 * *Double sibling. They collapse into three functions templated on the real
 * precision T (float or double): generate_uniform, generate_normal,
 * generate_lognormal. Each token-pastes its precision-specific name suffix onto
 * wwrrandGenerate via dispatch_macros.h.
 *
 * The wrappers take the raw wwrrandGenerator_t and return the raw
 * wwrrandStatus_t -- generator creation, seeding, stream binding, destruction
 * and error checking are the caller's (RAII and typed error handling live
 * outside this layer).
 *
 * Usage:
 *   import wwr.wrappers.rand;
 *   using namespace wwr;
 *
 *   wwrrandGenerator_t gen;
 *   wwrrandCreateGenerator(&gen, WWRRAND_RNG_PSEUDO_DEFAULT);
 *   wwrrandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
 *   generate_uniform<float>(gen, d_out, n);          // d_out is device memory
 *   wwrrandDestroyGenerator(gen);
 */

module;

#include "dispatch_macros.h"

export module wwr.wrappers.rand:exec;

import wwr.rand;
import :type_traits;
import std;

export namespace wwr {

// ========================================================================
// Uniform: Uniform (float) / UniformDouble (double)
// ========================================================================

/**
 * @brief Generate uniformly distributed values in (0, 1].
 *
 * @tparam T Real precision (float -> Uniform, double -> UniformDouble)
 * @param gen A configured generator
 * @param output Output buffer (device memory) for n values
 * @param n Number of values to generate
 * @return wwrrandStatus_t status code
 */
template<real_fp T>
wwrrandStatus_t generate_uniform(wwrrandGenerator_t gen, T *output, std::size_t n) {
  WWR_RAND_GENERATE_DISPATCH(T, Uniform, UniformDouble, gen, output, n);
}

// ========================================================================
// Normal: Normal (float) / NormalDouble (double)
// ========================================================================

/**
 * @brief Generate normally distributed values with the given mean and stddev.
 *
 * @tparam T Real precision (float -> Normal, double -> NormalDouble)
 * @param gen A configured generator
 * @param output Output buffer (device memory) for n values
 * @param n Number of values to generate
 * @param mean Distribution mean
 * @param stddev Distribution standard deviation
 * @return wwrrandStatus_t status code
 */
template<real_fp T>
wwrrandStatus_t generate_normal(wwrrandGenerator_t gen, T *output, std::size_t n, T mean, T stddev) {
  WWR_RAND_GENERATE_DISPATCH(T, Normal, NormalDouble, gen, output, n, mean, stddev);
}

// ========================================================================
// Log-normal: LogNormal (float) / LogNormalDouble (double)
// ========================================================================

/**
 * @brief Generate log-normally distributed values with the given mean/stddev.
 *
 * @tparam T Real precision (float -> LogNormal, double -> LogNormalDouble)
 * @param gen A configured generator
 * @param output Output buffer (device memory) for n values
 * @param n Number of values to generate
 * @param mean Mean of the underlying normal distribution
 * @param stddev Standard deviation of the underlying normal distribution
 * @return wwrrandStatus_t status code
 */
template<real_fp T>
wwrrandStatus_t generate_lognormal(wwrrandGenerator_t gen, T *output, std::size_t n, T mean,
                                   T stddev) {
  WWR_RAND_GENERATE_DISPATCH(T, LogNormal, LogNormalDouble, gen, output, n, mean, stddev);
}

// ==================== Explicit Template Instantiations ====================
// Hand-written, one per (function, precision). Matching `template`
// instantiations live in instantiations.cpp.

// Function: generate_uniform
extern template wwrrandStatus_t generate_uniform<float>(wwrrandGenerator_t, float *, std::size_t);
extern template wwrrandStatus_t generate_uniform<double>(wwrrandGenerator_t, double *, std::size_t);

// Function: generate_normal
extern template wwrrandStatus_t generate_normal<float>(wwrrandGenerator_t, float *, std::size_t,
                                                       float, float);
extern template wwrrandStatus_t generate_normal<double>(wwrrandGenerator_t, double *, std::size_t,
                                                        double, double);

// Function: generate_lognormal
extern template wwrrandStatus_t generate_lognormal<float>(wwrrandGenerator_t, float *, std::size_t,
                                                          float, float);
extern template wwrrandStatus_t generate_lognormal<double>(wwrrandGenerator_t, double *,
                                                           std::size_t, double, double);

} // namespace wwr
