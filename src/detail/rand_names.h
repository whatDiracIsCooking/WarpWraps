/**
 * @file detail/rand_names.h
 * @brief The backend-neutral cuRAND / hipRAND HOST API surface, as a
 *        macro-driven include fragment shared by every host consumption path
 *
 * NOT a standalone header: it is the list of wwrrand* / WWRRAND_* host names
 * (types, constants, functions and the two scramble-constant forwarders) with NO
 * namespace of its own and NO vendor #include. The includer supplies all of that
 * and pastes this inside its own `namespace wwr` -- so one list binds both ways
 * the host surface is consumed: wwr.rand (the module, `export namespace wwr`) and
 * wwr/rand.h (the non-module #include path). Add a host name here, once, and both
 * paths gain it.
 *
 * HOST only. The device generator state TYPES and the __device__ generator
 * functions are a disjoint surface that lives in rand.h (the types always, the
 * generators in its device-pass-gated section) -- so this fragment never reaches
 * a device pass and may key its one backend conditional on WWR_SELECTED_*. The
 * module re-exports the state types from rand.h with a plain `using`; a
 * non-module TU sees them directly from rand.h, so neither restates them here.
 *
 * Before including, the includer must have, in order:
 *   - the vendor host header in scope (curand.h / hiprand.h), which rand.h pulls
 *     in -- the `::curand*` / `::hiprand*` names the WWR_*_RAW expansions below
 *     put a `::` in front of must resolve to the real declarations;
 *   - WWR_SELECT_RAW(cuda, hip) picking the selected backend's raw name, plus
 *     WWR_TYPE_RAW / WWR_VALUE_RAW / WWR_FUNCTION_RAW on top of it -- keyed on
 *     WWR_GPU_BACKEND_* in the module (backend.h) or WWR_SELECTED_* in the
 *     #include path (wwr/rand.h), the one thing that legitimately differs
 *     between the sites;
 *   - WWR_SELECTED_CUDA / WWR_SELECTED_HIP (selected_backend.h, via rand.h) for
 *     the scramble-constant forwarders' one backend conditional.
 *
 * See src/rand.cppm, src/rand.h, src/wwr/rand.h and docs/architecture.md
 * sections 1 and 6.
 */

#pragma once

#ifndef WWR_FUNCTION_RAW
#error                                                                                             \
    "detail/rand_names.h is an include fragment, not a standalone header: define WWR_TYPE_RAW/VALUE_RAW/FUNCTION_RAW and WWR_SELECT_RAW, ensure the vendor host header (via rand.h) and WWR_SELECTED_* are in scope, and #include it inside namespace wwr. See src/rand.h, src/rand.cppm and src/wwr/rand.h."
#endif

// ========================================================================
// Types
// ========================================================================

WWR_TYPE_RAW(wwrrandGenerator_t, curandGenerator_t, hiprandGenerator_t)
WWR_TYPE_RAW(wwrrandStatus_t, curandStatus_t, hiprandStatus_t)
WWR_TYPE_RAW(wwrrandRngType_t, curandRngType_t, hiprandRngType_t)
WWR_TYPE_RAW(wwrrandOrdering_t, curandOrdering_t, hiprandOrdering_t)
WWR_TYPE_RAW(wwrrandDirectionVectorSet_t, curandDirectionVectorSet_t, hiprandDirectionVectorSet_t)
WWR_TYPE_RAW(wwrrandDirectionVectors32_t, curandDirectionVectors32_t, hiprandDirectionVectors32_t)
WWR_TYPE_RAW(wwrrandDirectionVectors64_t, curandDirectionVectors64_t, hiprandDirectionVectors64_t)
WWR_TYPE_RAW(wwrrandDiscreteDistribution_t, curandDiscreteDistribution_t,
             hiprandDiscreteDistribution_t)

// ========================================================================
// Constants - status
// ========================================================================

WWR_VALUE_RAW(WWRRAND_STATUS_SUCCESS, CURAND_STATUS_SUCCESS, HIPRAND_STATUS_SUCCESS)
WWR_VALUE_RAW(WWRRAND_STATUS_VERSION_MISMATCH, CURAND_STATUS_VERSION_MISMATCH,
              HIPRAND_STATUS_VERSION_MISMATCH)
WWR_VALUE_RAW(WWRRAND_STATUS_NOT_INITIALIZED, CURAND_STATUS_NOT_INITIALIZED,
              HIPRAND_STATUS_NOT_INITIALIZED)
WWR_VALUE_RAW(WWRRAND_STATUS_ALLOCATION_FAILED, CURAND_STATUS_ALLOCATION_FAILED,
              HIPRAND_STATUS_ALLOCATION_FAILED)
WWR_VALUE_RAW(WWRRAND_STATUS_TYPE_ERROR, CURAND_STATUS_TYPE_ERROR, HIPRAND_STATUS_TYPE_ERROR)
WWR_VALUE_RAW(WWRRAND_STATUS_OUT_OF_RANGE, CURAND_STATUS_OUT_OF_RANGE, HIPRAND_STATUS_OUT_OF_RANGE)
WWR_VALUE_RAW(WWRRAND_STATUS_LENGTH_NOT_MULTIPLE, CURAND_STATUS_LENGTH_NOT_MULTIPLE,
              HIPRAND_STATUS_LENGTH_NOT_MULTIPLE)
WWR_VALUE_RAW(WWRRAND_STATUS_DOUBLE_PRECISION_REQUIRED, CURAND_STATUS_DOUBLE_PRECISION_REQUIRED,
              HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED)
WWR_VALUE_RAW(WWRRAND_STATUS_LAUNCH_FAILURE, CURAND_STATUS_LAUNCH_FAILURE,
              HIPRAND_STATUS_LAUNCH_FAILURE)
WWR_VALUE_RAW(WWRRAND_STATUS_PREEXISTING_FAILURE, CURAND_STATUS_PREEXISTING_FAILURE,
              HIPRAND_STATUS_PREEXISTING_FAILURE)
WWR_VALUE_RAW(WWRRAND_STATUS_INITIALIZATION_FAILED, CURAND_STATUS_INITIALIZATION_FAILED,
              HIPRAND_STATUS_INITIALIZATION_FAILED)
WWR_VALUE_RAW(WWRRAND_STATUS_ARCH_MISMATCH, CURAND_STATUS_ARCH_MISMATCH,
              HIPRAND_STATUS_ARCH_MISMATCH)
WWR_VALUE_RAW(WWRRAND_STATUS_INTERNAL_ERROR, CURAND_STATUS_INTERNAL_ERROR,
              HIPRAND_STATUS_INTERNAL_ERROR)

// ========================================================================
// Constants - RNG type (values differ between backends, see rand.cppm header)
// ========================================================================

WWR_VALUE_RAW(WWRRAND_RNG_TEST, CURAND_RNG_TEST, HIPRAND_RNG_TEST)
WWR_VALUE_RAW(WWRRAND_RNG_PSEUDO_DEFAULT, CURAND_RNG_PSEUDO_DEFAULT, HIPRAND_RNG_PSEUDO_DEFAULT)
WWR_VALUE_RAW(WWRRAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_XORWOW, HIPRAND_RNG_PSEUDO_XORWOW)
WWR_VALUE_RAW(WWRRAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_MRG32K3A, HIPRAND_RNG_PSEUDO_MRG32K3A)
WWR_VALUE_RAW(WWRRAND_RNG_PSEUDO_MTGP32, CURAND_RNG_PSEUDO_MTGP32, HIPRAND_RNG_PSEUDO_MTGP32)
WWR_VALUE_RAW(WWRRAND_RNG_PSEUDO_MT19937, CURAND_RNG_PSEUDO_MT19937, HIPRAND_RNG_PSEUDO_MT19937)
WWR_VALUE_RAW(WWRRAND_RNG_PSEUDO_PHILOX4_32_10, CURAND_RNG_PSEUDO_PHILOX4_32_10,
              HIPRAND_RNG_PSEUDO_PHILOX4_32_10)
WWR_VALUE_RAW(WWRRAND_RNG_QUASI_DEFAULT, CURAND_RNG_QUASI_DEFAULT, HIPRAND_RNG_QUASI_DEFAULT)
WWR_VALUE_RAW(WWRRAND_RNG_QUASI_SOBOL32, CURAND_RNG_QUASI_SOBOL32, HIPRAND_RNG_QUASI_SOBOL32)
WWR_VALUE_RAW(WWRRAND_RNG_QUASI_SCRAMBLED_SOBOL32, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
              HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32)
WWR_VALUE_RAW(WWRRAND_RNG_QUASI_SOBOL64, CURAND_RNG_QUASI_SOBOL64, HIPRAND_RNG_QUASI_SOBOL64)
WWR_VALUE_RAW(WWRRAND_RNG_QUASI_SCRAMBLED_SOBOL64, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
              HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64)

// ========================================================================
// Constants - ordering
// ========================================================================

WWR_VALUE_RAW(WWRRAND_ORDERING_PSEUDO_BEST, CURAND_ORDERING_PSEUDO_BEST,
              HIPRAND_ORDERING_PSEUDO_BEST)
WWR_VALUE_RAW(WWRRAND_ORDERING_PSEUDO_DEFAULT, CURAND_ORDERING_PSEUDO_DEFAULT,
              HIPRAND_ORDERING_PSEUDO_DEFAULT)
WWR_VALUE_RAW(WWRRAND_ORDERING_PSEUDO_SEEDED, CURAND_ORDERING_PSEUDO_SEEDED,
              HIPRAND_ORDERING_PSEUDO_SEEDED)
WWR_VALUE_RAW(WWRRAND_ORDERING_PSEUDO_LEGACY, CURAND_ORDERING_PSEUDO_LEGACY,
              HIPRAND_ORDERING_PSEUDO_LEGACY)
WWR_VALUE_RAW(WWRRAND_ORDERING_PSEUDO_DYNAMIC, CURAND_ORDERING_PSEUDO_DYNAMIC,
              HIPRAND_ORDERING_PSEUDO_DYNAMIC)
WWR_VALUE_RAW(WWRRAND_ORDERING_QUASI_DEFAULT, CURAND_ORDERING_QUASI_DEFAULT,
              HIPRAND_ORDERING_QUASI_DEFAULT)

// ========================================================================
// Constants - direction vector set
// ========================================================================

WWR_VALUE_RAW(WWRRAND_DIRECTION_VECTORS_32_JOEKUO6, CURAND_DIRECTION_VECTORS_32_JOEKUO6,
              HIPRAND_DIRECTION_VECTORS_32_JOEKUO6)
WWR_VALUE_RAW(WWRRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6,
              CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6,
              HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
WWR_VALUE_RAW(WWRRAND_DIRECTION_VECTORS_64_JOEKUO6, CURAND_DIRECTION_VECTORS_64_JOEKUO6,
              HIPRAND_DIRECTION_VECTORS_64_JOEKUO6)
WWR_VALUE_RAW(WWRRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6,
              CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6,
              HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6)

// ========================================================================
// Functions
// ========================================================================

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables): each wwrrand*
// below is a deliberate constexpr reference to the selected backend's entry point
// (via WWR_FUNCTION_RAW). A reference to a vendor function has no const form, so
// the check cannot be satisfied without abandoning the alias pattern -- see
// backend.h and detail/runtime_api_names.h.

// Generator management and configuration
WWR_FUNCTION_RAW(wwrrandCreateGenerator, curandCreateGenerator, hiprandCreateGenerator)
WWR_FUNCTION_RAW(wwrrandCreateGeneratorHost, curandCreateGeneratorHost, hiprandCreateGeneratorHost)
WWR_FUNCTION_RAW(wwrrandDestroyGenerator, curandDestroyGenerator, hiprandDestroyGenerator)
WWR_FUNCTION_RAW(wwrrandGetVersion, curandGetVersion, hiprandGetVersion)
WWR_FUNCTION_RAW(wwrrandSetStream, curandSetStream, hiprandSetStream)
WWR_FUNCTION_RAW(wwrrandSetPseudoRandomGeneratorSeed, curandSetPseudoRandomGeneratorSeed,
                 hiprandSetPseudoRandomGeneratorSeed)
WWR_FUNCTION_RAW(wwrrandSetGeneratorOffset, curandSetGeneratorOffset, hiprandSetGeneratorOffset)
WWR_FUNCTION_RAW(wwrrandSetGeneratorOrdering, curandSetGeneratorOrdering,
                 hiprandSetGeneratorOrdering)
WWR_FUNCTION_RAW(wwrrandSetQuasiRandomGeneratorDimensions, curandSetQuasiRandomGeneratorDimensions,
                 hiprandSetQuasiRandomGeneratorDimensions)
WWR_FUNCTION_RAW(wwrrandGenerateSeeds, curandGenerateSeeds, hiprandGenerateSeeds)

// Generation
WWR_FUNCTION_RAW(wwrrandGenerate, curandGenerate, hiprandGenerate)
WWR_FUNCTION_RAW(wwrrandGenerateLongLong, curandGenerateLongLong, hiprandGenerateLongLong)
WWR_FUNCTION_RAW(wwrrandGenerateUniform, curandGenerateUniform, hiprandGenerateUniform)
WWR_FUNCTION_RAW(wwrrandGenerateUniformDouble, curandGenerateUniformDouble,
                 hiprandGenerateUniformDouble)
WWR_FUNCTION_RAW(wwrrandGenerateNormal, curandGenerateNormal, hiprandGenerateNormal)
WWR_FUNCTION_RAW(wwrrandGenerateNormalDouble, curandGenerateNormalDouble,
                 hiprandGenerateNormalDouble)
WWR_FUNCTION_RAW(wwrrandGenerateLogNormal, curandGenerateLogNormal, hiprandGenerateLogNormal)
WWR_FUNCTION_RAW(wwrrandGenerateLogNormalDouble, curandGenerateLogNormalDouble,
                 hiprandGenerateLogNormalDouble)
WWR_FUNCTION_RAW(wwrrandGeneratePoisson, curandGeneratePoisson, hiprandGeneratePoisson)

// Discrete distributions
WWR_FUNCTION_RAW(wwrrandCreatePoissonDistribution, curandCreatePoissonDistribution,
                 hiprandCreatePoissonDistribution)
WWR_FUNCTION_RAW(wwrrandDestroyDistribution, curandDestroyDistribution, hiprandDestroyDistribution)

// Quasirandom direction vectors and scramble constants
WWR_FUNCTION_RAW(wwrrandGetDirectionVectors32, curandGetDirectionVectors32,
                 hiprandGetDirectionVectors32)
WWR_FUNCTION_RAW(wwrrandGetDirectionVectors64, curandGetDirectionVectors64,
                 hiprandGetDirectionVectors64)

// The scramble constants are a read-only table owned by the library.
// hipRAND hands them out as const (const unsigned int**); cuRAND does not
// (unsigned int**). These keep hipRAND's const-correct signature on both
// backends. On CUDA the const_cast only changes how the out-pointer is
// written; the table itself is never written through it. Keyed on
// WWR_SELECTED_CUDA (self-contained via selected_backend.h), so the fragment
// picks the right form in the module and the non-module #include path alike.
#if defined(WWR_SELECTED_CUDA)
inline wwrrandStatus_t wwrrandGetScrambleConstants32(const unsigned int **constants) {
  return ::curandGetScrambleConstants32(const_cast<unsigned int **>(constants));
}
inline wwrrandStatus_t wwrrandGetScrambleConstants64(const unsigned long long **constants) {
  return ::curandGetScrambleConstants64(const_cast<unsigned long long **>(constants));
}
#else
WWR_FUNCTION_RAW(wwrrandGetScrambleConstants32, curandGetScrambleConstants32,
                 hiprandGetScrambleConstants32)
WWR_FUNCTION_RAW(wwrrandGetScrambleConstants64, curandGetScrambleConstants64,
                 hiprandGetScrambleConstants64)
#endif
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
