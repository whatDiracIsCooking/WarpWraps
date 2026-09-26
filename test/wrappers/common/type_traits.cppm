// type_traits.cppm - Compile-time tests for gpumod.wrappers.common
//
// common is the type vocabulary the four extensions (blas, solver, sparse, fft)
// are written against: the fp/int concepts and the real/complex/half type maps.
// These were, for a long time, exercised only incidentally -- wherever a
// downstream extension happened to use one. These checks import
// gpumod.wrappers.common directly so they stand on their own: a wrong mapping
// compiles clean and ships, and a static_assert is the only thing that sees it.
//
// This file asserts the type maps only. gpumod.wrappers.common carries no
// error-policy or RAII-handle layer; that lives in gpumod.extension.common.

export module gpumod.test.wrappers.common_type_traits;

import std;
import gpumod.complex;
import gpumod.fp16;
import gpumod.bf16;
import gpumod.wrappers.common;

namespace gpumod::test {

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// int_type -- the index-width constraint the BLAS wrappers put on IntT
//
// It must be exactly {int, int64_t}, because that is exactly what the
// *_DISPATCH_64 macros discriminate: `if constexpr (is_same_v<IntT, int>) ...
// else if (is_same_v<IntT, int64_t>) ...`, with no else. Any type that
// satisfies the concept but matches neither branch expands the wrapper body to
// nothing -- a non-void function returning gpublasStatus_t with no return. The
// concept is the guard that keeps that from being reachable, so these asserts
// pin its domain to the two types the dispatch handles and nothing wider.
//
// The earlier definition was `... || convertible_to<T, int> || convertible_to<T,
// int64_t>`, which made int_type<double> true (double converts to int) along
// with float, bool and char -- the wrongest possible domain for a concept whose
// whole job is to pin an index width. #58.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// The two types the dispatch handles.
static_assert(int_type<int>);
static_assert(int_type<std::int64_t>);

// Floating-point types: all convertible to int, so all accepted by the old
// definition. This is the regression #58 was about.
static_assert(!int_type<double>);
static_assert(!int_type<float>);
static_assert(!int_type<long double>);

// Other things that convert to an integer -- bool, char, an unscoped enum -- and
// so also slipped through before.
static_assert(!int_type<bool>);
static_assert(!int_type<char>);
enum unscoped_enum { e0 };
static_assert(!int_type<unscoped_enum>);

// Integral types the concept never handled correctly: the dispatch drops them
// (they match neither is_same_v branch), so admitting them only hid the
// mismatch. size_t is unsigned long on LP64, distinct from int64_t's signed
// long; unsigned/short/long long are all their own types.
static_assert(!int_type<std::size_t>);
static_assert(!int_type<unsigned int>);
static_assert(!int_type<short>);
static_assert(!int_type<long long>);

// Unrelated types are rejected, as they always were.
struct not_an_int {};
static_assert(!int_type<not_an_int>);
static_assert(!int_type<int *>);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// fp_types: the floating-point concepts
//
// solver/type_traits.cppm asserts real_fp/complex_fp/usual_fp and
// ComplexToRealType, but only as a side effect of testing the solver, and it
// never touches half_fp, usual_and_half_fp, RealToComplexType or
// HalfToFloatType. These are the full domain and anti-domain of each concept.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

static_assert(real_fp<float>);
static_assert(real_fp<double>);
static_assert(!real_fp<gpuFloatComplex>);
static_assert(!real_fp<gpuDoubleComplex>);
static_assert(!real_fp<gpuHalf>);
static_assert(!real_fp<int>);

static_assert(complex_fp<gpuFloatComplex>);
static_assert(complex_fp<gpuDoubleComplex>);
static_assert(!complex_fp<float>);
static_assert(!complex_fp<double>);

static_assert(usual_fp<float>);
static_assert(usual_fp<double>);
static_assert(usual_fp<gpuFloatComplex>);
static_assert(usual_fp<gpuDoubleComplex>);
static_assert(!usual_fp<gpuHalf>);
static_assert(!usual_fp<gpuBfloat16>);
static_assert(!usual_fp<int>);
static_assert(!usual_fp<long double>);

static_assert(half_fp<gpuHalf>);
static_assert(half_fp<gpuBfloat16>);
static_assert(!half_fp<float>);
static_assert(!half_fp<double>);
static_assert(!half_fp<gpuFloatComplex>);

static_assert(usual_and_half_fp<float>);
static_assert(usual_and_half_fp<double>);
static_assert(usual_and_half_fp<gpuFloatComplex>);
static_assert(usual_and_half_fp<gpuDoubleComplex>);
static_assert(usual_and_half_fp<gpuHalf>);
static_assert(usual_and_half_fp<gpuBfloat16>);
static_assert(!usual_and_half_fp<int>);
static_assert(!usual_and_half_fp<long double>);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// fp_types: the real <-> complex <-> half type maps
//
// RealToComplexType is the inverse of ComplexToRealType and is what a wrapper
// spells for the complex output of a real-input routine; a swapped precision
// here is a wrong explicit instantiation that still compiles. The two maps must
// round-trip.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

static_assert(std::is_same_v<RealToComplexType<float>, gpuFloatComplex>);
static_assert(std::is_same_v<RealToComplexType<double>, gpuDoubleComplex>);

static_assert(std::is_same_v<ComplexToRealType<float>, float>);
static_assert(std::is_same_v<ComplexToRealType<double>, double>);
static_assert(std::is_same_v<ComplexToRealType<gpuFloatComplex>, float>);
static_assert(std::is_same_v<ComplexToRealType<gpuDoubleComplex>, double>);

// real -> complex -> real is the identity on the reals
static_assert(std::is_same_v<ComplexToRealType<RealToComplexType<float>>, float>);
static_assert(std::is_same_v<ComplexToRealType<RealToComplexType<double>>, double>);

static_assert(std::is_same_v<HalfToFloatType<gpuHalf>, float>);
static_assert(std::is_same_v<HalfToFloatType<gpuBfloat16>, float>);

} // namespace gpumod::test
