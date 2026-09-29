// type_traits.cppm - Compile-time tests for wwr.wrappers.solver's type system

export module wwr.test.wrappers.solver_type_traits;

import std;
import wwr.solver;
import wwr.complex;
import wwr.wrappers.solver;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// get_wwrsolver_type<T>() and the concepts around it
//
// The legacy typed API picks its entry point by token-pasting a type letter,
// and solver_dispatch.toml checks that in the compiled objects. The modern
// (X-prefixed) API picks nothing: it passes get_wwrsolver_type<T>() as a
// runtime wwrsolverDataType_t, so a wrong mapping there is a value, not a call,
// and no disassembly can see it. It is a constant expression though, which
// makes it exactly a static_assert's business -- this is the modern API's half
// of the dispatch check.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

namespace wwr::test {

static_assert(get_wwrsolver_type<float>() == WWRSOLVER_R_32F);
static_assert(get_wwrsolver_type<double>() == WWRSOLVER_R_64F);
static_assert(get_wwrsolver_type<wwrFloatComplex>() == WWRSOLVER_C_32F);
static_assert(get_wwrsolver_type<wwrDoubleComplex>() == WWRSOLVER_C_64F);

// All four must be distinct, which is what makes the four asserts above a real
// constraint: a backend that collapsed two of these enumerators onto one value
// would let two of the mappings be swapped and still pass.
static_assert(WWRSOLVER_R_32F != WWRSOLVER_R_64F);
static_assert(WWRSOLVER_R_32F != WWRSOLVER_C_32F);
static_assert(WWRSOLVER_R_32F != WWRSOLVER_C_64F);
static_assert(WWRSOLVER_R_64F != WWRSOLVER_C_32F);
static_assert(WWRSOLVER_R_64F != WWRSOLVER_C_64F);
static_assert(WWRSOLVER_C_32F != WWRSOLVER_C_64F);

// get_wwrsolver_type is constrained to usual_fp, so the four types above are
// exactly its domain. wwrComplex is the same type as wwrFloatComplex (both name
// the vendor's single-precision complex), which is why it maps to C_32F too.
static_assert(usual_fp<float>);
static_assert(usual_fp<double>);
static_assert(usual_fp<wwrFloatComplex>);
static_assert(usual_fp<wwrDoubleComplex>);
static_assert(!usual_fp<int>);
static_assert(!usual_fp<long double>);
static_assert(std::is_same_v<wwrComplex, wwrFloatComplex>);
static_assert(get_wwrsolver_type<wwrComplex>() == WWRSOLVER_C_32F);

static_assert(real_fp<float> && real_fp<double>);
static_assert(!real_fp<wwrFloatComplex> && !real_fp<wwrDoubleComplex>);
static_assert(complex_fp<wwrFloatComplex> && complex_fp<wwrDoubleComplex>);
static_assert(!complex_fp<float> && !complex_fp<double>);

// ComplexToRealType is what the wrappers spell for the real-valued outputs of a
// complex solve -- the singular values of gesvd, the eigenvalues of heevd. Its
// instantiations.cpp signatures are written in terms of it, so a wrong mapping
// here is a wrong explicit instantiation that still compiles.
static_assert(std::is_same_v<ComplexToRealType<float>, float>);
static_assert(std::is_same_v<ComplexToRealType<double>, double>);
static_assert(std::is_same_v<ComplexToRealType<wwrFloatComplex>, float>);
static_assert(std::is_same_v<ComplexToRealType<wwrDoubleComplex>, double>);

// The precision a type maps to must match the precision of its real part: a
// C_32F whose ComplexToRealType was double would size every workspace wrong.
static_assert(get_wwrsolver_type<ComplexToRealType<wwrFloatComplex>>() == WWRSOLVER_R_32F);
static_assert(get_wwrsolver_type<ComplexToRealType<wwrDoubleComplex>>() == WWRSOLVER_R_64F);

} // namespace wwr::test
