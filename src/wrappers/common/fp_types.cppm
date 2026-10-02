/**
 * @file fp_types.cppm
 * @brief Backend-neutral floating-point concepts and type maps: the module face
 *        of fp_types.h
 *
 * The host-module counterpart to fp_types.h: it re-exports the concepts (real_fp
 * ... usual_and_half_fp), the type maps (RealToComplexType, ComplexToRealType,
 * HalfToFloatType) and the complex type aliases, so a host TU reaches them with
 * `import wwr.wrappers.common`. The definitions all live in fp_types.h -- the
 * same header a device .cu includes directly -- and this partition only brings
 * them into the module and exports them.
 *
 * Unlike complex.cppm / fp16.cppm, there is nothing here to duplicate: fp_types.h
 * holds only concepts and alias templates, pure compile-time constructs with no
 * __device__ bodies, so the single definition serves both this module and a
 * device include. The GMF #includes the header and the names are re-exported
 * below. See docs/architecture.md, section 3.
 *
 * Usage:
 *   import wwr.wrappers.common;
 *   using namespace wwr;
 */

module;

// The concepts, type maps, and (via complex.h) the complex type aliases. They
// land in the global module here and are re-exported below.
#include "fp_types.h"

export module wwr.wrappers.common:fp_types;

export namespace wwr {

// ========================================================================
// GPU Complex Types -- re-exported from fp_types.h (via complex.h), so
// importers of wwr.wrappers.common see them for convenience
// ========================================================================

using wwr::wwrComplex;
using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

// ========================================================================
// Floating-Point Type Concepts
// ========================================================================

using wwr::complex_fp;
using wwr::half_fp;
using wwr::real_fp;
using wwr::usual_and_half_fp;
using wwr::usual_fp;

// ========================================================================
// Real <-> Complex and Half-Precision Type Mappings
// ========================================================================

using wwr::ComplexToRealType;
using wwr::HalfToFloatType;
using wwr::RealToComplexType;

} // namespace wwr
