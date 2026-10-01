/**
 * @file type_traits.cppm
 * @brief Type system for GPU random-number generation
 *
 * Re-exports wwr.wrappers.common for the real_fp concept (float/double, already
 * backend-neutral over src's wwr* types). That is all rand needs: the host
 * generation calls are real-only (float / double), so there is no complex
 * element type to map and thus no FftComplex analogue here -- cuRAND / hipRAND
 * expose no complex host generate function, and the double entry points are a
 * name suffix rather than a distinct element type.
 *
 * Usage:
 *   import wwr.wrappers.rand;
 */

export module wwr.wrappers.rand:type_traits;

export import wwr.wrappers.common;
