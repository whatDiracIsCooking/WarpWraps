/**
 * @file detail/fp16_names.h
 * @brief The fp16 float<->half conversion wrappers, as a macro-driven include
 *        fragment shared by every consumption path
 *
 * NOT a standalone header: it is the two conversion wrappers (wwrFloat2Half /
 * wwrHalf2Float) with NO namespace of its own and NO vendor #include. The
 * includer supplies all of that and pastes this inside its own `namespace wwr`
 * -- so one definition serves all three ways the conversions are consumed:
 * fp16.cppm (the module, host, `export namespace wwr`), fp16.h's device section
 * (a device .cu/.cuh, which cannot import the module) and wwr/fp16.h (the
 * non-module #include path, host). Add or change a conversion here, once, and
 * every path gains it.
 *
 * The three paths differ in exactly one token -- the function qualifier -- which
 * the includer supplies as WWR_FP16_FN: `inline` for the two host paths,
 * `__device__ __forceinline__` for the device section. The bodies are identical
 * because __float2half / __half2float are external-linkage on both backends
 * (docs/architecture.md section 12 lists only fp4/fp6/fp8 as static-inline), so
 * a host TU and a device TU bind the same vendor entry points.
 *
 * Before including, the includer must have, in order:
 *   - the vendor fp16 header in scope and wwrHalf aliased (fp16.h supplies both);
 *   - WWR_FP16_FN defined to the function qualifier this path wants.
 *
 * See src/fp16.cppm, src/fp16.h and src/wwr/fp16.h.
 */

#pragma once

#ifndef WWR_FP16_FN
#error                                                                                             \
    "detail/fp16_names.h is an include fragment, not a standalone header: define WWR_FP16_FN (inline, or __device__ __forceinline__), ensure the vendor fp16 header and wwrHalf are in scope, and #include it inside namespace wwr. See src/fp16.h, src/fp16.cppm and src/wwr/fp16.h."
#endif

/// @brief Convert a float to half precision (round to nearest even)
WWR_FP16_FN wwrHalf wwrFloat2Half(const float value) { return ::__float2half(value); }

/// @brief Widen a half-precision value back to float (exact)
WWR_FP16_FN float wwrHalf2Float(const wwrHalf value) { return ::__half2float(value); }
