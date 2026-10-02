/**
 * @file detail/bf16_names.h
 * @brief The bf16 float<->bfloat16 conversion wrappers, as a macro-driven
 *        include fragment shared by every consumption path
 *
 * NOT a standalone header: it is the two conversion wrappers (wwrFloat2Bfloat16 /
 * wwrBfloat162Float) with NO namespace of its own and NO vendor #include. The
 * includer supplies all of that and pastes this inside its own `namespace wwr`
 * -- so one definition serves all three ways the conversions are consumed:
 * bf16.cppm (the module, host, `export namespace wwr`), bf16.h's device section
 * (a device .cu/.cuh, which cannot import the module) and wwr/bf16.h (the
 * non-module #include path, host). Add or change a conversion here, once, and
 * every path gains it.
 *
 * The three paths differ in exactly one token -- the function qualifier -- which
 * the includer supplies as WWR_BF16_FN: `inline` for the two host paths,
 * `__device__ __forceinline__` for the device section. The bodies are identical
 * because __float2bfloat16 / __bfloat162float are external-linkage on both
 * backends (docs/architecture.md section 12 lists only fp4/fp6/fp8 as
 * static-inline), so a host TU and a device TU bind the same vendor entry points.
 *
 * Before including, the includer must have, in order:
 *   - the vendor bf16 header in scope and wwrBfloat16 aliased (bf16.h supplies
 *     both -- the type DIVERGES by backend, __nv_bfloat16 vs __hip_bfloat16);
 *   - WWR_BF16_FN defined to the function qualifier this path wants.
 *
 * See src/bf16.cppm, src/bf16.h and src/wwr/bf16.h.
 */

#pragma once

#ifndef WWR_BF16_FN
#error                                                                                             \
    "detail/bf16_names.h is an include fragment, not a standalone header: define WWR_BF16_FN (inline, or __device__ __forceinline__), ensure the vendor bf16 header and wwrBfloat16 are in scope, and #include it inside namespace wwr. See src/bf16.h, src/bf16.cppm and src/wwr/bf16.h."
#endif

/// @brief Convert a float to bfloat16 (round to nearest even)
WWR_BF16_FN wwrBfloat16 wwrFloat2Bfloat16(const float value) { return ::__float2bfloat16(value); }

/// @brief Widen a bfloat16 value back to float (exact)
WWR_BF16_FN float wwrBfloat162Float(const wwrBfloat16 value) { return ::__bfloat162float(value); }
