/**
 * @file detail/vector_types_names.h
 * @brief The vector-type make_* constructors, as a macro-driven include fragment
 *        shared by every consumption path
 *
 * NOT a standalone header: it is the make_<type> constructors (make_float2 /
 * make_int4 / ...) with NO namespace of its own and NO vendor #include. The
 * includer supplies all of that and pastes this inside its own `namespace wwr`
 * -- so one definition serves all three ways the constructors are consumed:
 * vector_types.cppm (the module, host, `export namespace wwr`),
 * vector_types.h's device section (a device .cu/.cuh, which cannot import the
 * module) and wwr/vector_types.h (the non-module #include path, host). Add or
 * change a constructor here, once, and every path gains it.
 *
 * The three paths differ in exactly one token -- the function qualifier -- which
 * the includer supplies as WWR_VT_FN: `inline` for the two host paths,
 * `__device__ __forceinline__` for the device section. The bodies are identical
 * because construction is a portable brace T{...} on both backends (a CUDA
 * aggregate, a HIP HIP_vector_type constructor) naming no vendor symbol -- so
 * unlike fp16/bf16 this fragment binds no vendor entry point at all, and unlike
 * blas it needs no _RAW macros. See vector_types.h for why no vendor make_* is
 * called.
 *
 * The local WWR_VT_MAKE1..4 helpers expand to one constructor apiece and are
 * #undef'd at the end; the element type E matches the vendor's own make_*
 * parameter type, so no brace narrows. The scope -- the 12 scalar bases in ranks
 * 1-3, rank 4 only for the bases CUDA 13 does not deprecate -- is documented in
 * vector_types.h.
 *
 * Before including, the includer must have, in order:
 *   - the vector-type aliases in `namespace wwr` (vector_types.h supplies them);
 *   - WWR_VT_FN defined to the function qualifier this path wants.
 *
 * See src/vector_types.cppm, src/vector_types.h and src/wwr/vector_types.h.
 */

#pragma once

#ifndef WWR_VT_FN
#error                                                                                             \
    "detail/vector_types_names.h is an include fragment, not a standalone header: define WWR_VT_FN (inline, or __device__ __forceinline__), ensure the vector-type aliases are in scope, and #include it inside namespace wwr. See src/vector_types.h, src/vector_types.cppm and src/wwr/vector_types.h."
#endif

#define WWR_VT_MAKE1(T, E)                                                                          \
  WWR_VT_FN T make_##T(const E x) { return T{x}; }
#define WWR_VT_MAKE2(T, E)                                                                          \
  WWR_VT_FN T make_##T(const E x, const E y) { return T{x, y}; }
#define WWR_VT_MAKE3(T, E)                                                                          \
  WWR_VT_FN T make_##T(const E x, const E y, const E z) { return T{x, y, z}; }
#define WWR_VT_MAKE4(T, E)                                                                          \
  WWR_VT_FN T make_##T(const E x, const E y, const E z, const E w) { return T{x, y, z, w}; }

WWR_VT_MAKE1(char1, signed char)
WWR_VT_MAKE2(char2, signed char)
WWR_VT_MAKE3(char3, signed char)
WWR_VT_MAKE4(char4, signed char)
WWR_VT_MAKE1(uchar1, unsigned char)
WWR_VT_MAKE2(uchar2, unsigned char)
WWR_VT_MAKE3(uchar3, unsigned char)
WWR_VT_MAKE4(uchar4, unsigned char)

WWR_VT_MAKE1(short1, short)
WWR_VT_MAKE2(short2, short)
WWR_VT_MAKE3(short3, short)
WWR_VT_MAKE4(short4, short)
WWR_VT_MAKE1(ushort1, unsigned short)
WWR_VT_MAKE2(ushort2, unsigned short)
WWR_VT_MAKE3(ushort3, unsigned short)
WWR_VT_MAKE4(ushort4, unsigned short)

WWR_VT_MAKE1(int1, int)
WWR_VT_MAKE2(int2, int)
WWR_VT_MAKE3(int3, int)
WWR_VT_MAKE4(int4, int)
WWR_VT_MAKE1(uint1, unsigned int)
WWR_VT_MAKE2(uint2, unsigned int)
WWR_VT_MAKE3(uint3, unsigned int)
WWR_VT_MAKE4(uint4, unsigned int)

WWR_VT_MAKE1(long1, long int)
WWR_VT_MAKE2(long2, long int)
WWR_VT_MAKE3(long3, long int)
WWR_VT_MAKE1(ulong1, unsigned long int)
WWR_VT_MAKE2(ulong2, unsigned long int)
WWR_VT_MAKE3(ulong3, unsigned long int)

WWR_VT_MAKE1(longlong1, long long int)
WWR_VT_MAKE2(longlong2, long long int)
WWR_VT_MAKE3(longlong3, long long int)
WWR_VT_MAKE1(ulonglong1, unsigned long long int)
WWR_VT_MAKE2(ulonglong2, unsigned long long int)
WWR_VT_MAKE3(ulonglong3, unsigned long long int)

WWR_VT_MAKE1(float1, float)
WWR_VT_MAKE2(float2, float)
WWR_VT_MAKE3(float3, float)
WWR_VT_MAKE4(float4, float)

WWR_VT_MAKE1(double1, double)
WWR_VT_MAKE2(double2, double)
WWR_VT_MAKE3(double3, double)

#undef WWR_VT_MAKE1
#undef WWR_VT_MAKE2
#undef WWR_VT_MAKE3
#undef WWR_VT_MAKE4
