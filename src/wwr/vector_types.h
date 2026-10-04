/**
 * @file wwr/vector_types.h
 * @brief The backend-neutral vector-types surface as a single, self-contained
 *        HOST #include, for a TU that does not use C++ modules
 *
 * The non-module counterpart to `import wwr.vector_types;`: a plain .cpp that
 * #includes this one header gets the same wwr* surface the module exports -- the
 * float2 / int4 / ... aliases and the host make_* constructors -- bound to the
 * selected backend's identically-spelled vector types.
 *
 * It is thin because the vector-types layer already shares its pieces. The type
 * aliases and the vendor header come from vector_types.h (the same src/-root
 * header the module's GMF and a device .cu include); a non-module TU sees
 * wwr::float2 straight from it, so this file does not re-export them. The make_*
 * constructors come from detail/vector_types_names.h, the one fragment every path
 * shares -- pasted here with the host `inline` qualifier exactly as
 * vector_types.cppm's purview pastes it (vector_types.h's device section pastes the
 * same list with __device__ __forceinline__). Add a constructor there, once, and
 * this path gains it too.
 *
 * Unlike fp16's host path, the make_* name no vendor symbol: construction is a
 * portable brace T{...} (a CUDA aggregate, a HIP HIP_vector_type constructor), so
 * this file defines no _RAW binding macros -- it only sets WWR_VT_FN. See
 * vector_types.h.
 *
 * Backend selection and the include root arrive by linking wwr::vector_types::host.
 * HOST only: a device .cu/.cuh gets these constructors from vector_types.h's
 * device section directly. See src/vector_types.cppm, src/vector_types.h and
 * docs/architecture.md, section 3.
 */

#pragma once

// HOST only -- see the file header. A device pass gets the constructors from
// vector_types.h's own device-pass-gated section.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#error                                                                                              \
    "wwr/vector_types.h is the HOST #include path; a device .cu/.cuh gets the vector-type constructors from vector_types.h's device section. See src/README.md."
#endif

// The vector-type aliases and the vendor header the make_* brace over -- reached
// through vector_types.h, the vector-types layer's one neutral vendor-include
// point, not by re-including the vendor header here. Angle brackets, not quotes:
// this file is itself src/wwr/vector_types.h, so a quoted "vector_types.h" would
// resolve to this file (quotes search the current file's directory first) and
// #pragma-once to nothing; <vector_types.h> skips that search and resolves to
// src/vector_types.h on the include root. Compiled as host C++, that header's
// device-pass-gated section is absent -- this path supplies the host form below.
#include <vector_types.h>

namespace wwr {
// The make_* constructors, from the one fragment every path shares -- here with
// the host `inline` qualifier, as vector_types.cppm's purview has them. See
// detail/vector_types_names.h.
#define WWR_VT_FN inline
#include "detail/vector_types_names.h"
#undef WWR_VT_FN
} // namespace wwr
