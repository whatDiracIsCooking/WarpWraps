/**
 * @file wwr/fp16.h
 * @brief The backend-neutral fp16 surface as a single, self-contained HOST
 *        #include, for a TU that does not use C++ modules
 *
 * The non-module counterpart to `import wwr.fp16;`: a plain .cpp that #includes
 * this one header gets the same wwr* half surface the module exports -- wwrHalf
 * and the float<->half conversions wwrFloat2Half / wwrHalf2Float -- bound to the
 * selected backend's __half and its __float2half / __half2float.
 *
 * It is thin because the fp16 layer already shares its pieces. The type and the
 * vendor header come from fp16.h (the same src/-root header the module's GMF and
 * a device .cu include); a non-module TU sees wwrHalf straight from it, so this
 * file does not re-export it. The two conversions come from
 * detail/fp16_names.h, the one fragment every path shares -- pasted here with the
 * host `inline` qualifier exactly as fp16.cppm's purview pastes it (fp16.h's
 * device section pastes the same list with __device__ __forceinline__). Add a
 * conversion there, once, and this path gains it too.
 *
 * Backend selection and the include root arrive by linking wwr::fp16::host. HOST
 * only: a device .cu/.cuh gets these conversions from fp16.h's device section
 * directly. See src/fp16.cppm, src/fp16.h and docs/architecture.md, section 3.
 */

#pragma once

// HOST only -- see the file header. A device pass gets the conversions from
// fp16.h's own device-pass-gated section.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#error                                                                                              \
    "wwr/fp16.h is the HOST #include path; a device .cu/.cuh gets the fp16 conversions from fp16.h's device section. See src/README.md."
#endif

// wwrHalf and the vendor fp16 header whose __float2half / __half2float the
// conversions call -- reached through fp16.h, the fp16 layer's one neutral
// vendor-include point, not by re-including the vendor header here. Angle
// brackets, not quotes: this file is itself src/wwr/fp16.h, so a quoted "fp16.h"
// would resolve to this file (quotes search the current file's directory first)
// and #pragma-once to nothing; <fp16.h> skips that search and resolves to
// src/fp16.h on the include root. Compiled as host C++, fp16.h's
// device-pass-gated section is absent -- this path supplies the host form below.
#include <fp16.h>

namespace wwr {
// The two conversions, from the one fragment every path shares -- here with the
// host `inline` qualifier, as fp16.cppm's purview has them. See
// detail/fp16_names.h.
#define WWR_FP16_FN inline
#include "detail/fp16_names.h"
#undef WWR_FP16_FN
} // namespace wwr
