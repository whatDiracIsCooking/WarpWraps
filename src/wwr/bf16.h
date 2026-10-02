/**
 * @file wwr/bf16.h
 * @brief The backend-neutral bf16 surface as a single, self-contained HOST
 *        #include, for a TU that does not use C++ modules
 *
 * The non-module counterpart to `import wwr.bf16;`: a plain .cpp that #includes
 * this one header gets the same wwr* bfloat16 surface the module exports --
 * wwrBfloat16 and the float<->bfloat16 conversions wwrFloat2Bfloat16 /
 * wwrBfloat162Float -- bound to the selected backend's __nv_bfloat16 /
 * __hip_bfloat16 and its __float2bfloat16 / __bfloat162float.
 *
 * It is thin because the bf16 layer already shares its pieces. The type (which
 * DIVERGES by backend) and the vendor header come from bf16.h (the same src/-root
 * header the module's GMF and a device .cu include); a non-module TU sees
 * wwrBfloat16 straight from it, so this file does not re-export it. The two
 * conversions come from detail/bf16_names.h, the one fragment every path shares
 * -- pasted here with the host `inline` qualifier exactly as bf16.cppm's purview
 * pastes it (bf16.h's device section pastes the same list with __device__
 * __forceinline__). Add a conversion there, once, and this path gains it too.
 *
 * Backend selection and the include root arrive by linking wwr::bf16::host. HOST
 * only: a device .cu/.cuh gets these conversions from bf16.h's device section
 * directly. See src/bf16.cppm, src/bf16.h and docs/architecture.md, section 3.
 */

#pragma once

// HOST only -- see the file header. A device pass gets the conversions from
// bf16.h's own device-pass-gated section.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#error                                                                                              \
    "wwr/bf16.h is the HOST #include path; a device .cu/.cuh gets the bf16 conversions from bf16.h's device section. See src/README.md."
#endif

// wwrBfloat16 and the vendor bf16 header whose __float2bfloat16 / __bfloat162float
// the conversions call -- reached through bf16.h, the bf16 layer's one neutral
// vendor-include point, not by re-including the vendor header here. Angle
// brackets, not quotes: this file is itself src/wwr/bf16.h, so a quoted "bf16.h"
// would resolve to this file (quotes search the current file's directory first)
// and #pragma-once to nothing; <bf16.h> skips that search and resolves to
// src/bf16.h on the include root. Compiled as host C++, bf16.h's
// device-pass-gated section is absent -- this path supplies the host form below.
#include <bf16.h>

namespace wwr {
// The two conversions, from the one fragment every path shares -- here with the
// host `inline` qualifier, as bf16.cppm's purview has them. See
// detail/bf16_names.h.
#define WWR_BF16_FN inline
#include "detail/bf16_names.h"
#undef WWR_BF16_FN
} // namespace wwr
