/**
 * @file comp.h
 * @brief The single nvCOMP / hipCOMP vendor-include point for the comp layer
 *
 * A src/-root header that brings in the vendor compression headers for whichever
 * backend was selected, plus <cstddef> for the std::size_t the batched
 * signatures and the Cascaded opts name, and nothing else: the wwrcomp* surface
 * itself is the shared fragment detail/comp_names.h, bound to the `::nvcomp*` /
 * `::hipcomp*` declarations this header supplies. It is the "rand.h shape" for a
 * host-only vendor library -- comp.cppm draws its whole surface from here
 * (binding wwr* references straight to the declarations with the _RAW macros,
 * importing no raw vendor module), and wwr/comp.h (the non-module #include path)
 * does the same, so this is the one place the vendor header enters the comp
 * layer. The raw module wwr.cuda.nvcomp / wwr.hip.hipcomp stays as a
 * single-vendor surface for the full nvCOMP 5.3 / hipCOMP 2.2 surface the neutral
 * layer cannot express (Deflate/GZIP/Zstd/GDeflate/Bitcomp/ANS, CRC32, the
 * hardware-decompression backend, ...), off this path. See src/comp.cppm and
 * src/README.md.
 *
 * Only the LZ4 / Snappy / Cascaded headers are pulled, plus the base header for
 * the shared status/type enums: those three algorithms are the measured hipCOMP
 * 2.2 intersection the neutral layer carries (issue #110). The nvCOMP-only extras
 * live behind the raw module, so their headers do not belong here.
 *
 * HOST only -- nvCOMP/hipCOMP's batched LLIF is a host-launched API, so unlike
 * rand.h / complex.h this header carries no device-pass-gated section. It reaches
 * selected_backend.h directly, not device_guard.h, so it compiles in the host TUs
 * that include it.
 */

#pragma once

// std::size_t for the Cascaded opts' chunk_size and the batched forwarders' size
// parameters (see detail/comp_names.h). <cstddef> on both host paths, so the
// fragment is self-sufficient and needs no `import std` -- which wwr/comp.h could
// not provide anyway. First, so libc++'s __config is processed before any vendor
// header; see the HIP branch below.
#include <cstddef>

// WWR_SELECTED_CUDA / WWR_SELECTED_HIP, from WWR_GPU_BACKEND_* in a host compile.
#include "selected_backend.h"

#if defined(WWR_SELECTED_CUDA)

#include <cuda_runtime.h>

#include <nvcomp.h>
#include <nvcomp/cascaded.h>
#include <nvcomp/lz4.h>
#include <nvcomp/snappy.h>

#else

// No `#include <array>` pre-include here, and deliberately so: unlike sparse.h /
// fft.h / tensor.h (and the HIP raw modules they mirror), the raw
// wwr.hip.hipcomp module's global module fragment carries none either. hipCOMP's
// headers are a hipify of nvCOMP's C API and do not drag in the
// amd_hip_vector_types.h / host_defines.h that poison __noinline__ for libc++'s
// __config (docs/architecture.md §9), so hipcomp.h is not on the §9 "Affected"
// list. <cstddef> above has processed __config cleanly regardless.
#include <hipcomp.h>
#include <hipcomp/cascaded.h>
#include <hipcomp/lz4.h>
#include <hipcomp/snappy.h>

#endif
