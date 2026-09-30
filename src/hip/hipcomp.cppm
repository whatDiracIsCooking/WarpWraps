/**
 * @file hipcomp.cppm
 * @brief hipCOMP API module wrapper for wwr project
 *
 * Wraps hipCOMP -- AMD's GPU lossless-compression library: the batched
 * low-level interface for LZ4, Snappy, Cascaded, GDeflate, Bitcomp and ANS, the
 * generic (metadata-driven) decompress entry points, and the shared status/type
 * enums. CUDA counterpart: wwr.cuda.nvcomp.
 *
 * hipCOMP is a hipify of NVIDIA/nvcomp branch-2.2 -- the last open-source
 * nvCOMP -- so its surface matches nvCOMP 2.2, several major versions behind the
 * 5.3 the CUDA image ships. That version skew is why the backend-neutral
 * wwr.comp layer above this pair (src/comp.cppm) is built from hand-written
 * forwarding shims rather than aliases, and carries only the 2.2 intersection
 * (LZ4/Snappy/Cascaded): the batched signatures diverge (a single opts struct
 * here vs split compress/decompress opts there, no device-status parameter, one
 * temp-size query rather than Sync/Async), so no wwr* alias could present one
 * portable signature. This raw module remains the way to reach the hipCOMP
 * surface wwr.comp omits. See issue #110 (option a).
 *
 * hipCOMP self-describes as an early-access preview and marks every algorithm
 * experimental and not performance-optimized; Bitcomp, ANS and GDeflate are the
 * NVIDIA-proprietary schemes whose headers are present but whose device support
 * is not. This module re-exports the header surface (a compile/link contract),
 * not a runtime guarantee.
 *
 * Not wrapped: the C++ high-level interface managers (hipcompManager.hpp /
 * hipcompManagerFactory.hpp) -- the "optionally later" typed-wrapper layer, not
 * this raw re-export. The default-option structs (hipcompBatched<Algo>DefaultOpts
 * and the low-level *DefaultOpts) are also NOT re-exported: the vendor declares
 * them as file-scope `static const` (internal linkage), which a module may not
 * name in an `export`ed declaration and which -- being class-typed -- has no
 * well-formed constant-expression copy. Construct the exported opts structs
 * directly.
 *
 * Usage:
 *   import wwr.hip.hipcomp;
 */

module;

#include <hipcomp.h>
#include <hipcomp/ans.h>
#include <hipcomp/bitcomp.h>
#include <hipcomp/cascaded.h>
#include <hipcomp/gdeflate.h>
#include <hipcomp/lz4.h>
#include <hipcomp/snappy.h>

export module wwr.hip.hipcomp;

import std;

export namespace wwr::hip {

// ========================================================================
// Dependent type from <hip/hip_runtime.h> that appears in the async signatures
// ========================================================================
using ::hipStream_t;

// ========================================================================
// Shared status enum (hipcomp/shared_types.h)
// ========================================================================
using ::hipcompStatus_t;
using ::hipcompSuccess;
using ::hipcompErrorInvalidValue;
using ::hipcompErrorNotSupported;
using ::hipcompErrorCannotDecompress;
using ::hipcompErrorCudaError;
using ::hipcompErrorInternal;

// ========================================================================
// Data-type enum and generic metadata-driven decompress API (hipcomp.h)
// ========================================================================
using ::hipcompType_t;
using ::HIPCOMP_TYPE_CHAR;
using ::HIPCOMP_TYPE_UCHAR;
using ::HIPCOMP_TYPE_SHORT;
using ::HIPCOMP_TYPE_USHORT;
using ::HIPCOMP_TYPE_INT;
using ::HIPCOMP_TYPE_UINT;
using ::HIPCOMP_TYPE_LONGLONG;
using ::HIPCOMP_TYPE_ULONGLONG;
using ::HIPCOMP_TYPE_BITS;
using ::hipcompDecompressGetMetadata;
using ::hipcompDecompressDestroyMetadata;
using ::hipcompDecompressGetTempSize;
using ::hipcompDecompressGetOutputSize;
using ::hipcompDecompressGetType;
using ::hipcompDecompressAsync;

// ========================================================================
// LZ4 (hipcomp/lz4.h)
// ========================================================================
using ::hipcompLZ4FormatOpts;
using ::hipcompBatchedLZ4Opts_t;
using ::hipcompBatchedLZ4CompressGetTempSize;
using ::hipcompBatchedLZ4CompressGetMaxOutputChunkSize;
using ::hipcompBatchedLZ4CompressAsync;
using ::hipcompBatchedLZ4DecompressGetTempSize;
using ::hipcompBatchedLZ4DecompressAsync;
using ::hipcompBatchedLZ4GetDecompressSizeAsync;

// ========================================================================
// Snappy (hipcomp/snappy.h)
// ========================================================================
using ::hipcompBatchedSnappyOpts_t;
using ::hipcompBatchedSnappyCompressGetTempSize;
using ::hipcompBatchedSnappyCompressGetMaxOutputChunkSize;
using ::hipcompBatchedSnappyCompressAsync;
using ::hipcompBatchedSnappyDecompressGetTempSize;
using ::hipcompBatchedSnappyDecompressAsync;
using ::hipcompBatchedSnappyGetDecompressSizeAsync;

// ========================================================================
// Cascaded (hipcomp/cascaded.h)
// ========================================================================
using ::hipcompCascadedFormatOpts;
using ::hipcompBatchedCascadedOpts_t;
using ::hipcompBatchedCascadedCompressGetTempSize;
using ::hipcompBatchedCascadedCompressGetMaxOutputChunkSize;
using ::hipcompBatchedCascadedCompressAsync;
using ::hipcompBatchedCascadedDecompressGetTempSize;
using ::hipcompBatchedCascadedDecompressAsync;
using ::hipcompBatchedCascadedGetDecompressSizeAsync;

// ========================================================================
// GDeflate (hipcomp/gdeflate.h)
// ========================================================================
using ::hipcompBatchedGdeflateOpts_t;
using ::hipcompBatchedGdeflateCompressGetTempSize;
using ::hipcompBatchedGdeflateCompressGetMaxOutputChunkSize;
using ::hipcompBatchedGdeflateCompressAsync;
using ::hipcompBatchedGdeflateDecompressGetTempSize;
using ::hipcompBatchedGdeflateDecompressAsync;
using ::hipcompBatchedGdeflateGetDecompressSizeAsync;

// ========================================================================
// Bitcomp (hipcomp/bitcomp.h) -- carries a low-level (non-batched) API too
// ========================================================================
using ::hipcompBitcompFormatOpts;
using ::hipcompBitcompCompressConfigure;
using ::hipcompBitcompCompressAsync;
using ::hipcompBitcompDecompressConfigure;
using ::hipcompBitcompDestroyMetadata;
using ::hipcompBitcompDecompressAsync;
using ::hipcompIsBitcompData;
using ::hipcompBatchedBitcompFormatOpts;
using ::hipcompBatchedBitcompCompressGetMaxOutputChunkSize;
using ::hipcompBatchedBitcompCompressAsync;
using ::hipcompBatchedBitcompDecompressAsync;
using ::hipcompBatchedBitcompGetDecompressSizeAsync;
using ::hipcompBatchedBitcompCompressGetTempSize;
using ::hipcompBatchedBitcompDecompressGetTempSize;

// ========================================================================
// ANS (hipcomp/ans.h)
// ========================================================================
using ::hipcompANSType_t;
using ::hipcomp_rANS;
using ::hipcompBatchedANSOpts_t;
using ::hipcompBatchedANSCompressGetTempSize;
using ::hipcompBatchedANSCompressGetMaxOutputChunkSize;
using ::hipcompBatchedANSCompressAsync;
using ::hipcompBatchedANSDecompressGetTempSize;
using ::hipcompBatchedANSDecompressAsync;
using ::hipcompBatchedANSGetDecompressSizeAsync;

} // namespace wwr::hip
