/**
 * @file nvcomp.cppm
 * @brief nvCOMP API module wrapper for wwr project
 *
 * Wraps nvCOMP -- NVIDIA's GPU lossless-compression library: the batched
 * low-level interface (LLIF) for LZ4, Snappy, Cascaded, Deflate, GZIP, Zstd,
 * GDeflate, Bitcomp and ANS, plus the CRC32 checksum API and the shared
 * status/type enums. HIP counterpart: wwr.hip.hipcomp.
 *
 * There is deliberately NO backend-neutral wwr.comp layer above this pair. The
 * two libraries are version-skewed -- nvCOMP is 5.3, hipCOMP is a hipify of
 * nvCOMP 2.2 -- so the batched signatures diverge (split compress/decompress
 * opts, an extra device-status / decompress-opts parameter, and Sync/Async
 * temp-size queries with no 2.2 counterpart). A wwr* alias could not present one
 * portable signature, so each backend is reachable only through its own vendor
 * module. See issue #110 for the measured intersection.
 *
 * Not wrapped: the C++ high-level interface (HLIF) managers
 * (nvcompManager.hpp / nvcompManagerFactory.hpp) and the CPU managers -- those
 * are the "optionally later" typed-wrapper layer, not this raw re-export. The
 * version macros (NVCOMP_VER*, the *_FROM_SEMVER helpers) are likewise not
 * exposed: their values are backend-specific and nvcompGetProperties is the
 * queryable equivalent.
 *
 * Two families of vendor constants are handled specially, both because the
 * headers declare them as file-scope `static const` (internal linkage), which a
 * module may not name in an `export`ed declaration:
 *   - The per-algorithm chunk-size limits and alignment requirements are
 *     integral, so they are usable in constant expressions and re-declared here
 *     as `inline constexpr size_t` copies -- drift-proof, no ODR-use of the
 *     internal object.
 *   - The default-option structs (nvcompBatched<Algo>{Compress,Decompress}Opts)
 *     and the CRC32 model presets (nvcompCRC32*) are class-typed, so no such
 *     copy is well-formed; they are NOT re-exported. Construct the exported opts
 *     structs directly (all are class-typed and exported), or read the vendor
 *     header for the CRC32 presets.
 *
 * Usage:
 *   import wwr.cuda.nvcomp;
 */

module;

#include <nvcomp.h>
#include <nvcomp/ans.h>
#include <nvcomp/bitcomp.h>
#include <nvcomp/cascaded.h>
#include <nvcomp/crc32.h>
#include <nvcomp/deflate.h>
#include <nvcomp/gdeflate.h>
#include <nvcomp/gzip.h>
#include <nvcomp/lz4.h>
#include <nvcomp/snappy.h>
#include <nvcomp/zstd.h>

export module wwr.cuda.nvcomp;

import std;

export namespace wwr::cuda {

// ========================================================================
// Dependent type from <cuda_runtime.h> that appears in the async signatures
// ========================================================================
using ::cudaStream_t;

// ========================================================================
// Shared status / data-type / backend enums (nvcomp/shared_types.h)
// ========================================================================
using ::nvcompStatus_t;
using ::nvcompSuccess;
using ::nvcompErrorInvalidValue;
using ::nvcompErrorNotSupported;
using ::nvcompErrorCannotDecompress;
using ::nvcompErrorBadChecksum;
using ::nvcompErrorCannotVerifyChecksums;
using ::nvcompErrorOutputBufferTooSmall;
using ::nvcompErrorWrongHeaderLength;
using ::nvcompErrorAlignment;
using ::nvcompErrorChunkSizeTooLarge;
using ::nvcompErrorCannotCompress;
using ::nvcompErrorWrongInputLength;
using ::nvcompErrorBatchSizeTooLarge;
using ::nvcompErrorSubChunkCountTooLarge;
using ::nvcompErrorSubChunkCountTooSmall;
using ::nvcompErrorOutputBufferAlignmentTooSmall;
using ::nvcompErrorCudaError;
using ::nvcompErrorInternal;

using ::nvcompType_t;
using ::NVCOMP_TYPE_CHAR;
using ::NVCOMP_TYPE_UCHAR;
using ::NVCOMP_TYPE_SHORT;
using ::NVCOMP_TYPE_USHORT;
using ::NVCOMP_TYPE_INT;
using ::NVCOMP_TYPE_UINT;
using ::NVCOMP_TYPE_LONGLONG;
using ::NVCOMP_TYPE_ULONGLONG;
using ::NVCOMP_TYPE_FLOAT16;
using ::NVCOMP_TYPE_FLOAT8_E4M3;
using ::NVCOMP_TYPE_BITS;

using ::nvcompDecompressBackend_t;
using ::NVCOMP_DECOMPRESS_BACKEND_DEFAULT;
using ::NVCOMP_DECOMPRESS_BACKEND_HARDWARE;
using ::NVCOMP_DECOMPRESS_BACKEND_CUDA;

using ::nvcompProperties_t;
using ::nvcompAlignmentRequirements_t;

using ::nvcompBitshuffleMode_t;
using ::NVCOMP_BITSHUFFLE_NONE;
using ::NVCOMP_BITSHUFFLE_MSB_FIRST;
using ::NVCOMP_BITSHUFFLE_LSB_FIRST;

// ========================================================================
// Library properties / status string (nvcomp.h)
// ========================================================================
using ::nvcompGetProperties;
using ::nvcompGetStatusString;

// ========================================================================
// LZ4 (nvcomp/lz4.h)
// ========================================================================
using ::nvcompBatchedLZ4CompressOpts_t;
using ::nvcompBatchedLZ4DecompressOpts_t;
inline constexpr size_t nvcompLZ4CompressionMaxAllowedChunkSize = ::nvcompLZ4CompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompLZ4DecompressionMaxAllowedChunkSize = ::nvcompLZ4DecompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompLZ4RequiredCompressionAlignment = ::nvcompLZ4RequiredCompressionAlignment;
inline constexpr size_t nvcompLZ4RequiredDecompressionAlignment = ::nvcompLZ4RequiredDecompressionAlignment;
using ::nvcompBatchedLZ4CompressGetRequiredAlignments;
using ::nvcompBatchedLZ4CompressGetTempSizeAsync;
using ::nvcompBatchedLZ4CompressGetTempSizeSync;
using ::nvcompBatchedLZ4CompressGetMaxOutputChunkSize;
using ::nvcompBatchedLZ4CompressAsync;
using ::nvcompBatchedLZ4DecompressGetRequiredAlignments;
using ::nvcompBatchedLZ4DecompressGetTempSizeAsync;
using ::nvcompBatchedLZ4DecompressGetTempSizeSync;
using ::nvcompBatchedLZ4GetDecompressSizeAsync;
using ::nvcompBatchedLZ4DecompressAsync;

// ========================================================================
// Snappy (nvcomp/snappy.h)
// ========================================================================
using ::nvcompBatchedSnappyCompressOpts_t;
using ::nvcompBatchedSnappyDecompressOpts_t;
inline constexpr size_t nvcompSnappyCompressionMaxAllowedChunkSize = ::nvcompSnappyCompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompSnappyDecompressionMaxAllowedChunkSize = ::nvcompSnappyDecompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompSnappyRequiredCompressionAlignment = ::nvcompSnappyRequiredCompressionAlignment;
inline constexpr size_t nvcompSnappyRequiredDecompressionAlignment = ::nvcompSnappyRequiredDecompressionAlignment;
using ::nvcompBatchedSnappyCompressGetRequiredAlignments;
using ::nvcompBatchedSnappyCompressGetTempSizeAsync;
using ::nvcompBatchedSnappyCompressGetTempSizeSync;
using ::nvcompBatchedSnappyCompressGetMaxOutputChunkSize;
using ::nvcompBatchedSnappyCompressAsync;
using ::nvcompBatchedSnappyDecompressGetRequiredAlignments;
using ::nvcompBatchedSnappyDecompressGetTempSizeAsync;
using ::nvcompBatchedSnappyDecompressGetTempSizeSync;
using ::nvcompBatchedSnappyGetDecompressSizeAsync;
using ::nvcompBatchedSnappyDecompressAsync;

// ========================================================================
// Cascaded (nvcomp/cascaded.h)
// ========================================================================
using ::nvcompBatchedCascadedCompressOpts_t;
using ::nvcompBatchedCascadedDecompressOpts_t;
inline constexpr size_t nvcompCascadedCompressionMaxAllowedChunkSize = ::nvcompCascadedCompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompCascadedDecompressionMaxAllowedChunkSize = ::nvcompCascadedDecompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompCascadedRequiredCompressionAlignment = ::nvcompCascadedRequiredCompressionAlignment;
inline constexpr size_t nvcompCascadedRequiredDecompressionAlignment = ::nvcompCascadedRequiredDecompressionAlignment;
using ::nvcompBatchedCascadedCompressGetRequiredAlignments;
using ::nvcompBatchedCascadedCompressGetTempSizeAsync;
using ::nvcompBatchedCascadedCompressGetTempSizeSync;
using ::nvcompBatchedCascadedCompressGetMaxOutputChunkSize;
using ::nvcompBatchedCascadedCompressAsync;
using ::nvcompBatchedCascadedDecompressGetRequiredAlignments;
using ::nvcompBatchedCascadedDecompressGetTempSizeAsync;
using ::nvcompBatchedCascadedDecompressGetTempSizeSync;
using ::nvcompBatchedCascadedGetDecompressSizeAsync;
using ::nvcompBatchedCascadedDecompressAsync;

// ========================================================================
// Deflate (nvcomp/deflate.h)
// ========================================================================
using ::nvcompBatchedDeflateCompressOpts_t;
using ::nvcompBatchedDeflateDecompressOpts_t;
inline constexpr size_t nvcompDeflateCompressionMaxAllowedChunkSize = ::nvcompDeflateCompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompDeflateDecompressionMaxAllowedChunkSize = ::nvcompDeflateDecompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompDeflateRequiredCompressionAlignment = ::nvcompDeflateRequiredCompressionAlignment;
inline constexpr size_t nvcompDeflateRequiredDecompressionAlignment = ::nvcompDeflateRequiredDecompressionAlignment;
using ::nvcompBatchedDeflateCompressGetRequiredAlignments;
using ::nvcompBatchedDeflateCompressGetTempSizeAsync;
using ::nvcompBatchedDeflateCompressGetTempSizeSync;
using ::nvcompBatchedDeflateCompressGetMaxOutputChunkSize;
using ::nvcompBatchedDeflateCompressAsync;
using ::nvcompBatchedDeflateDecompressGetRequiredAlignments;
using ::nvcompBatchedDeflateDecompressGetTempSizeAsync;
using ::nvcompBatchedDeflateDecompressGetTempSizeSync;
using ::nvcompBatchedDeflateGetDecompressSizeAsync;
using ::nvcompBatchedDeflateDecompressAsync;

// ========================================================================
// GZIP (nvcomp/gzip.h)
// ========================================================================
using ::nvcompBatchedGzipCompressOpts_t;
using ::nvcompBatchedGzipDecompressAlgorithm_t;
using ::NVCOMP_GZIP_DECOMPRESS_ALGORITHM_NAIVE;
using ::NVCOMP_GZIP_DECOMPRESS_ALGORITHM_LOOKAHEAD;
using ::nvcompBatchedGzipDecompressOpts_t;
inline constexpr size_t nvcompGzipCompressionMaxAllowedChunkSize = ::nvcompGzipCompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompGzipNaiveDecompressionMaxAllowedChunkSize = ::nvcompGzipNaiveDecompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompGzipLookaheadDecompressionMaxAllowedChunkSize = ::nvcompGzipLookaheadDecompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompGzipRequiredDecompressionAlignment = ::nvcompGzipRequiredDecompressionAlignment;
using ::nvcompBatchedGzipCompressGetRequiredAlignments;
using ::nvcompBatchedGzipCompressGetTempSizeAsync;
using ::nvcompBatchedGzipCompressGetTempSizeSync;
using ::nvcompBatchedGzipCompressGetMaxOutputChunkSize;
using ::nvcompBatchedGzipCompressAsync;
using ::nvcompBatchedGzipDecompressGetRequiredAlignments;
using ::nvcompBatchedGzipDecompressGetTempSizeAsync;
using ::nvcompBatchedGzipDecompressGetTempSizeSync;
using ::nvcompBatchedGzipGetDecompressSizeAsync;
using ::nvcompBatchedGzipDecompressAsync;

// ========================================================================
// Zstd (nvcomp/zstd.h)
// ========================================================================
using ::nvcompBatchedZstdCompressOpts_t;
using ::nvcompBatchedZstdDecompressOpts_t;
inline constexpr size_t nvcompZstdCompressionMaxAllowedChunkSize = ::nvcompZstdCompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompZstdDecompressionMaxAllowedChunkSize = ::nvcompZstdDecompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompZstdRequiredCompressionAlignment = ::nvcompZstdRequiredCompressionAlignment;
inline constexpr size_t nvcompZstdRequiredDecompressionAlignment = ::nvcompZstdRequiredDecompressionAlignment;
using ::nvcompBatchedZstdCompressGetRequiredAlignments;
using ::nvcompBatchedZstdCompressGetTempSizeAsync;
using ::nvcompBatchedZstdCompressGetTempSizeSync;
using ::nvcompBatchedZstdCompressGetMaxOutputChunkSize;
using ::nvcompBatchedZstdCompressAsync;
using ::nvcompBatchedZstdDecompressGetRequiredAlignments;
using ::nvcompBatchedZstdDecompressGetTempSizeAsync;
using ::nvcompBatchedZstdDecompressGetTempSizeSync;
using ::nvcompBatchedZstdGetDecompressSizeAsync;
using ::nvcompBatchedZstdDecompressAsync;

// ========================================================================
// GDeflate (nvcomp/gdeflate.h)
// ========================================================================
using ::nvcompBatchedGdeflateCompressOpts_t;
using ::nvcompBatchedGdeflateDecompressOpts_t;
inline constexpr size_t nvcompGdeflateCompressionMaxAllowedChunkSize = ::nvcompGdeflateCompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompGdeflateDecompressionMaxAllowedChunkSize = ::nvcompGdeflateDecompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompGdeflateRequiredCompressionAlignment = ::nvcompGdeflateRequiredCompressionAlignment;
inline constexpr size_t nvcompGdeflateRequiredDecompressionAlignment = ::nvcompGdeflateRequiredDecompressionAlignment;
using ::nvcompBatchedGdeflateCompressGetRequiredAlignments;
using ::nvcompBatchedGdeflateCompressGetTempSizeAsync;
using ::nvcompBatchedGdeflateCompressGetTempSizeSync;
using ::nvcompBatchedGdeflateCompressGetMaxOutputChunkSize;
using ::nvcompBatchedGdeflateCompressAsync;
using ::nvcompBatchedGdeflateDecompressGetRequiredAlignments;
using ::nvcompBatchedGdeflateDecompressGetTempSizeAsync;
using ::nvcompBatchedGdeflateDecompressGetTempSizeSync;
using ::nvcompBatchedGdeflateGetDecompressSizeAsync;
using ::nvcompBatchedGdeflateDecompressAsync;

// ========================================================================
// Bitcomp (nvcomp/bitcomp.h)
// ========================================================================
using ::nvcompBatchedBitcompCompressOpts_t;
using ::nvcompBatchedBitcompDecompressOpts_t;
inline constexpr size_t nvcompBitcompCompressionMaxAllowedChunkSize = ::nvcompBitcompCompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompBitcompDecompressionMaxAllowedChunkSize = ::nvcompBitcompDecompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompBitcompRequiredCompressionAlignment = ::nvcompBitcompRequiredCompressionAlignment;
inline constexpr size_t nvcompBitcompRequiredDecompressionAlignment = ::nvcompBitcompRequiredDecompressionAlignment;
using ::nvcompBatchedBitcompCompressGetRequiredAlignments;
using ::nvcompBatchedBitcompCompressGetTempSizeAsync;
using ::nvcompBatchedBitcompCompressGetTempSizeSync;
using ::nvcompBatchedBitcompCompressGetMaxOutputChunkSize;
using ::nvcompBatchedBitcompCompressAsync;
using ::nvcompBatchedBitcompDecompressGetRequiredAlignments;
using ::nvcompBatchedBitcompDecompressGetTempSizeAsync;
using ::nvcompBatchedBitcompDecompressGetTempSizeSync;
using ::nvcompBatchedBitcompGetDecompressSizeAsync;
using ::nvcompBatchedBitcompDecompressAsync;

// ========================================================================
// ANS (nvcomp/ans.h)
// ========================================================================
using ::nvcompANSType_t;
using ::nvcomp_rANS;
using ::nvcompBatchedANSCompressOpts_t;
using ::nvcompBatchedANSDecompressOpts_t;
inline constexpr size_t nvcompANSCompressionMaxAllowedChunkSize = ::nvcompANSCompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompANSDecompressionMaxAllowedChunkSize = ::nvcompANSDecompressionMaxAllowedChunkSize;
inline constexpr size_t nvcompANSRequiredCompressionAlignment = ::nvcompANSRequiredCompressionAlignment;
inline constexpr size_t nvcompANSRequiredDecompressionAlignment = ::nvcompANSRequiredDecompressionAlignment;
using ::nvcompBatchedANSCompressGetRequiredAlignments;
using ::nvcompBatchedANSCompressGetTempSizeAsync;
using ::nvcompBatchedANSCompressGetTempSizeSync;
using ::nvcompBatchedANSCompressGetMaxOutputChunkSize;
using ::nvcompBatchedANSCompressAsync;
using ::nvcompBatchedANSDecompressGetRequiredAlignments;
using ::nvcompBatchedANSDecompressGetTempSizeAsync;
using ::nvcompBatchedANSDecompressGetTempSizeSync;
using ::nvcompBatchedANSGetDecompressSizeAsync;
using ::nvcompBatchedANSDecompressAsync;

// ========================================================================
// CRC32 checksums (nvcomp/crc32.h). The model presets (the nvcompCRC32*
// nvcompCRC32Spec_t constants) are static-const structs -- see the file header
// for why they are not re-exported; construct a nvcompCRC32Spec_t directly.
// ========================================================================
using ::nvcompCRC32Spec_t;
using ::nvcompCRC32KernelKind_t;
using ::nvcompCRC32WarpKernel;
using ::nvcompCRC32BlockKernel;
using ::nvcompCRC32KernelConf_t;
using ::nvcompBatchedCRC32Opts_t;
using ::nvcompCRC32SegmentKind_t;
using ::nvcompCRC32OnlySegment;
using ::nvcompCRC32FirstSegment;
using ::nvcompCRC32MidSegment;
using ::nvcompCRC32LastSegment;
inline constexpr size_t nvcompCRC32DeducedMaxInputChunkBytes = ::nvcompCRC32DeducedMaxInputChunkBytes;
using ::nvcompBatchedCRC32Async;
using ::nvcompBatchedCRC32GetHeuristicConf;
using ::nvcompBatchedCRC32SearchConf;

} // namespace wwr::cuda
