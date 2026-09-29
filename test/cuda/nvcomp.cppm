// nvcomp.cppm - Compile-time tests for wwr.cuda.nvcomp

module;

#include "test/shared/link_check.h"

export module wwr.test.cuda.nvcomp;

import std;
import wwr.cuda.nvcomp;

// ========================================================================
// Compile-time tests for wwr.cuda.nvcomp
//
// Verifies at compile-time that:
//   - Enum types satisfy std::is_enum_v
//   - Enumerator values match the nvCOMP-specified integers (matching NAMES
//     never guarantee matching VALUES across backends -- cf. WWRRAND_RNG_*)
// Link-time checks (WWR_LINK_CHECK) verify that every re-exported batched-LLIF
// function symbol resolves against libnvcomp at link time. The re-exported
// data (default-option structs, chunk-size limits, CRC32 presets) is not
// link-checked: those are inline definitions inside the module, not external
// library symbols.
// ========================================================================

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<nvcompStatus_t>);
static_assert(std::is_enum_v<nvcompType_t>);
static_assert(std::is_enum_v<nvcompDecompressBackend_t>);
static_assert(std::is_enum_v<nvcompBitshuffleMode_t>);
static_assert(std::is_enum_v<nvcompANSType_t>);
static_assert(std::is_enum_v<nvcompBatchedGzipDecompressAlgorithm_t>);
static_assert(std::is_enum_v<nvcompCRC32KernelKind_t>);
static_assert(std::is_enum_v<nvcompCRC32SegmentKind_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvcompStatus_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(nvcompSuccess) == 0);
static_assert(static_cast<int>(nvcompErrorInvalidValue) == 10);
static_assert(static_cast<int>(nvcompErrorNotSupported) == 11);
static_assert(static_cast<int>(nvcompErrorCannotDecompress) == 12);
static_assert(static_cast<int>(nvcompErrorBadChecksum) == 13);
static_assert(static_cast<int>(nvcompErrorCannotVerifyChecksums) == 14);
static_assert(static_cast<int>(nvcompErrorOutputBufferTooSmall) == 15);
static_assert(static_cast<int>(nvcompErrorWrongHeaderLength) == 16);
static_assert(static_cast<int>(nvcompErrorAlignment) == 17);
static_assert(static_cast<int>(nvcompErrorChunkSizeTooLarge) == 18);
static_assert(static_cast<int>(nvcompErrorCannotCompress) == 19);
static_assert(static_cast<int>(nvcompErrorWrongInputLength) == 20);
static_assert(static_cast<int>(nvcompErrorBatchSizeTooLarge) == 21);
static_assert(static_cast<int>(nvcompErrorSubChunkCountTooLarge) == 22);
static_assert(static_cast<int>(nvcompErrorSubChunkCountTooSmall) == 23);
static_assert(static_cast<int>(nvcompErrorOutputBufferAlignmentTooSmall) == 24);
static_assert(static_cast<int>(nvcompErrorCudaError) == 1000);
static_assert(static_cast<int>(nvcompErrorInternal) == 10000);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvcompType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVCOMP_TYPE_CHAR) == 0);
static_assert(static_cast<int>(NVCOMP_TYPE_UCHAR) == 1);
static_assert(static_cast<int>(NVCOMP_TYPE_SHORT) == 2);
static_assert(static_cast<int>(NVCOMP_TYPE_USHORT) == 3);
static_assert(static_cast<int>(NVCOMP_TYPE_INT) == 4);
static_assert(static_cast<int>(NVCOMP_TYPE_UINT) == 5);
static_assert(static_cast<int>(NVCOMP_TYPE_LONGLONG) == 6);
static_assert(static_cast<int>(NVCOMP_TYPE_ULONGLONG) == 7);
static_assert(static_cast<int>(NVCOMP_TYPE_FLOAT16) == 9);
static_assert(static_cast<int>(NVCOMP_TYPE_FLOAT8_E4M3) == 10);
static_assert(static_cast<int>(NVCOMP_TYPE_BITS) == 0xff);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvcompDecompressBackend_t / nvcompBitshuffleMode_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVCOMP_DECOMPRESS_BACKEND_DEFAULT) == 0);
static_assert(static_cast<int>(NVCOMP_DECOMPRESS_BACKEND_HARDWARE) == 1);
static_assert(static_cast<int>(NVCOMP_DECOMPRESS_BACKEND_CUDA) == 2);

static_assert(static_cast<int>(NVCOMP_BITSHUFFLE_NONE) == 0);
static_assert(static_cast<int>(NVCOMP_BITSHUFFLE_MSB_FIRST) == 1);
static_assert(static_cast<int>(NVCOMP_BITSHUFFLE_LSB_FIRST) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvcompBatchedGzipDecompressAlgorithm_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVCOMP_GZIP_DECOMPRESS_ALGORITHM_NAIVE) == 0);
static_assert(static_cast<int>(NVCOMP_GZIP_DECOMPRESS_ALGORITHM_LOOKAHEAD) == 1);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution: batched LLIF + umbrella + CRC32
// ────────────────────────────────────────────────────────────────────────

// nvcomp.h
WWR_LINK_CHECK(nvcompGetProperties)
WWR_LINK_CHECK(nvcompGetStatusString)

// lz4.h
WWR_LINK_CHECK(nvcompBatchedLZ4CompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedLZ4CompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedLZ4CompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedLZ4CompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(nvcompBatchedLZ4CompressAsync)
WWR_LINK_CHECK(nvcompBatchedLZ4DecompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedLZ4DecompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedLZ4DecompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedLZ4GetDecompressSizeAsync)
WWR_LINK_CHECK(nvcompBatchedLZ4DecompressAsync)

// snappy.h
WWR_LINK_CHECK(nvcompBatchedSnappyCompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedSnappyCompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedSnappyCompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedSnappyCompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(nvcompBatchedSnappyCompressAsync)
WWR_LINK_CHECK(nvcompBatchedSnappyDecompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedSnappyDecompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedSnappyDecompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedSnappyGetDecompressSizeAsync)
WWR_LINK_CHECK(nvcompBatchedSnappyDecompressAsync)

// cascaded.h
WWR_LINK_CHECK(nvcompBatchedCascadedCompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedCascadedCompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedCascadedCompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedCascadedCompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(nvcompBatchedCascadedCompressAsync)
WWR_LINK_CHECK(nvcompBatchedCascadedDecompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedCascadedDecompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedCascadedDecompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedCascadedGetDecompressSizeAsync)
WWR_LINK_CHECK(nvcompBatchedCascadedDecompressAsync)

// deflate.h
WWR_LINK_CHECK(nvcompBatchedDeflateCompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedDeflateCompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedDeflateCompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedDeflateCompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(nvcompBatchedDeflateCompressAsync)
WWR_LINK_CHECK(nvcompBatchedDeflateDecompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedDeflateDecompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedDeflateDecompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedDeflateGetDecompressSizeAsync)
WWR_LINK_CHECK(nvcompBatchedDeflateDecompressAsync)

// gzip.h
WWR_LINK_CHECK(nvcompBatchedGzipCompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedGzipCompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedGzipCompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedGzipCompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(nvcompBatchedGzipCompressAsync)
WWR_LINK_CHECK(nvcompBatchedGzipDecompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedGzipDecompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedGzipDecompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedGzipGetDecompressSizeAsync)
WWR_LINK_CHECK(nvcompBatchedGzipDecompressAsync)

// zstd.h
WWR_LINK_CHECK(nvcompBatchedZstdCompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedZstdCompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedZstdCompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedZstdCompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(nvcompBatchedZstdCompressAsync)
WWR_LINK_CHECK(nvcompBatchedZstdDecompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedZstdDecompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedZstdDecompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedZstdGetDecompressSizeAsync)
WWR_LINK_CHECK(nvcompBatchedZstdDecompressAsync)

// gdeflate.h
WWR_LINK_CHECK(nvcompBatchedGdeflateCompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedGdeflateCompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedGdeflateCompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedGdeflateCompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(nvcompBatchedGdeflateCompressAsync)
WWR_LINK_CHECK(nvcompBatchedGdeflateDecompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedGdeflateDecompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedGdeflateDecompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedGdeflateGetDecompressSizeAsync)
WWR_LINK_CHECK(nvcompBatchedGdeflateDecompressAsync)

// bitcomp.h
WWR_LINK_CHECK(nvcompBatchedBitcompCompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedBitcompCompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedBitcompCompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedBitcompCompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(nvcompBatchedBitcompCompressAsync)
WWR_LINK_CHECK(nvcompBatchedBitcompDecompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedBitcompDecompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedBitcompDecompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedBitcompGetDecompressSizeAsync)
WWR_LINK_CHECK(nvcompBatchedBitcompDecompressAsync)

// ans.h
WWR_LINK_CHECK(nvcompBatchedANSCompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedANSCompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedANSCompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedANSCompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(nvcompBatchedANSCompressAsync)
WWR_LINK_CHECK(nvcompBatchedANSDecompressGetRequiredAlignments)
WWR_LINK_CHECK(nvcompBatchedANSDecompressGetTempSizeAsync)
WWR_LINK_CHECK(nvcompBatchedANSDecompressGetTempSizeSync)
WWR_LINK_CHECK(nvcompBatchedANSGetDecompressSizeAsync)
WWR_LINK_CHECK(nvcompBatchedANSDecompressAsync)

// crc32.h
WWR_LINK_CHECK(nvcompBatchedCRC32Async)
WWR_LINK_CHECK(nvcompBatchedCRC32GetHeuristicConf)
WWR_LINK_CHECK(nvcompBatchedCRC32SearchConf)

} // namespace wwr::cuda::test
