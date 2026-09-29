// hipcomp.cppm - Compile-time tests for wwr.hip.hipcomp

module;

#include "test/shared/link_check.h"

export module wwr.test.hip.hipcomp;

import std;
import wwr.hip.hipcomp;

// ========================================================================
// Compile-time tests for wwr.hip.hipcomp
//
// Verifies at compile-time that:
//   - Enum types satisfy std::is_enum_v
//   - Enumerator values match the hipCOMP-specified integers (matching NAMES
//     never guarantee matching VALUES across backends -- cf. WWRRAND_RNG_*).
//     hipcompStatus_t skips the 13..24 checksum/buffer codes nvCOMP 5.3 added.
// Link-time checks (WWR_LINK_CHECK) verify that every re-exported batched-LLIF
// function symbol resolves against libhipcomp at link time. The re-exported
// default-option structs are inline definitions inside the module, not external
// library symbols, so they are not link-checked.
// ========================================================================

namespace wwr::hip::test {

using namespace wwr::hip;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<hipcompStatus_t>);
static_assert(std::is_enum_v<hipcompType_t>);
static_assert(std::is_enum_v<hipcompANSType_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipcompStatus_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(hipcompSuccess) == 0);
static_assert(static_cast<int>(hipcompErrorInvalidValue) == 10);
static_assert(static_cast<int>(hipcompErrorNotSupported) == 11);
static_assert(static_cast<int>(hipcompErrorCannotDecompress) == 12);
static_assert(static_cast<int>(hipcompErrorCudaError) == 1000);
static_assert(static_cast<int>(hipcompErrorInternal) == 10000);

// ────────────────────────────────────────────────────────────────────────
// Enum values: hipcompType_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(HIPCOMP_TYPE_CHAR) == 0);
static_assert(static_cast<int>(HIPCOMP_TYPE_UCHAR) == 1);
static_assert(static_cast<int>(HIPCOMP_TYPE_SHORT) == 2);
static_assert(static_cast<int>(HIPCOMP_TYPE_USHORT) == 3);
static_assert(static_cast<int>(HIPCOMP_TYPE_INT) == 4);
static_assert(static_cast<int>(HIPCOMP_TYPE_UINT) == 5);
static_assert(static_cast<int>(HIPCOMP_TYPE_LONGLONG) == 6);
static_assert(static_cast<int>(HIPCOMP_TYPE_ULONGLONG) == 7);
static_assert(static_cast<int>(HIPCOMP_TYPE_BITS) == 0xff);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution: batched LLIF + generic decompress
// ────────────────────────────────────────────────────────────────────────

// hipcomp.h
WWR_LINK_CHECK(hipcompDecompressGetMetadata)
WWR_LINK_CHECK(hipcompDecompressDestroyMetadata)
WWR_LINK_CHECK(hipcompDecompressGetTempSize)
WWR_LINK_CHECK(hipcompDecompressGetOutputSize)
WWR_LINK_CHECK(hipcompDecompressGetType)
WWR_LINK_CHECK(hipcompDecompressAsync)

// lz4.h
WWR_LINK_CHECK(hipcompBatchedLZ4CompressGetTempSize)
WWR_LINK_CHECK(hipcompBatchedLZ4CompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(hipcompBatchedLZ4CompressAsync)
WWR_LINK_CHECK(hipcompBatchedLZ4DecompressGetTempSize)
WWR_LINK_CHECK(hipcompBatchedLZ4DecompressAsync)
WWR_LINK_CHECK(hipcompBatchedLZ4GetDecompressSizeAsync)

// snappy.h
WWR_LINK_CHECK(hipcompBatchedSnappyDecompressGetTempSize)
WWR_LINK_CHECK(hipcompBatchedSnappyGetDecompressSizeAsync)
WWR_LINK_CHECK(hipcompBatchedSnappyDecompressAsync)
WWR_LINK_CHECK(hipcompBatchedSnappyCompressGetTempSize)
WWR_LINK_CHECK(hipcompBatchedSnappyCompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(hipcompBatchedSnappyCompressAsync)

// cascaded.h
WWR_LINK_CHECK(hipcompBatchedCascadedCompressGetTempSize)
WWR_LINK_CHECK(hipcompBatchedCascadedCompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(hipcompBatchedCascadedCompressAsync)
WWR_LINK_CHECK(hipcompBatchedCascadedDecompressGetTempSize)
WWR_LINK_CHECK(hipcompBatchedCascadedDecompressAsync)
WWR_LINK_CHECK(hipcompBatchedCascadedGetDecompressSizeAsync)

// gdeflate.h
WWR_LINK_CHECK(hipcompBatchedGdeflateCompressGetTempSize)
WWR_LINK_CHECK(hipcompBatchedGdeflateCompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(hipcompBatchedGdeflateCompressAsync)
WWR_LINK_CHECK(hipcompBatchedGdeflateDecompressGetTempSize)
WWR_LINK_CHECK(hipcompBatchedGdeflateDecompressAsync)
WWR_LINK_CHECK(hipcompBatchedGdeflateGetDecompressSizeAsync)

// bitcomp.h
WWR_LINK_CHECK(hipcompBitcompCompressConfigure)
WWR_LINK_CHECK(hipcompBitcompCompressAsync)
WWR_LINK_CHECK(hipcompBitcompDecompressConfigure)
WWR_LINK_CHECK(hipcompBitcompDestroyMetadata)
WWR_LINK_CHECK(hipcompBitcompDecompressAsync)
WWR_LINK_CHECK(hipcompIsBitcompData)
WWR_LINK_CHECK(hipcompBatchedBitcompCompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(hipcompBatchedBitcompCompressAsync)
WWR_LINK_CHECK(hipcompBatchedBitcompDecompressAsync)
WWR_LINK_CHECK(hipcompBatchedBitcompGetDecompressSizeAsync)
WWR_LINK_CHECK(hipcompBatchedBitcompCompressGetTempSize)
WWR_LINK_CHECK(hipcompBatchedBitcompDecompressGetTempSize)

// ans.h
WWR_LINK_CHECK(hipcompBatchedANSCompressGetTempSize)
WWR_LINK_CHECK(hipcompBatchedANSCompressGetMaxOutputChunkSize)
WWR_LINK_CHECK(hipcompBatchedANSCompressAsync)
WWR_LINK_CHECK(hipcompBatchedANSDecompressGetTempSize)
WWR_LINK_CHECK(hipcompBatchedANSGetDecompressSizeAsync)
WWR_LINK_CHECK(hipcompBatchedANSDecompressAsync)

} // namespace wwr::hip::test
