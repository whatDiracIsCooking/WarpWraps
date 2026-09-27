// nvjpeg.cppm - Compile-time tests for wwr.cuda.nvjpeg

module;

#include "test/shared/link_check.h"

export module wwr.test.cuda.nvjpeg;

import std;
import wwr.cuda.nvjpeg;

// ========================================================================
// Compile-time tests for wwr.cuda.nvjpeg
//
// Verifies at compile-time that:
//   - Enum types satisfy std::is_enum_v
//   - Key enumerator values match the nvJPEG-specified integer values
//   - Opaque handle types are pointer types (std::is_pointer_v)
// Link-time checks (WWR_LINK_CHECK) verify that every re-exported function
// symbol resolves at link time.
// ========================================================================

namespace wwr::cuda::test {

using namespace wwr::cuda;

// ────────────────────────────────────────────────────────────────────────
// Enum type checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_enum_v<nvjpegStatus_t>);
static_assert(std::is_enum_v<nvjpegExifOrientation_t>);
static_assert(std::is_enum_v<nvjpegChromaSubsampling_t>);
static_assert(std::is_enum_v<nvjpegOutputFormat_t>);
static_assert(std::is_enum_v<nvjpegInputFormat_t>);
static_assert(std::is_enum_v<nvjpegBackend_t>);
static_assert(std::is_enum_v<nvjpegEncBackend_t>);
static_assert(std::is_enum_v<nvjpegJpegEncoding_t>);
static_assert(std::is_enum_v<nvjpegScaleFactor_t>);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvjpegStatus_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVJPEG_STATUS_SUCCESS) == 0);
static_assert(static_cast<int>(NVJPEG_STATUS_NOT_INITIALIZED) == 1);
static_assert(static_cast<int>(NVJPEG_STATUS_INVALID_PARAMETER) == 2);
static_assert(static_cast<int>(NVJPEG_STATUS_BAD_JPEG) == 3);
static_assert(static_cast<int>(NVJPEG_STATUS_JPEG_NOT_SUPPORTED) == 4);
static_assert(static_cast<int>(NVJPEG_STATUS_ALLOCATOR_FAILURE) == 5);
static_assert(static_cast<int>(NVJPEG_STATUS_EXECUTION_FAILED) == 6);
static_assert(static_cast<int>(NVJPEG_STATUS_ARCH_MISMATCH) == 7);
static_assert(static_cast<int>(NVJPEG_STATUS_INTERNAL_ERROR) == 8);
static_assert(static_cast<int>(NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED) == 9);
static_assert(static_cast<int>(NVJPEG_STATUS_INCOMPLETE_BITSTREAM) == 10);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvjpegExifOrientation_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVJPEG_ORIENTATION_UNKNOWN) == 0);
static_assert(static_cast<int>(NVJPEG_ORIENTATION_NORMAL) == 1);
static_assert(static_cast<int>(NVJPEG_ORIENTATION_FLIP_HORIZONTAL) == 2);
static_assert(static_cast<int>(NVJPEG_ORIENTATION_ROTATE_180) == 3);
static_assert(static_cast<int>(NVJPEG_ORIENTATION_FLIP_VERTICAL) == 4);
static_assert(static_cast<int>(NVJPEG_ORIENTATION_TRANSPOSE) == 5);
static_assert(static_cast<int>(NVJPEG_ORIENTATION_ROTATE_90) == 6);
static_assert(static_cast<int>(NVJPEG_ORIENTATION_TRANSVERSE) == 7);
static_assert(static_cast<int>(NVJPEG_ORIENTATION_ROTATE_270) == 8);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvjpegChromaSubsampling_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVJPEG_CSS_444) == 0);
static_assert(static_cast<int>(NVJPEG_CSS_422) == 1);
static_assert(static_cast<int>(NVJPEG_CSS_420) == 2);
static_assert(static_cast<int>(NVJPEG_CSS_440) == 3);
static_assert(static_cast<int>(NVJPEG_CSS_411) == 4);
static_assert(static_cast<int>(NVJPEG_CSS_410) == 5);
static_assert(static_cast<int>(NVJPEG_CSS_GRAY) == 6);
static_assert(static_cast<int>(NVJPEG_CSS_410V) == 7);
static_assert(static_cast<int>(NVJPEG_CSS_UNKNOWN) == -1);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvjpegOutputFormat_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVJPEG_OUTPUT_UNCHANGED) == 0);
static_assert(static_cast<int>(NVJPEG_OUTPUT_YUV) == 1);
static_assert(static_cast<int>(NVJPEG_OUTPUT_Y) == 2);
static_assert(static_cast<int>(NVJPEG_OUTPUT_RGB) == 3);
static_assert(static_cast<int>(NVJPEG_OUTPUT_BGR) == 4);
static_assert(static_cast<int>(NVJPEG_OUTPUT_RGBI) == 5);
static_assert(static_cast<int>(NVJPEG_OUTPUT_BGRI) == 6);
static_assert(static_cast<int>(NVJPEG_OUTPUT_UNCHANGEDI_U16) == 7);
static_assert(static_cast<int>(NVJPEG_OUTPUT_NV12) == 8);
static_assert(static_cast<int>(NVJPEG_OUTPUT_YUY2) == 9);
static_assert(static_cast<int>(NVJPEG_OUTPUT_FORMAT_MAX) == 9);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvjpegInputFormat_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVJPEG_INPUT_YUV) == 1);
static_assert(static_cast<int>(NVJPEG_INPUT_RGB) == 3);
static_assert(static_cast<int>(NVJPEG_INPUT_BGR) == 4);
static_assert(static_cast<int>(NVJPEG_INPUT_RGBI) == 5);
static_assert(static_cast<int>(NVJPEG_INPUT_BGRI) == 6);
static_assert(static_cast<int>(NVJPEG_INPUT_NV12) == 8);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvjpegBackend_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVJPEG_BACKEND_DEFAULT) == 0);
static_assert(static_cast<int>(NVJPEG_BACKEND_HYBRID) == 1);
static_assert(static_cast<int>(NVJPEG_BACKEND_GPU_HYBRID) == 2);
static_assert(static_cast<int>(NVJPEG_BACKEND_HARDWARE) == 3);
static_assert(static_cast<int>(NVJPEG_BACKEND_GPU_HYBRID_DEVICE) == 4);
static_assert(static_cast<int>(NVJPEG_BACKEND_HARDWARE_DEVICE) == 5);
static_assert(static_cast<int>(NVJPEG_BACKEND_LOSSLESS_JPEG) == 6);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvjpegEncBackend_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVJPEG_ENC_BACKEND_DEFAULT) == 0);
static_assert(static_cast<int>(NVJPEG_ENC_BACKEND_GPU) == 1);
static_assert(static_cast<int>(NVJPEG_ENC_BACKEND_HARDWARE) == 2);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvjpegJpegEncoding_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVJPEG_ENCODING_UNKNOWN) == 0x00);
static_assert(static_cast<int>(NVJPEG_ENCODING_BASELINE_DCT) == 0xc0);
static_assert(static_cast<int>(NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN) == 0xc1);
static_assert(static_cast<int>(NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN) == 0xc2);
static_assert(static_cast<int>(NVJPEG_ENCODING_LOSSLESS_HUFFMAN) == 0xc3);

// ────────────────────────────────────────────────────────────────────────
// Enum values: nvjpegScaleFactor_t
// ────────────────────────────────────────────────────────────────────────

static_assert(static_cast<int>(NVJPEG_SCALE_NONE) == 0);
static_assert(static_cast<int>(NVJPEG_SCALE_1_BY_2) == 1);
static_assert(static_cast<int>(NVJPEG_SCALE_1_BY_4) == 2);
static_assert(static_cast<int>(NVJPEG_SCALE_1_BY_8) == 3);

// ────────────────────────────────────────────────────────────────────────
// Opaque handle pointer checks
// ────────────────────────────────────────────────────────────────────────

static_assert(std::is_pointer_v<nvjpegHandle_t>);
static_assert(std::is_pointer_v<nvjpegJpegState_t>);
static_assert(std::is_pointer_v<nvjpegEncoderState_t>);
static_assert(std::is_pointer_v<nvjpegEncoderParams_t>);
static_assert(std::is_pointer_v<nvjpegBufferPinned_t>);
static_assert(std::is_pointer_v<nvjpegBufferDevice_t>);
static_assert(std::is_pointer_v<nvjpegJpegStream_t>);
static_assert(std::is_pointer_v<nvjpegDecodeParams_t>);
static_assert(std::is_pointer_v<nvjpegJpegDecoder_t>);

// ────────────────────────────────────────────────────────────────────────
// Link-time symbol resolution
// Forces the linker to resolve every re-exported NVJPEGAPI function symbol.
// ────────────────────────────────────────────────────────────────────────

// Library property and version
WWR_LINK_CHECK(nvjpegGetProperty)
WWR_LINK_CHECK(nvjpegGetCudartProperty)

// Library handle management
WWR_LINK_CHECK(nvjpegCreate)
WWR_LINK_CHECK(nvjpegCreateSimple)
WWR_LINK_CHECK(nvjpegCreateEx)
WWR_LINK_CHECK(nvjpegCreateExV2)
WWR_LINK_CHECK(nvjpegDestroy)

// Memory padding configuration
WWR_LINK_CHECK(nvjpegSetDeviceMemoryPadding)
WWR_LINK_CHECK(nvjpegGetDeviceMemoryPadding)
WWR_LINK_CHECK(nvjpegSetPinnedMemoryPadding)
WWR_LINK_CHECK(nvjpegGetPinnedMemoryPadding)

// Hardware info
WWR_LINK_CHECK(nvjpegGetHardwareDecoderInfo)
WWR_LINK_CHECK(nvjpegGetHardwareEncoderInfo)

// Decoder state management
WWR_LINK_CHECK(nvjpegJpegStateCreate)
WWR_LINK_CHECK(nvjpegJpegStateDestroy)

// Image info query
WWR_LINK_CHECK(nvjpegGetImageInfo)

// Simple decode
WWR_LINK_CHECK(nvjpegDecode)

// Batch decoding
WWR_LINK_CHECK(nvjpegDecodeBatchedInitialize)
WWR_LINK_CHECK(nvjpegDecodeBatched)
WWR_LINK_CHECK(nvjpegDecodeBatchedPreAllocate)
WWR_LINK_CHECK(nvjpegDecodeBatchedParseJpegTables)
WWR_LINK_CHECK(nvjpegDecodeBatchedSupported)
WWR_LINK_CHECK(nvjpegDecodeBatchedSupportedEx)
WWR_LINK_CHECK(nvjpegDecodeBatchedEx)

// Encoder state and params management
WWR_LINK_CHECK(nvjpegEncoderStateCreate)
WWR_LINK_CHECK(nvjpegEncoderStateCreateWithBackend)
WWR_LINK_CHECK(nvjpegEncoderStateDestroy)
WWR_LINK_CHECK(nvjpegEncoderParamsCreate)
WWR_LINK_CHECK(nvjpegEncoderParamsDestroy)
WWR_LINK_CHECK(nvjpegEncoderParamsSetQuality)
WWR_LINK_CHECK(nvjpegEncoderParamsSetEncoding)
WWR_LINK_CHECK(nvjpegEncoderParamsSetOptimizedHuffman)
WWR_LINK_CHECK(nvjpegEncoderParamsSetSamplingFactors)
WWR_LINK_CHECK(nvjpegEncoderParamsSetRestartInterval)

// Encode functions
WWR_LINK_CHECK(nvjpegEncodeGetBufferSize)
WWR_LINK_CHECK(nvjpegEncodeYUV)
WWR_LINK_CHECK(nvjpegEncodeImage)
WWR_LINK_CHECK(nvjpegEncode)
WWR_LINK_CHECK(nvjpegEncodeRetrieveBitstreamDevice)
WWR_LINK_CHECK(nvjpegEncodeRetrieveBitstream)

// Buffer management (API v2)
WWR_LINK_CHECK(nvjpegBufferPinnedCreate)
WWR_LINK_CHECK(nvjpegBufferPinnedCreateV2)
WWR_LINK_CHECK(nvjpegBufferPinnedResize)
WWR_LINK_CHECK(nvjpegBufferPinnedDestroy)
WWR_LINK_CHECK(nvjpegBufferPinnedRetrieve)
WWR_LINK_CHECK(nvjpegBufferDeviceCreate)
WWR_LINK_CHECK(nvjpegBufferDeviceCreateV2)
WWR_LINK_CHECK(nvjpegBufferDeviceResize)
WWR_LINK_CHECK(nvjpegBufferDeviceDestroy)
WWR_LINK_CHECK(nvjpegBufferDeviceRetrieve)
WWR_LINK_CHECK(nvjpegStateAttachPinnedBuffer)
WWR_LINK_CHECK(nvjpegStateAttachDeviceBuffer)

// JPEG stream (header / metadata parsing)
WWR_LINK_CHECK(nvjpegJpegStreamCreate)
WWR_LINK_CHECK(nvjpegJpegStreamDestroy)
WWR_LINK_CHECK(nvjpegJpegStreamParse)
WWR_LINK_CHECK(nvjpegJpegStreamParseHeader)
WWR_LINK_CHECK(nvjpegJpegStreamParseTables)
WWR_LINK_CHECK(nvjpegJpegStreamGetJpegEncoding)
WWR_LINK_CHECK(nvjpegJpegStreamGetFrameDimensions)
WWR_LINK_CHECK(nvjpegJpegStreamGetComponentsNum)
WWR_LINK_CHECK(nvjpegJpegStreamGetComponentDimensions)
WWR_LINK_CHECK(nvjpegJpegStreamGetExifOrientation)
WWR_LINK_CHECK(nvjpegJpegStreamGetSamplePrecision)
WWR_LINK_CHECK(nvjpegJpegStreamGetChromaSubsampling)

// Decode params management
WWR_LINK_CHECK(nvjpegDecodeParamsCreate)
WWR_LINK_CHECK(nvjpegDecodeParamsDestroy)
WWR_LINK_CHECK(nvjpegDecodeParamsSetOutputFormat)
WWR_LINK_CHECK(nvjpegDecodeParamsSetROI)
WWR_LINK_CHECK(nvjpegDecodeParamsSetAllowCMYK)
WWR_LINK_CHECK(nvjpegDecodeParamsSetScaleFactor)
WWR_LINK_CHECK(nvjpegDecodeParamsSetExifOrientation)

// Advanced decoder lifecycle (API v2)
WWR_LINK_CHECK(nvjpegDecoderCreate)
WWR_LINK_CHECK(nvjpegDecoderDestroy)
WWR_LINK_CHECK(nvjpegDecoderJpegSupported)
WWR_LINK_CHECK(nvjpegDecoderStateCreate)

// Advanced decode functions (API v2)
WWR_LINK_CHECK(nvjpegDecodeJpeg)
WWR_LINK_CHECK(nvjpegDecodeJpegHost)
WWR_LINK_CHECK(nvjpegDecodeJpegTransferToDevice)
WWR_LINK_CHECK(nvjpegDecodeJpegDevice)

// Note: nvjpegEncoderParamsCopyMetadata and nvjpegEncoderParamsCopyQuantizationTables
// are declared without NVJPEGAPI in the header (transcoding helpers); WWR_LINK_CHECK is
// intentionally omitted for those two symbols.

} // namespace wwr::cuda::test
