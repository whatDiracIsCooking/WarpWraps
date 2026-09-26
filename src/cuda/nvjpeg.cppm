/**
 * @file nvjpeg.cppm
 * @brief Primary interface for gpumod.cuda.nvjpeg
 *
 * This module wraps the nvJPEG API and exports types, constants,
 * and functions for GPU-accelerated JPEG encoding and decoding.
 *
 * Usage:
 *   import gpumod.cuda.nvjpeg;
 */

module;

#include <nvjpeg.h>

export module gpumod.cuda.nvjpeg;

import std;

// ========================================================================
// Export all nvJPEG types and functions in gpumod namespace
// ========================================================================

export namespace gpumod::cuda {

// ========================================================================
// Status Codes
// ========================================================================
using ::nvjpegStatus_t;

using ::NVJPEG_STATUS_ALLOCATOR_FAILURE;
using ::NVJPEG_STATUS_ARCH_MISMATCH;
using ::NVJPEG_STATUS_BAD_JPEG;
using ::NVJPEG_STATUS_EXECUTION_FAILED;
using ::NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
using ::NVJPEG_STATUS_INCOMPLETE_BITSTREAM;
using ::NVJPEG_STATUS_INTERNAL_ERROR;
using ::NVJPEG_STATUS_INVALID_PARAMETER;
using ::NVJPEG_STATUS_JPEG_NOT_SUPPORTED;
using ::NVJPEG_STATUS_NOT_INITIALIZED;
using ::NVJPEG_STATUS_SUCCESS;

// ========================================================================
// Enumerations - EXIF Orientation
// ========================================================================
using ::nvjpegExifOrientation_t;

using ::NVJPEG_ORIENTATION_FLIP_HORIZONTAL;
using ::NVJPEG_ORIENTATION_FLIP_VERTICAL;
using ::NVJPEG_ORIENTATION_NORMAL;
using ::NVJPEG_ORIENTATION_ROTATE_180;
using ::NVJPEG_ORIENTATION_ROTATE_270;
using ::NVJPEG_ORIENTATION_ROTATE_90;
using ::NVJPEG_ORIENTATION_TRANSPOSE;
using ::NVJPEG_ORIENTATION_TRANSVERSE;
using ::NVJPEG_ORIENTATION_UNKNOWN;

// ========================================================================
// Enumerations - Chroma Subsampling
// ========================================================================
using ::nvjpegChromaSubsampling_t;

using ::NVJPEG_CSS_410;
using ::NVJPEG_CSS_410V;
using ::NVJPEG_CSS_411;
using ::NVJPEG_CSS_420;
using ::NVJPEG_CSS_422;
using ::NVJPEG_CSS_440;
using ::NVJPEG_CSS_444;
using ::NVJPEG_CSS_GRAY;
using ::NVJPEG_CSS_UNKNOWN;

// ========================================================================
// Enumerations - Output Format
// ========================================================================
using ::nvjpegOutputFormat_t;

using ::NVJPEG_OUTPUT_BGR;
using ::NVJPEG_OUTPUT_BGRI;
using ::NVJPEG_OUTPUT_FORMAT_MAX;
using ::NVJPEG_OUTPUT_NV12;
using ::NVJPEG_OUTPUT_RGB;
using ::NVJPEG_OUTPUT_RGBI;
using ::NVJPEG_OUTPUT_UNCHANGED;
using ::NVJPEG_OUTPUT_UNCHANGEDI_U16;
using ::NVJPEG_OUTPUT_Y;
using ::NVJPEG_OUTPUT_YUV;
using ::NVJPEG_OUTPUT_YUY2;

// ========================================================================
// Enumerations - Input Format
// ========================================================================
using ::nvjpegInputFormat_t;

using ::NVJPEG_INPUT_BGR;
using ::NVJPEG_INPUT_BGRI;
using ::NVJPEG_INPUT_NV12;
using ::NVJPEG_INPUT_RGB;
using ::NVJPEG_INPUT_RGBI;
using ::NVJPEG_INPUT_YUV;

// ========================================================================
// Enumerations - Backend
// ========================================================================
using ::nvjpegBackend_t;

using ::NVJPEG_BACKEND_DEFAULT;
using ::NVJPEG_BACKEND_GPU_HYBRID;
using ::NVJPEG_BACKEND_GPU_HYBRID_DEVICE;
using ::NVJPEG_BACKEND_HARDWARE;
using ::NVJPEG_BACKEND_HARDWARE_DEVICE;
using ::NVJPEG_BACKEND_HYBRID;
using ::NVJPEG_BACKEND_LOSSLESS_JPEG;

// ========================================================================
// Enumerations - Encoder Backend
// ========================================================================
using ::nvjpegEncBackend_t;

using ::NVJPEG_ENC_BACKEND_DEFAULT;
using ::NVJPEG_ENC_BACKEND_GPU;
using ::NVJPEG_ENC_BACKEND_HARDWARE;

// ========================================================================
// Enumerations - JPEG Encoding Type
// ========================================================================
using ::nvjpegJpegEncoding_t;

using ::NVJPEG_ENCODING_BASELINE_DCT;
using ::NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN;
using ::NVJPEG_ENCODING_LOSSLESS_HUFFMAN;
using ::NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN;
using ::NVJPEG_ENCODING_UNKNOWN;

// ========================================================================
// Enumerations - Scale Factor
// ========================================================================
using ::nvjpegScaleFactor_t;

using ::NVJPEG_SCALE_1_BY_2;
using ::NVJPEG_SCALE_1_BY_4;
using ::NVJPEG_SCALE_1_BY_8;
using ::NVJPEG_SCALE_NONE;

// ========================================================================
// Structs and Image Descriptor
// ========================================================================
using ::nvjpegImage_t;

// Memory allocator callback types
using ::tDevFree;
using ::tDevFreeV2;
using ::tDevMalloc;
using ::tDevMallocV2;
using ::tPinnedFree;
using ::tPinnedFreeV2;
using ::tPinnedMalloc;
using ::tPinnedMallocV2;

// Allocator structs
using ::nvjpegDevAllocator_t;
using ::nvjpegDevAllocatorV2_t;
using ::nvjpegPinnedAllocator_t;
using ::nvjpegPinnedAllocatorV2_t;

// ========================================================================
// Opaque Handle Types
// ========================================================================
using ::nvjpegHandle;
using ::nvjpegHandle_t;

using ::nvjpegJpegState;
using ::nvjpegJpegState_t;

using ::nvjpegEncoderState;
using ::nvjpegEncoderState_t;

using ::nvjpegEncoderParams;
using ::nvjpegEncoderParams_t;

using ::nvjpegBufferPinned;
using ::nvjpegBufferPinned_t;

using ::nvjpegBufferDevice;
using ::nvjpegBufferDevice_t;

using ::nvjpegJpegStream;
using ::nvjpegJpegStream_t;

using ::nvjpegDecodeParams;
using ::nvjpegDecodeParams_t;

using ::nvjpegJpegDecoder;
using ::nvjpegJpegDecoder_t;

// ========================================================================
// Library Property and Version Functions
// ========================================================================
using ::nvjpegGetCudartProperty;
using ::nvjpegGetProperty;

// ========================================================================
// Library Handle Management
// ========================================================================
using ::nvjpegCreate;
using ::nvjpegCreateEx;
using ::nvjpegCreateExV2;
using ::nvjpegCreateSimple;
using ::nvjpegDestroy;

// ========================================================================
// Memory Padding Configuration
// ========================================================================
using ::nvjpegGetDeviceMemoryPadding;
using ::nvjpegGetPinnedMemoryPadding;
using ::nvjpegSetDeviceMemoryPadding;
using ::nvjpegSetPinnedMemoryPadding;

// ========================================================================
// Hardware Info
// ========================================================================
using ::nvjpegGetHardwareDecoderInfo;
using ::nvjpegGetHardwareEncoderInfo;

// ========================================================================
// Decoder State Management
// ========================================================================
using ::nvjpegJpegStateCreate;
using ::nvjpegJpegStateDestroy;

// ========================================================================
// Image Info Query
// ========================================================================
using ::nvjpegGetImageInfo;

// ========================================================================
// Simple Decode Functions
// ========================================================================
using ::nvjpegDecode;

// ========================================================================
// Batch Decoding Functions
// ========================================================================
using ::nvjpegDecodeBatched;
using ::nvjpegDecodeBatchedEx;
using ::nvjpegDecodeBatchedInitialize;
using ::nvjpegDecodeBatchedParseJpegTables;
using ::nvjpegDecodeBatchedPreAllocate;
using ::nvjpegDecodeBatchedSupported;
using ::nvjpegDecodeBatchedSupportedEx;

// ========================================================================
// Encoder State and Params Management
// ========================================================================
using ::nvjpegEncoderParamsCreate;
using ::nvjpegEncoderParamsDestroy;
using ::nvjpegEncoderParamsSetEncoding;
using ::nvjpegEncoderParamsSetOptimizedHuffman;
using ::nvjpegEncoderParamsSetQuality;
using ::nvjpegEncoderParamsSetRestartInterval;
using ::nvjpegEncoderParamsSetSamplingFactors;
using ::nvjpegEncoderStateCreate;
using ::nvjpegEncoderStateCreateWithBackend;
using ::nvjpegEncoderStateDestroy;

// ========================================================================
// Encode Functions
// ========================================================================
using ::nvjpegEncode;
using ::nvjpegEncodeGetBufferSize;
using ::nvjpegEncodeImage;
using ::nvjpegEncodeRetrieveBitstream;
using ::nvjpegEncodeRetrieveBitstreamDevice;
using ::nvjpegEncodeYUV;

// ========================================================================
// Buffer Management (API v2)
// ========================================================================
using ::nvjpegBufferPinnedCreate;
using ::nvjpegBufferPinnedCreateV2;
using ::nvjpegBufferPinnedDestroy;
using ::nvjpegBufferPinnedResize;
using ::nvjpegBufferPinnedRetrieve;

using ::nvjpegBufferDeviceCreate;
using ::nvjpegBufferDeviceCreateV2;
using ::nvjpegBufferDeviceDestroy;
using ::nvjpegBufferDeviceResize;
using ::nvjpegBufferDeviceRetrieve;

// Buffer attachment to decoder state
using ::nvjpegStateAttachDeviceBuffer;
using ::nvjpegStateAttachPinnedBuffer;

// ========================================================================
// JPEG Stream (Header / Metadata Parsing)
// ========================================================================
using ::nvjpegJpegStreamCreate;
using ::nvjpegJpegStreamDestroy;
using ::nvjpegJpegStreamGetChromaSubsampling;
using ::nvjpegJpegStreamGetComponentDimensions;
using ::nvjpegJpegStreamGetComponentsNum;
using ::nvjpegJpegStreamGetExifOrientation;
using ::nvjpegJpegStreamGetFrameDimensions;
using ::nvjpegJpegStreamGetJpegEncoding;
using ::nvjpegJpegStreamGetSamplePrecision;
using ::nvjpegJpegStreamParse;
using ::nvjpegJpegStreamParseHeader;
using ::nvjpegJpegStreamParseTables;

// ========================================================================
// Decode Params Management
// ========================================================================
using ::nvjpegDecodeParamsCreate;
using ::nvjpegDecodeParamsDestroy;
using ::nvjpegDecodeParamsSetAllowCMYK;
using ::nvjpegDecodeParamsSetExifOrientation;
using ::nvjpegDecodeParamsSetOutputFormat;
using ::nvjpegDecodeParamsSetROI;
using ::nvjpegDecodeParamsSetScaleFactor;

// ========================================================================
// Advanced Decoder Lifecycle (API v2)
// ========================================================================
using ::nvjpegDecoderCreate;
using ::nvjpegDecoderDestroy;
using ::nvjpegDecoderJpegSupported;
using ::nvjpegDecoderStateCreate;

// ========================================================================
// Advanced Decode Functions (API v2)
// ========================================================================
using ::nvjpegDecodeJpeg;
using ::nvjpegDecodeJpegDevice;
using ::nvjpegDecodeJpegHost;
using ::nvjpegDecodeJpegTransferToDevice;

// ========================================================================
// Transcoding Helper Functions
// ========================================================================
using ::nvjpegEncoderParamsCopyMetadata;
using ::nvjpegEncoderParamsCopyQuantizationTables;

} // namespace gpumod::cuda
