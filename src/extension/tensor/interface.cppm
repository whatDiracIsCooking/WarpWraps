/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.tensor
 *
 * The error-handling and RAII layer for GPU tensor contractions (cuTENSOR or
 * hipTensor, per WWR_GPU_BACKEND). cuTENSOR/hipTensor are descriptor- and
 * runtime-datatype-driven, so there is no layer-2 generic execution wrapper here
 * (unlike wwr.wrappers.fft): the convenience this layer adds is RAII over the
 * vendor's hand-paired create/destroy calls, plus an error policy for the vendor
 * status type. It aggregates:
 * - :tensor_error  - Error code specializations for wwrtensorStatus_t
 * - :tensor_handle - RAII wrapper for the tensor library handle
 * - :tensor_plan   - RAII wrapper for a tensor descriptor (see that header for
 *                    why a descriptor rather than a full contraction plan)
 *
 * Usage:
 *   import wwr.extension.tensor;
 *   using namespace wwr::extension;
 */

export module wwr.extension.tensor;

import std;

// Re-export the vendor tensor module: wwrtensorHandle_t / wwrtensorStatus_t /
// wwrtensorTensorDescriptor_t are the wrappers' handle and error types, so a
// consumer can name them without importing wwr.tensor separately.
export import wwr.tensor;
export import :tensor_error;
export import :tensor_handle;
export import :tensor_plan;
