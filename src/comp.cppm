/**
 * @file comp.cppm
 * @brief Backend-neutral batched compression: wwrcomp* over nvCOMP / hipCOMP
 *
 * The one vendor pair where the two backends are version-skewed rather than
 * prefix-skewed: nvCOMP is 5.3, hipCOMP is a hipify of nvCOMP branch-2.2, so the
 * batched low-level signatures diverge. This layer carries option (a) of issue
 * #110 -- the measured 2.2 intersection -- for the three algorithms both
 * backends implement in both directions: LZ4, Snappy and Cascaded. (Bitcomp,
 * ANS and GDeflate are NVIDIA-proprietary schemes hipCOMP ships as headers
 * without device support, so they are reachable only through the raw vendor
 * modules, never portably.)
 *
 * Each wwrcompBatched<Algo>* function is written once against the hipCOMP 2.2
 * shape and forwards to whichever backend was selected. Reconciling the skew
 * costs three things on the CUDA path, all invisible to the neutral caller:
 *   - the split Sync/Async temp-size queries collapse to the Async (upper-bound)
 *     form; the exact Sync query is unreachable here;
 *   - compress device_statuses is passed NULL (nvCOMP documents this as "error
 *     status not reported"); hipCOMP 2.2 has no such parameter;
 *   - the nvCOMP-only decompress backend / bitshuffle options are defaulted.
 * A caller needing any of those -- or Zstd/GZIP/Deflate/CRC32, or the hardware
 * decompression engine -- uses wwr.cuda.nvcomp directly. This layer is for
 * backend-portable code that accepts the 2.2 ceiling. hipCOMP self-describes as
 * an early-access preview (every algorithm experimental, not performance-tuned),
 * so the HIP path is a portability contract, not a production guarantee.
 *
 * Opts carry only the fields both backends share: LZ4 its data_type, Snappy
 * nothing, Cascaded its full tuning (chunk_size / type / num_RLEs / num_deltas /
 * use_bp -- these survive field-for-field into nvCOMP 5.3). The neutral opts are
 * plain structs, distinct from either vendor's; the forwarders build the backend
 * struct from them.
 *
 * Enumerator VALUES are pinned in test/gpu/comp.cppm, one line per name: matching
 * names never guarantee matching values (this bit WWRRAND_RNG_* -- see
 * rand.cppm). Here they happen to agree, which the pins lock in.
 *
 * Usage:
 *   import wwr.comp;
 *
 *   wwrcompBatchedLZ4Opts opts{WWRCOMP_TYPE_CHAR};
 *   wwrcompBatchedLZ4CompressGetTempSize(batch, max_chunk, opts, &temp_bytes);
 *   wwrcompBatchedLZ4CompressAsync(unc_ptrs, unc_bytes, max_chunk, batch,
 *       temp, temp_bytes, comp_ptrs, comp_bytes, opts, stream);
 */

module;

#include <cstddef>

#include "backend.h"

export module wwr.comp;

#if defined(WWR_GPU_BACKEND_CUDA)
import wwr.cuda.nvcomp;
#else
import wwr.hip.hipcomp;
#endif

export namespace wwr {

// ========================================================================
// Types
// ========================================================================
WWR_TYPE(wwrcompStatus_t, nvcompStatus_t, hipcompStatus_t)
WWR_TYPE(wwrcompType_t, nvcompType_t, hipcompType_t)
WWR_TYPE(wwrcompStream_t, cudaStream_t, hipStream_t)

// ========================================================================
// Status codes -- the six both backends define (nvCOMP 5.3 adds twelve more,
// which have no hipCOMP 2.2 counterpart and are absent here). Values agree
// across backends; the pins in test/gpu/comp.cppm keep it that way.
// ========================================================================
WWR_VALUE(WWRCOMP_SUCCESS, nvcompSuccess, hipcompSuccess)
WWR_VALUE(WWRCOMP_ERROR_INVALID_VALUE, nvcompErrorInvalidValue, hipcompErrorInvalidValue)
WWR_VALUE(WWRCOMP_ERROR_NOT_SUPPORTED, nvcompErrorNotSupported, hipcompErrorNotSupported)
WWR_VALUE(WWRCOMP_ERROR_CANNOT_DECOMPRESS, nvcompErrorCannotDecompress, hipcompErrorCannotDecompress)
WWR_VALUE(WWRCOMP_ERROR_CUDA_ERROR, nvcompErrorCudaError, hipcompErrorCudaError)
WWR_VALUE(WWRCOMP_ERROR_INTERNAL, nvcompErrorInternal, hipcompErrorInternal)

// ========================================================================
// Data types -- the nine both backends define (nvCOMP adds FLOAT16 / FLOAT8,
// absent from hipCOMP 2.2 and so absent here).
// ========================================================================
WWR_VALUE(WWRCOMP_TYPE_CHAR, NVCOMP_TYPE_CHAR, HIPCOMP_TYPE_CHAR)
WWR_VALUE(WWRCOMP_TYPE_UCHAR, NVCOMP_TYPE_UCHAR, HIPCOMP_TYPE_UCHAR)
WWR_VALUE(WWRCOMP_TYPE_SHORT, NVCOMP_TYPE_SHORT, HIPCOMP_TYPE_SHORT)
WWR_VALUE(WWRCOMP_TYPE_USHORT, NVCOMP_TYPE_USHORT, HIPCOMP_TYPE_USHORT)
WWR_VALUE(WWRCOMP_TYPE_INT, NVCOMP_TYPE_INT, HIPCOMP_TYPE_INT)
WWR_VALUE(WWRCOMP_TYPE_UINT, NVCOMP_TYPE_UINT, HIPCOMP_TYPE_UINT)
WWR_VALUE(WWRCOMP_TYPE_LONGLONG, NVCOMP_TYPE_LONGLONG, HIPCOMP_TYPE_LONGLONG)
WWR_VALUE(WWRCOMP_TYPE_ULONGLONG, NVCOMP_TYPE_ULONGLONG, HIPCOMP_TYPE_ULONGLONG)
WWR_VALUE(WWRCOMP_TYPE_BITS, NVCOMP_TYPE_BITS, HIPCOMP_TYPE_BITS)

// ========================================================================
// Neutral options -- only the fields both backends share.
// ========================================================================
struct wwrcompBatchedLZ4Opts {
  wwrcompType_t data_type;
};

struct wwrcompBatchedSnappyOpts {};

struct wwrcompBatchedCascadedOpts {
  std::size_t chunk_size;
  wwrcompType_t type;
  int num_RLEs;
  int num_deltas;
  int use_bp;
};

} // namespace wwr

// ========================================================================
// Backend option builders: neutral opts -> the selected backend's struct.
// Overloaded on the neutral type so the forwarders can call one name.
// ========================================================================
namespace wwr::detail {

#if defined(WWR_GPU_BACKEND_CUDA)
inline ::wwr::cuda::nvcompBatchedLZ4CompressOpts_t backend_opts(wwr::wwrcompBatchedLZ4Opts o) {
  ::wwr::cuda::nvcompBatchedLZ4CompressOpts_t r{};
  r.data_type = o.data_type;
  return r;
}
inline ::wwr::cuda::nvcompBatchedSnappyCompressOpts_t backend_opts(wwr::wwrcompBatchedSnappyOpts) {
  return {};
}
inline ::wwr::cuda::nvcompBatchedCascadedCompressOpts_t backend_opts(wwr::wwrcompBatchedCascadedOpts o) {
  ::wwr::cuda::nvcompBatchedCascadedCompressOpts_t r{};
  r.internal_chunk_bytes = o.chunk_size;
  r.type = o.type;
  r.num_RLEs = o.num_RLEs;
  r.num_deltas = o.num_deltas;
  r.use_bp = o.use_bp;
  return r;
}
#else
inline ::wwr::hip::hipcompBatchedLZ4Opts_t backend_opts(wwr::wwrcompBatchedLZ4Opts o) {
  return {o.data_type};
}
inline ::wwr::hip::hipcompBatchedSnappyOpts_t backend_opts(wwr::wwrcompBatchedSnappyOpts) {
  return {};
}
inline ::wwr::hip::hipcompBatchedCascadedOpts_t backend_opts(wwr::wwrcompBatchedCascadedOpts o) {
  return {o.chunk_size, o.type, o.num_RLEs, o.num_deltas, o.use_bp};
}
#endif

} // namespace wwr::detail

// ========================================================================
// Batched LLIF forwarders. One neutral signature per call, modelled on the
// hipCOMP 2.2 shape; the backend body reconciles the nvCOMP 5.3 skew (Async
// temp-size, NULL compress statuses, defaulted decompress opts). The macro
// generates the six-call surface identically for each algorithm; the only
// per-algorithm variation -- opts type and vendor symbol stem -- are its
// arguments. Defined per backend so each expansion is a single backend's body.
// ========================================================================
#if defined(WWR_GPU_BACKEND_CUDA)
#define WWR_COMP_DEFINE(Algo, Opts)                                                                 \
  export namespace wwr {                                                                            \
  inline wwrcompStatus_t wwrcompBatched##Algo##CompressGetTempSize(                                 \
      std::size_t batch_size, std::size_t max_uncompressed_chunk_bytes, Opts opts,                  \
      std::size_t *temp_bytes) {                                                                    \
    return ::wwr::cuda::nvcompBatched##Algo##CompressGetTempSizeAsync(                              \
        batch_size, max_uncompressed_chunk_bytes, ::wwr::detail::backend_opts(opts), temp_bytes,   \
        batch_size *max_uncompressed_chunk_bytes);                                                  \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##CompressGetMaxOutputChunkSize(                       \
      std::size_t max_uncompressed_chunk_bytes, Opts opts, std::size_t *max_compressed_chunk_bytes) { \
    return ::wwr::cuda::nvcompBatched##Algo##CompressGetMaxOutputChunkSize(                         \
        max_uncompressed_chunk_bytes, ::wwr::detail::backend_opts(opts), max_compressed_chunk_bytes); \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##CompressAsync(                                       \
      const void *const *device_uncompressed_chunk_ptrs,                                            \
      const std::size_t *device_uncompressed_chunk_bytes,                                           \
      std::size_t max_uncompressed_chunk_bytes, std::size_t batch_size, void *device_temp_ptr,      \
      std::size_t temp_bytes, void *const *device_compressed_chunk_ptrs,                            \
      std::size_t *device_compressed_chunk_bytes, Opts opts, wwrcompStream_t stream) {              \
    return ::wwr::cuda::nvcompBatched##Algo##CompressAsync(                                         \
        device_uncompressed_chunk_ptrs, device_uncompressed_chunk_bytes,                            \
        max_uncompressed_chunk_bytes, batch_size, device_temp_ptr, temp_bytes,                      \
        device_compressed_chunk_ptrs, device_compressed_chunk_bytes,                                \
        ::wwr::detail::backend_opts(opts), nullptr, stream);                                        \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##DecompressGetTempSize(                               \
      std::size_t num_chunks, std::size_t max_uncompressed_chunk_bytes, std::size_t *temp_bytes) {  \
    return ::wwr::cuda::nvcompBatched##Algo##DecompressGetTempSizeAsync(                            \
        num_chunks, max_uncompressed_chunk_bytes,                                                   \
        ::wwr::cuda::nvcompBatched##Algo##DecompressOpts_t{}, temp_bytes,                           \
        num_chunks *max_uncompressed_chunk_bytes);                                                  \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##GetDecompressSizeAsync(                              \
      const void *const *device_compressed_chunk_ptrs,                                              \
      const std::size_t *device_compressed_chunk_bytes,                                             \
      std::size_t *device_uncompressed_chunk_bytes, std::size_t batch_size,                         \
      wwrcompStream_t stream) {                                                                     \
    return ::wwr::cuda::nvcompBatched##Algo##GetDecompressSizeAsync(                                \
        device_compressed_chunk_ptrs, device_compressed_chunk_bytes,                                \
        device_uncompressed_chunk_bytes, batch_size, stream);                                       \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##DecompressAsync(                                     \
      const void *const *device_compressed_chunk_ptrs,                                              \
      const std::size_t *device_compressed_chunk_bytes,                                             \
      const std::size_t *device_uncompressed_buffer_bytes,                                          \
      std::size_t *device_uncompressed_chunk_bytes, std::size_t num_chunks, void *device_temp_ptr,  \
      std::size_t temp_bytes, void *const *device_uncompressed_chunk_ptrs,                          \
      wwrcompStatus_t *device_statuses, wwrcompStream_t stream) {                                   \
    return ::wwr::cuda::nvcompBatched##Algo##DecompressAsync(                                       \
        device_compressed_chunk_ptrs, device_compressed_chunk_bytes,                                \
        device_uncompressed_buffer_bytes, device_uncompressed_chunk_bytes, num_chunks,             \
        device_temp_ptr, temp_bytes, device_uncompressed_chunk_ptrs,                                \
        ::wwr::cuda::nvcompBatched##Algo##DecompressOpts_t{}, device_statuses, stream);             \
  }                                                                                                 \
  }
#else
#define WWR_COMP_DEFINE(Algo, Opts)                                                                 \
  export namespace wwr {                                                                            \
  inline wwrcompStatus_t wwrcompBatched##Algo##CompressGetTempSize(                                 \
      std::size_t batch_size, std::size_t max_uncompressed_chunk_bytes, Opts opts,                  \
      std::size_t *temp_bytes) {                                                                    \
    return ::wwr::hip::hipcompBatched##Algo##CompressGetTempSize(                                   \
        batch_size, max_uncompressed_chunk_bytes, ::wwr::detail::backend_opts(opts), temp_bytes);  \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##CompressGetMaxOutputChunkSize(                       \
      std::size_t max_uncompressed_chunk_bytes, Opts opts, std::size_t *max_compressed_chunk_bytes) { \
    return ::wwr::hip::hipcompBatched##Algo##CompressGetMaxOutputChunkSize(                         \
        max_uncompressed_chunk_bytes, ::wwr::detail::backend_opts(opts), max_compressed_chunk_bytes); \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##CompressAsync(                                       \
      const void *const *device_uncompressed_chunk_ptrs,                                            \
      const std::size_t *device_uncompressed_chunk_bytes,                                           \
      std::size_t max_uncompressed_chunk_bytes, std::size_t batch_size, void *device_temp_ptr,      \
      std::size_t temp_bytes, void *const *device_compressed_chunk_ptrs,                            \
      std::size_t *device_compressed_chunk_bytes, Opts opts, wwrcompStream_t stream) {              \
    return ::wwr::hip::hipcompBatched##Algo##CompressAsync(                                         \
        device_uncompressed_chunk_ptrs, device_uncompressed_chunk_bytes,                            \
        max_uncompressed_chunk_bytes, batch_size, device_temp_ptr, temp_bytes,                      \
        device_compressed_chunk_ptrs, device_compressed_chunk_bytes,                                \
        ::wwr::detail::backend_opts(opts), stream);                                                 \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##DecompressGetTempSize(                               \
      std::size_t num_chunks, std::size_t max_uncompressed_chunk_bytes, std::size_t *temp_bytes) {  \
    return ::wwr::hip::hipcompBatched##Algo##DecompressGetTempSize(                                 \
        num_chunks, max_uncompressed_chunk_bytes, temp_bytes);                                      \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##GetDecompressSizeAsync(                              \
      const void *const *device_compressed_chunk_ptrs,                                              \
      const std::size_t *device_compressed_chunk_bytes,                                             \
      std::size_t *device_uncompressed_chunk_bytes, std::size_t batch_size,                         \
      wwrcompStream_t stream) {                                                                     \
    return ::wwr::hip::hipcompBatched##Algo##GetDecompressSizeAsync(                                \
        device_compressed_chunk_ptrs, device_compressed_chunk_bytes,                                \
        device_uncompressed_chunk_bytes, batch_size, stream);                                       \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##DecompressAsync(                                     \
      const void *const *device_compressed_chunk_ptrs,                                              \
      const std::size_t *device_compressed_chunk_bytes,                                             \
      const std::size_t *device_uncompressed_buffer_bytes,                                          \
      std::size_t *device_uncompressed_chunk_bytes, std::size_t num_chunks, void *device_temp_ptr,  \
      std::size_t temp_bytes, void *const *device_uncompressed_chunk_ptrs,                          \
      wwrcompStatus_t *device_statuses, wwrcompStream_t stream) {                                   \
    return ::wwr::hip::hipcompBatched##Algo##DecompressAsync(                                       \
        device_compressed_chunk_ptrs, device_compressed_chunk_bytes,                                \
        device_uncompressed_buffer_bytes, device_uncompressed_chunk_bytes, num_chunks,             \
        device_temp_ptr, temp_bytes, device_uncompressed_chunk_ptrs, device_statuses, stream);      \
  }                                                                                                 \
  }
#endif

WWR_COMP_DEFINE(LZ4, wwrcompBatchedLZ4Opts)
WWR_COMP_DEFINE(Snappy, wwrcompBatchedSnappyOpts)
WWR_COMP_DEFINE(Cascaded, wwrcompBatchedCascadedOpts)

#undef WWR_COMP_DEFINE
