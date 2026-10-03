/**
 * @file detail/comp_names.h
 * @brief The backend-neutral batched-compression surface, as a macro-driven
 *        include fragment shared by the module and the non-module #include path
 *
 * NOT a standalone header: it is the wwrcomp* / WWRCOMP_* surface -- the three
 * type aliases, the status and data-type constants, the neutral option structs,
 * the backend option builders, and the LZ4 / Snappy / Cascaded batched
 * forwarders -- with NO namespace of its own and NO vendor #include. The includer
 * supplies all of that and pastes this inside its own `namespace wwr` -- so one
 * list binds both ways the surface is consumed: wwr.comp (the module,
 * `export namespace wwr`) and wwr/comp.h (the non-module #include path). Add a
 * name here, once, and both paths gain it.
 *
 * HOST only -- the batched LLIF is a host-launched API. The surface binds
 * straight to the vendor's external-linkage `::nvcomp*` / `::hipcomp*`
 * declarations via the _RAW macros (the "rand.h shape"): no raw vendor module is
 * imported, so the same `::`-prefixed names resolve in the module (vendor header
 * in its GMF) and the #include path alike. The full nvCOMP 5.3 / hipCOMP 2.2
 * surface the neutral layer omits (Deflate/GZIP/Zstd/GDeflate/Bitcomp/ANS,
 * CRC32, the hardware-decompression backend, the exact Sync temp-size query, ...)
 * stays behind the raw modules; reach it through wwr.cuda.nvcomp / wwr.hip.hipcomp.
 *
 * Unlike the alias-only fragments (blas / solver / ...), comp's functions are
 * hand-written forwarding shims, not WWR_FUNCTION_RAW references: nvCOMP 5.3 and
 * hipCOMP 2.2 are version-skewed (split compress/decompress opts, an extra
 * device-status / decompress-opts parameter, Sync/Async temp-size queries with no
 * 2.2 counterpart), so no reference-alias could present one portable signature.
 * Each shim is written once against the hipCOMP 2.2 shape; the backend body
 * reconciles the skew (see src/comp.cppm's file header). Only the WWR_TYPE_RAW /
 * WWR_VALUE_RAW lines are the mechanical fragment material the other modules are.
 *
 * Before including, the includer must have, in order:
 *   - the vendor headers in scope (nvcomp/{,lz4,snappy,cascaded}.h /
 *     hipcomp/{,lz4,snappy,cascaded}.h), which comp.h pulls in;
 *   - std::size_t (the opts and the forwarders name it) -- from <cstddef>, which
 *     comp.h pulls in, on both host paths;
 *   - WWR_SELECT_RAW(cuda, hip) plus WWR_TYPE_RAW / WWR_VALUE_RAW on top of it --
 *     keyed on WWR_GPU_BACKEND_* in the module (backend.h) or WWR_SELECTED_* in
 *     the #include path (wwr/comp.h);
 *   - WWR_SELECTED_CUDA / WWR_SELECTED_HIP (selected_backend.h, via comp.h) for
 *     the builders' and forwarders' backend conditional.
 *
 * See src/comp.cppm, src/comp.h, src/wwr/comp.h and docs/architecture.md.
 */

#pragma once

#ifndef WWR_TYPE_RAW
#error                                                                                             \
    "detail/comp_names.h is an include fragment, not a standalone header: define WWR_TYPE_RAW/VALUE_RAW and WWR_SELECT_RAW, ensure the vendor headers and std::size_t (via comp.h) and WWR_SELECTED_* are in scope, and #include it inside namespace wwr. See src/comp.h, src/comp.cppm and src/wwr/comp.h."
#endif

// ========================================================================
// Types
// ========================================================================
WWR_TYPE_RAW(wwrcompStatus_t, nvcompStatus_t, hipcompStatus_t)
WWR_TYPE_RAW(wwrcompType_t, nvcompType_t, hipcompType_t)
WWR_TYPE_RAW(wwrcompStream_t, cudaStream_t, hipStream_t)

// ========================================================================
// Status codes -- the six both backends define (nvCOMP 5.3 adds twelve more,
// which have no hipCOMP 2.2 counterpart and are absent here). Values agree
// across backends; the pins in test/gpu/comp.cppm keep it that way.
// ========================================================================
WWR_VALUE_RAW(WWRCOMP_SUCCESS, nvcompSuccess, hipcompSuccess)
WWR_VALUE_RAW(WWRCOMP_ERROR_INVALID_VALUE, nvcompErrorInvalidValue, hipcompErrorInvalidValue)
WWR_VALUE_RAW(WWRCOMP_ERROR_NOT_SUPPORTED, nvcompErrorNotSupported, hipcompErrorNotSupported)
WWR_VALUE_RAW(WWRCOMP_ERROR_CANNOT_DECOMPRESS, nvcompErrorCannotDecompress, hipcompErrorCannotDecompress)
WWR_VALUE_RAW(WWRCOMP_ERROR_CUDA_ERROR, nvcompErrorCudaError, hipcompErrorCudaError)
WWR_VALUE_RAW(WWRCOMP_ERROR_INTERNAL, nvcompErrorInternal, hipcompErrorInternal)

// ========================================================================
// Data types -- the nine both backends define (nvCOMP adds FLOAT16 / FLOAT8,
// absent from hipCOMP 2.2 and so absent here).
// ========================================================================
WWR_VALUE_RAW(WWRCOMP_TYPE_CHAR, NVCOMP_TYPE_CHAR, HIPCOMP_TYPE_CHAR)
WWR_VALUE_RAW(WWRCOMP_TYPE_UCHAR, NVCOMP_TYPE_UCHAR, HIPCOMP_TYPE_UCHAR)
WWR_VALUE_RAW(WWRCOMP_TYPE_SHORT, NVCOMP_TYPE_SHORT, HIPCOMP_TYPE_SHORT)
WWR_VALUE_RAW(WWRCOMP_TYPE_USHORT, NVCOMP_TYPE_USHORT, HIPCOMP_TYPE_USHORT)
WWR_VALUE_RAW(WWRCOMP_TYPE_INT, NVCOMP_TYPE_INT, HIPCOMP_TYPE_INT)
WWR_VALUE_RAW(WWRCOMP_TYPE_UINT, NVCOMP_TYPE_UINT, HIPCOMP_TYPE_UINT)
WWR_VALUE_RAW(WWRCOMP_TYPE_LONGLONG, NVCOMP_TYPE_LONGLONG, HIPCOMP_TYPE_LONGLONG)
WWR_VALUE_RAW(WWRCOMP_TYPE_ULONGLONG, NVCOMP_TYPE_ULONGLONG, HIPCOMP_TYPE_ULONGLONG)
WWR_VALUE_RAW(WWRCOMP_TYPE_BITS, NVCOMP_TYPE_BITS, HIPCOMP_TYPE_BITS)

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

// ========================================================================
// Backend option builders: neutral opts -> the selected backend's struct.
// Overloaded on the neutral type so the forwarders can call one name. In the
// module these are exported along with the surface (harmless: inline helpers in
// wwr::detail); in the #include path they are ordinary functions.
// ========================================================================
namespace detail {

#if defined(WWR_SELECTED_CUDA)
inline ::nvcompBatchedLZ4CompressOpts_t backend_opts(wwrcompBatchedLZ4Opts o) {
  ::nvcompBatchedLZ4CompressOpts_t r{};
  r.data_type = o.data_type;
  return r;
}
inline ::nvcompBatchedSnappyCompressOpts_t backend_opts(wwrcompBatchedSnappyOpts) {
  return {};
}
inline ::nvcompBatchedCascadedCompressOpts_t backend_opts(wwrcompBatchedCascadedOpts o) {
  ::nvcompBatchedCascadedCompressOpts_t r{};
  r.internal_chunk_bytes = o.chunk_size;
  r.type = o.type;
  r.num_RLEs = o.num_RLEs;
  r.num_deltas = o.num_deltas;
  r.use_bp = o.use_bp;
  return r;
}
#else
inline ::hipcompBatchedLZ4Opts_t backend_opts(wwrcompBatchedLZ4Opts o) {
  return {o.data_type};
}
inline ::hipcompBatchedSnappyOpts_t backend_opts(wwrcompBatchedSnappyOpts) {
  return {};
}
inline ::hipcompBatchedCascadedOpts_t backend_opts(wwrcompBatchedCascadedOpts o) {
  return {o.chunk_size, o.type, o.num_RLEs, o.num_deltas, o.use_bp};
}
#endif

} // namespace detail

// ========================================================================
// Batched LLIF forwarders. One neutral signature per call, modelled on the
// hipCOMP 2.2 shape; the backend body reconciles the nvCOMP 5.3 skew (Async
// temp-size, NULL compress statuses, defaulted decompress opts). The macro
// generates the six-call surface identically for each algorithm; the only
// per-algorithm variation -- opts type and vendor symbol stem -- are its
// arguments. Defined per backend so each expansion is a single backend's body.
// The includer's enclosing [export] namespace wwr owns these; the macro adds no
// namespace of its own.
// ========================================================================
#if defined(WWR_SELECTED_CUDA)
#define WWR_COMP_DEFINE(Algo, Opts)                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##CompressGetTempSize(                                 \
      std::size_t batch_size, std::size_t max_uncompressed_chunk_bytes, Opts opts,                  \
      std::size_t *temp_bytes) {                                                                    \
    return ::nvcompBatched##Algo##CompressGetTempSizeAsync(                                         \
        batch_size, max_uncompressed_chunk_bytes, ::wwr::detail::backend_opts(opts), temp_bytes,   \
        batch_size *max_uncompressed_chunk_bytes);                                                  \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##CompressGetMaxOutputChunkSize(                       \
      std::size_t max_uncompressed_chunk_bytes, Opts opts, std::size_t *max_compressed_chunk_bytes) { \
    return ::nvcompBatched##Algo##CompressGetMaxOutputChunkSize(                                    \
        max_uncompressed_chunk_bytes, ::wwr::detail::backend_opts(opts), max_compressed_chunk_bytes); \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##CompressAsync(                                       \
      const void *const *device_uncompressed_chunk_ptrs,                                            \
      const std::size_t *device_uncompressed_chunk_bytes,                                           \
      std::size_t max_uncompressed_chunk_bytes, std::size_t batch_size, void *device_temp_ptr,      \
      std::size_t temp_bytes, void *const *device_compressed_chunk_ptrs,                            \
      std::size_t *device_compressed_chunk_bytes, Opts opts, wwrcompStream_t stream) {              \
    return ::nvcompBatched##Algo##CompressAsync(                                                    \
        device_uncompressed_chunk_ptrs, device_uncompressed_chunk_bytes,                            \
        max_uncompressed_chunk_bytes, batch_size, device_temp_ptr, temp_bytes,                      \
        device_compressed_chunk_ptrs, device_compressed_chunk_bytes,                                \
        ::wwr::detail::backend_opts(opts), nullptr, stream);                                        \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##DecompressGetTempSize(                               \
      std::size_t num_chunks, std::size_t max_uncompressed_chunk_bytes, std::size_t *temp_bytes) {  \
    return ::nvcompBatched##Algo##DecompressGetTempSizeAsync(                                       \
        num_chunks, max_uncompressed_chunk_bytes, ::nvcompBatched##Algo##DecompressOpts_t{},        \
        temp_bytes, num_chunks *max_uncompressed_chunk_bytes);                                      \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##GetDecompressSizeAsync(                              \
      const void *const *device_compressed_chunk_ptrs,                                              \
      const std::size_t *device_compressed_chunk_bytes,                                             \
      std::size_t *device_uncompressed_chunk_bytes, std::size_t batch_size,                         \
      wwrcompStream_t stream) {                                                                     \
    return ::nvcompBatched##Algo##GetDecompressSizeAsync(                                           \
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
    return ::nvcompBatched##Algo##DecompressAsync(                                                  \
        device_compressed_chunk_ptrs, device_compressed_chunk_bytes,                                \
        device_uncompressed_buffer_bytes, device_uncompressed_chunk_bytes, num_chunks,             \
        device_temp_ptr, temp_bytes, device_uncompressed_chunk_ptrs,                                \
        ::nvcompBatched##Algo##DecompressOpts_t{}, device_statuses, stream);                        \
  }
#else
#define WWR_COMP_DEFINE(Algo, Opts)                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##CompressGetTempSize(                                 \
      std::size_t batch_size, std::size_t max_uncompressed_chunk_bytes, Opts opts,                  \
      std::size_t *temp_bytes) {                                                                    \
    return ::hipcompBatched##Algo##CompressGetTempSize(                                             \
        batch_size, max_uncompressed_chunk_bytes, ::wwr::detail::backend_opts(opts), temp_bytes);  \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##CompressGetMaxOutputChunkSize(                       \
      std::size_t max_uncompressed_chunk_bytes, Opts opts, std::size_t *max_compressed_chunk_bytes) { \
    return ::hipcompBatched##Algo##CompressGetMaxOutputChunkSize(                                   \
        max_uncompressed_chunk_bytes, ::wwr::detail::backend_opts(opts), max_compressed_chunk_bytes); \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##CompressAsync(                                       \
      const void *const *device_uncompressed_chunk_ptrs,                                            \
      const std::size_t *device_uncompressed_chunk_bytes,                                           \
      std::size_t max_uncompressed_chunk_bytes, std::size_t batch_size, void *device_temp_ptr,      \
      std::size_t temp_bytes, void *const *device_compressed_chunk_ptrs,                            \
      std::size_t *device_compressed_chunk_bytes, Opts opts, wwrcompStream_t stream) {              \
    return ::hipcompBatched##Algo##CompressAsync(                                                   \
        device_uncompressed_chunk_ptrs, device_uncompressed_chunk_bytes,                            \
        max_uncompressed_chunk_bytes, batch_size, device_temp_ptr, temp_bytes,                      \
        device_compressed_chunk_ptrs, device_compressed_chunk_bytes,                                \
        ::wwr::detail::backend_opts(opts), stream);                                                 \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##DecompressGetTempSize(                               \
      std::size_t num_chunks, std::size_t max_uncompressed_chunk_bytes, std::size_t *temp_bytes) {  \
    return ::hipcompBatched##Algo##DecompressGetTempSize(                                           \
        num_chunks, max_uncompressed_chunk_bytes, temp_bytes);                                      \
  }                                                                                                 \
  inline wwrcompStatus_t wwrcompBatched##Algo##GetDecompressSizeAsync(                              \
      const void *const *device_compressed_chunk_ptrs,                                              \
      const std::size_t *device_compressed_chunk_bytes,                                             \
      std::size_t *device_uncompressed_chunk_bytes, std::size_t batch_size,                         \
      wwrcompStream_t stream) {                                                                     \
    return ::hipcompBatched##Algo##GetDecompressSizeAsync(                                          \
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
    return ::hipcompBatched##Algo##DecompressAsync(                                                 \
        device_compressed_chunk_ptrs, device_compressed_chunk_bytes,                                \
        device_uncompressed_buffer_bytes, device_uncompressed_chunk_bytes, num_chunks,             \
        device_temp_ptr, temp_bytes, device_uncompressed_chunk_ptrs, device_statuses, stream);      \
  }
#endif

WWR_COMP_DEFINE(LZ4, wwrcompBatchedLZ4Opts)
WWR_COMP_DEFINE(Snappy, wwrcompBatchedSnappyOpts)
WWR_COMP_DEFINE(Cascaded, wwrcompBatchedCascadedOpts)

#undef WWR_COMP_DEFINE
