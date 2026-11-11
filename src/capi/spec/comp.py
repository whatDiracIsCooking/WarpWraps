"""The `comp` (nvCOMP / hipCOMP) batched-compression surface, HAND-WRITTEN.

comp is the second hand-written surface after complex, and for the same reason:
its functions are wwr-AUTHORED forwarding shims (detail/comp_names.h's
WWR_COMP_DEFINE macro over the hipCOMP 2.2 shape, reconciling the nvCOMP 5.3
skew), NOT WWR_FUNCTION_RAW references -- there is no vendor `decl` in any
manifest to source a signature from. So the surface is enumerated here by hand,
once per (algorithm, call), against the neutral wwrcomp* signatures the fragment
defines.

The surface is small and regular: three algorithms (LZ4, Snappy, Cascaded) times
six calls each = 18 functions, plus the status / data-type value enums, the
stream handle, and three by-value option PODs. The thunk (generate.py
gen_comp_cpp) imports ONLY wwr.comp -> links only the vendor compression library
(nvcomp / hipcomp), the per-unit isolation. Every call executes on the device, so
the thunks are link-checked on no-GPU CI.

Snappy's neutral opts struct is empty; the capi POD carries one reserved byte so
C and Rust agree on a non-zero size (an empty struct's size differs between C,
C++ and Rust), and the thunk ignores it -- it builds wwr::wwrcompBatchedSnappyOpts{}.
"""

from __future__ import annotations

from captypes import Handle, ValueEnum

FRAGMENT = "detail/comp_names.h"

# The stream handle (cudaStream_t / hipStream_t), shared capi name wwr_stream
# (same as blas / sparse / fft) -- opaque void*.
HANDLES = (
    Handle(vendor_types=("cudaStream_t", "hipStream_t"), capi_type="wwr_stream"),
)

STATUS = ValueEnum(
    vendor_types=("nvcompStatus_t", "hipcompStatus_t"),
    capi_type="wwr_comp_status",
    wwr_type="wwrcompStatus_t",
    fallback="WWR_COMP_STATUS_UNKNOWN",
    entries=(
        ("WWR_COMP_SUCCESS", "WWRCOMP_SUCCESS"),
        ("WWR_COMP_ERROR_INVALID_VALUE", "WWRCOMP_ERROR_INVALID_VALUE"),
        ("WWR_COMP_ERROR_NOT_SUPPORTED", "WWRCOMP_ERROR_NOT_SUPPORTED"),
        ("WWR_COMP_ERROR_CANNOT_DECOMPRESS", "WWRCOMP_ERROR_CANNOT_DECOMPRESS"),
        ("WWR_COMP_ERROR_CUDA_ERROR", "WWRCOMP_ERROR_CUDA_ERROR"),
        ("WWR_COMP_ERROR_INTERNAL", "WWRCOMP_ERROR_INTERNAL"),
    ),
)

DATA_TYPE = ValueEnum(
    vendor_types=("nvcompType_t", "hipcompType_t"),
    capi_type="wwr_comp_type",
    wwr_type="wwrcompType_t",
    fallback="WWR_COMP_TYPE_CHAR",
    entries=(
        ("WWR_COMP_TYPE_CHAR", "WWRCOMP_TYPE_CHAR"),
        ("WWR_COMP_TYPE_UCHAR", "WWRCOMP_TYPE_UCHAR"),
        ("WWR_COMP_TYPE_SHORT", "WWRCOMP_TYPE_SHORT"),
        ("WWR_COMP_TYPE_USHORT", "WWRCOMP_TYPE_USHORT"),
        ("WWR_COMP_TYPE_INT", "WWRCOMP_TYPE_INT"),
        ("WWR_COMP_TYPE_UINT", "WWRCOMP_TYPE_UINT"),
        ("WWR_COMP_TYPE_LONGLONG", "WWRCOMP_TYPE_LONGLONG"),
        ("WWR_COMP_TYPE_ULONGLONG", "WWRCOMP_TYPE_ULONGLONG"),
        ("WWR_COMP_TYPE_BITS", "WWRCOMP_TYPE_BITS"),
    ),
)

VALUE_ENUMS = (STATUS, DATA_TYPE)

# The three by-value option PODs. `capi` is the C/Rust struct name; `wwr` is the
# neutral wwr struct the thunk builds and forwards; `fields` are (capi_c_type,
# name, how-to-map). A `type` field holds a wwr_comp_type enum, carried through
# the value-enum map. Snappy is empty on the wwr side; the capi POD keeps one
# reserved byte (see module docstring) the thunk ignores.
LZ4_OPTS = {
    "capi": "wwr_comp_lz4_opts", "wwr": "wwrcompBatchedLZ4Opts",
    "fields": [("wwr_comp_type", "data_type", "enum")],
}
SNAPPY_OPTS = {
    "capi": "wwr_comp_snappy_opts", "wwr": "wwrcompBatchedSnappyOpts",
    "fields": [("uint8_t", "reserved", "ignore")],
}
CASCADED_OPTS = {
    "capi": "wwr_comp_cascaded_opts", "wwr": "wwrcompBatchedCascadedOpts",
    "fields": [("size_t", "chunk_size", "scalar"),
               ("wwr_comp_type", "type", "enum"),
               ("int", "num_RLEs", "scalar"),
               ("int", "num_deltas", "scalar"),
               ("int", "use_bp", "scalar")],
}
OPTS = [LZ4_OPTS, SNAPPY_OPTS, CASCADED_OPTS]

# The three algorithms: (capi infix, wwr infix, opts POD). The capi infix is the
# snake form; the wwr infix is the camel form the neutral names use.
ALGOS = [
    ("lz4", "LZ4", LZ4_OPTS),
    ("snappy", "Snappy", SNAPPY_OPTS),
    ("cascaded", "Cascaded", CASCADED_OPTS),
]

# The six batched calls each algorithm provides, as (capi suffix, wwr suffix,
# params). A param is (classified_repr, capi_c_type, name) where repr drives the
# thunk: "scalar"/"pointer" pass through (a pointer is reinterpret_cast to the
# wwr parameter type), "opts" is the by-value POD built into the wwr opts,
# "stream" is the wwr_stream handle (void* -> the backend stream). Every call
# returns wwr_comp_status.
#
# DecompressAsync's `device_statuses` is a DEVICE-written out-array the kernel
# fills with BACKEND-NATIVE status codes AFTER the async launch returns, so the
# host cannot translate it to the stable capi enum (§6) the way a synchronous
# return is translated. It is therefore exposed as an opaque `void *` carrying the
# raw backend codes -- an explicit, documented divergence (the function is still
# wrapped; nothing is dropped), reported by generate.py.
#
# The signatures mirror detail/comp_names.h's WWR_COMP_DEFINE macro exactly (the
# hipCOMP 2.2 shape the neutral layer presents).
def _calls(opts_capi: str):
    return [
        ("compress_get_temp_size", "CompressGetTempSize", [
            ("scalar", "size_t", "batch_size"),
            ("scalar", "size_t", "max_uncompressed_chunk_bytes"),
            ("opts", opts_capi, "opts"),
            ("pointer", "size_t *", "temp_bytes"),
        ]),
        ("compress_get_max_output_chunk_size", "CompressGetMaxOutputChunkSize", [
            ("scalar", "size_t", "max_uncompressed_chunk_bytes"),
            ("opts", opts_capi, "opts"),
            ("pointer", "size_t *", "max_compressed_chunk_bytes"),
        ]),
        ("compress_async", "CompressAsync", [
            ("pointer", "const void *const *", "device_uncompressed_chunk_ptrs"),
            ("pointer", "const size_t *", "device_uncompressed_chunk_bytes"),
            ("scalar", "size_t", "max_uncompressed_chunk_bytes"),
            ("scalar", "size_t", "batch_size"),
            ("pointer", "void *", "device_temp_ptr"),
            ("scalar", "size_t", "temp_bytes"),
            ("pointer", "void *const *", "device_compressed_chunk_ptrs"),
            ("pointer", "size_t *", "device_compressed_chunk_bytes"),
            ("opts", opts_capi, "opts"),
            ("stream", "wwr_stream", "stream"),
        ]),
        ("decompress_get_temp_size", "DecompressGetTempSize", [
            ("scalar", "size_t", "num_chunks"),
            ("scalar", "size_t", "max_uncompressed_chunk_bytes"),
            ("pointer", "size_t *", "temp_bytes"),
        ]),
        ("get_decompress_size_async", "GetDecompressSizeAsync", [
            ("pointer", "const void *const *", "device_compressed_chunk_ptrs"),
            ("pointer", "const size_t *", "device_compressed_chunk_bytes"),
            ("pointer", "size_t *", "device_uncompressed_chunk_bytes"),
            ("scalar", "size_t", "batch_size"),
            ("stream", "wwr_stream", "stream"),
        ]),
        ("decompress_async", "DecompressAsync", [
            ("pointer", "const void *const *", "device_compressed_chunk_ptrs"),
            ("pointer", "const size_t *", "device_compressed_chunk_bytes"),
            ("pointer", "const size_t *", "device_uncompressed_buffer_bytes"),
            ("pointer", "size_t *", "device_uncompressed_chunk_bytes"),
            ("scalar", "size_t", "num_chunks"),
            ("pointer", "void *", "device_temp_ptr"),
            ("scalar", "size_t", "temp_bytes"),
            ("pointer", "void *const *", "device_uncompressed_chunk_ptrs"),
            ("pointer", "void *", "device_statuses"),
            ("stream", "wwr_stream", "stream"),
        ]),
    ]


def functions():
    """The 18 resolved comp calls, each a dict with capi name, wwr name, params
    (list of (repr, capi_c, name)), and the opts POD it carries (if any)."""
    out = []
    for capi_algo, wwr_algo, opts in ALGOS:
        for capi_suffix, wwr_suffix, params in _calls(opts["capi"]):
            out.append({
                "capi": f"wwr_comp_batched_{capi_algo}_{capi_suffix}",
                "wwr": f"wwr::wwrcompBatched{wwr_algo}{wwr_suffix}",
                "params": params,
                "opts": opts,
            })
    return out
