"""The `sparse` (cuSPARSE / hipSPARSE) surface as a FRAGMENT-SOURCED spec.

The same machine spec/blas.py drives (spec/library.py's LibrarySpec), over the
legacy cuSPARSE / hipSPARSE surface detail/sparse_names.h curates: the handle /
stream / pointer-mode calls, the legacy matrix-descriptor helpers, the BSR
mat-vec, the tridiagonal / pentadiagonal batch solvers, csrgeam2, and the nnz /
gebsr2gebsc / csr2gebsr conversions. It REGISTERS the vendor-free types this
surface forces:

  * HANDLES  wwrsparseHandle_t -> void*, cudaStream_t -> wwr_stream (shared), and
    the opaque legacy MATRIX DESCRIPTOR wwrsparseMatDescr_t (a vendor pointer
    typedef, by value and as an out-param) -> wwr_sparse_mat_descr
  * VALUE ENUMS  Status / Operation / Direction / Action / IndexBase /
    MatrixType / FillMode / DiagType / PointerMode -- each a stable capi enum the
    thunk maps capi<->neutral at the boundary (§6)
  * POINTEE PODS  cuComplex* -> wwr_complex64*, cuDoubleComplex* -> wwr_complex128*
    (the shared common.h PODs; the *_bufferSize complex shims name them)

The *_bufferSize conversion shims are hand-written on CUDA (an int-vs-size_t skew
reconciled in the fragment) but appear as WWR_FUNCTION_RAW on the HIP branch, so
the fragment parser sees them and the generator sources the HIP `size_t *`
signature -- the one both backends present. Everything sparse executes runs on
the device, so the thunks are link-checked on no-GPU CI.
"""

from __future__ import annotations

from captypes import Handle, PointeePOD, ValueEnum
from library import LibrarySpec, snake_capi_name

FRAGMENT = "detail/sparse_names.h"
PREFIX = "wwrsparse"
CAPI_PREFIX = "wwr_sparse_"

CUDA_MANIFEST = "cuda-13.0.x/cusparse.json"
HIP_MANIFEST = "rocm-10.0.0/hipsparse.json"
MODULES = ["wwr.sparse", "wwr.complex"]

HANDLES = [
    Handle(vendor_types=("cusparseHandle_t", "hipsparseHandle_t"),
           capi_type="wwr_sparse_handle"),
    Handle(vendor_types=("cudaStream_t", "hipStream_t"), capi_type="wwr_stream"),
    # The legacy matrix descriptor: a vendor pointer typedef, opaque to the ABI.
    Handle(vendor_types=("cusparseMatDescr_t", "hipsparseMatDescr_t"),
           capi_type="wwr_sparse_mat_descr"),
]

POINTEE_PODS = [
    PointeePOD(vendor_types=("cuComplex", "hipComplex"), capi_pod="wwr_complex64"),
    PointeePOD(vendor_types=("cuDoubleComplex", "hipDoubleComplex"),
               capi_pod="wwr_complex128"),
]

STATUS = ValueEnum(
    vendor_types=("cusparseStatus_t", "hipsparseStatus_t"),
    capi_type="wwr_sparse_status",
    wwr_type="wwrsparseStatus_t",
    fallback="WWR_SPARSE_STATUS_UNKNOWN",
    entries=(
        ("WWR_SPARSE_STATUS_SUCCESS", "WWRSPARSE_STATUS_SUCCESS"),
    ),
)

OPERATION = ValueEnum(
    vendor_types=("cusparseOperation_t", "hipsparseOperation_t"),
    capi_type="wwr_sparse_operation",
    wwr_type="wwrsparseOperation_t",
    fallback="WWR_SPARSE_OPERATION_NON_TRANSPOSE",
    entries=(
        ("WWR_SPARSE_OPERATION_NON_TRANSPOSE", "WWRSPARSE_OPERATION_NON_TRANSPOSE"),
        ("WWR_SPARSE_OPERATION_TRANSPOSE", "WWRSPARSE_OPERATION_TRANSPOSE"),
        ("WWR_SPARSE_OPERATION_CONJUGATE_TRANSPOSE",
         "WWRSPARSE_OPERATION_CONJUGATE_TRANSPOSE"),
    ),
)

DIRECTION = ValueEnum(
    vendor_types=("cusparseDirection_t", "hipsparseDirection_t"),
    capi_type="wwr_sparse_direction",
    wwr_type="wwrsparseDirection_t",
    fallback="WWR_SPARSE_DIRECTION_ROW",
    entries=(
        ("WWR_SPARSE_DIRECTION_ROW", "WWRSPARSE_DIRECTION_ROW"),
        ("WWR_SPARSE_DIRECTION_COLUMN", "WWRSPARSE_DIRECTION_COLUMN"),
    ),
)

ACTION = ValueEnum(
    vendor_types=("cusparseAction_t", "hipsparseAction_t"),
    capi_type="wwr_sparse_action",
    wwr_type="wwrsparseAction_t",
    fallback="WWR_SPARSE_ACTION_SYMBOLIC",
    entries=(
        ("WWR_SPARSE_ACTION_SYMBOLIC", "WWRSPARSE_ACTION_SYMBOLIC"),
        ("WWR_SPARSE_ACTION_NUMERIC", "WWRSPARSE_ACTION_NUMERIC"),
    ),
)

INDEX_BASE = ValueEnum(
    vendor_types=("cusparseIndexBase_t", "hipsparseIndexBase_t"),
    capi_type="wwr_sparse_index_base",
    wwr_type="wwrsparseIndexBase_t",
    fallback="WWR_SPARSE_INDEX_BASE_ZERO",
    entries=(
        ("WWR_SPARSE_INDEX_BASE_ZERO", "WWRSPARSE_INDEX_BASE_ZERO"),
        ("WWR_SPARSE_INDEX_BASE_ONE", "WWRSPARSE_INDEX_BASE_ONE"),
    ),
)

MATRIX_TYPE = ValueEnum(
    vendor_types=("cusparseMatrixType_t", "hipsparseMatrixType_t"),
    capi_type="wwr_sparse_matrix_type",
    wwr_type="wwrsparseMatrixType_t",
    fallback="WWR_SPARSE_MATRIX_TYPE_GENERAL",
    entries=(
        ("WWR_SPARSE_MATRIX_TYPE_GENERAL", "WWRSPARSE_MATRIX_TYPE_GENERAL"),
        ("WWR_SPARSE_MATRIX_TYPE_SYMMETRIC", "WWRSPARSE_MATRIX_TYPE_SYMMETRIC"),
        ("WWR_SPARSE_MATRIX_TYPE_HERMITIAN", "WWRSPARSE_MATRIX_TYPE_HERMITIAN"),
        ("WWR_SPARSE_MATRIX_TYPE_TRIANGULAR", "WWRSPARSE_MATRIX_TYPE_TRIANGULAR"),
    ),
)

FILL_MODE = ValueEnum(
    vendor_types=("cusparseFillMode_t", "hipsparseFillMode_t"),
    capi_type="wwr_sparse_fill_mode",
    wwr_type="wwrsparseFillMode_t",
    fallback="WWR_SPARSE_FILL_MODE_LOWER",
    entries=(
        ("WWR_SPARSE_FILL_MODE_LOWER", "WWRSPARSE_FILL_MODE_LOWER"),
        ("WWR_SPARSE_FILL_MODE_UPPER", "WWRSPARSE_FILL_MODE_UPPER"),
    ),
)

DIAG_TYPE = ValueEnum(
    vendor_types=("cusparseDiagType_t", "hipsparseDiagType_t"),
    capi_type="wwr_sparse_diag_type",
    wwr_type="wwrsparseDiagType_t",
    fallback="WWR_SPARSE_DIAG_TYPE_NON_UNIT",
    entries=(
        ("WWR_SPARSE_DIAG_TYPE_NON_UNIT", "WWRSPARSE_DIAG_TYPE_NON_UNIT"),
        ("WWR_SPARSE_DIAG_TYPE_UNIT", "WWRSPARSE_DIAG_TYPE_UNIT"),
    ),
)

POINTER_MODE = ValueEnum(
    vendor_types=("cusparsePointerMode_t", "hipsparsePointerMode_t"),
    capi_type="wwr_sparse_pointer_mode",
    wwr_type="wwrsparsePointerMode_t",
    fallback="WWR_SPARSE_POINTER_MODE_HOST",
    entries=(
        ("WWR_SPARSE_POINTER_MODE_HOST", "WWRSPARSE_POINTER_MODE_HOST"),
        ("WWR_SPARSE_POINTER_MODE_DEVICE", "WWRSPARSE_POINTER_MODE_DEVICE"),
    ),
)

VALUE_ENUMS = [STATUS, OPERATION, DIRECTION, ACTION, INDEX_BASE, MATRIX_TYPE,
               FILL_MODE, DIAG_TYPE, POINTER_MODE]


def capi_name(wwr_name: str) -> str:
    return snake_capi_name(wwr_name, PREFIX, CAPI_PREFIX)


SPEC = LibrarySpec(
    name="sparse",
    fragment=FRAGMENT,
    capi_prefix=CAPI_PREFIX,
    cuda_manifest=CUDA_MANIFEST,
    hip_manifest=HIP_MANIFEST,
    modules=MODULES,
    handles=HANDLES,
    pointee_pods=POINTEE_PODS,
    value_enums=VALUE_ENUMS,
    capi_name=capi_name,
)
