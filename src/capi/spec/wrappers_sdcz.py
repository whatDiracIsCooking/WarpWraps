"""The `wrappers.<family>` DEVICE surfaces, sourced from `instantiations.cpp`.

The second kind of device/template C ABI (issue #347, architecture.md §25), after
the extension_random_normal spike (#371). That one wrapped a single hand-written
launcher template; this wraps the repo's TYPE-SAFE sdcz-dispatch wrappers under
src/wrappers -- the `wwr::<fn><T>` family each of blas / fft / sparse / solver /
rand EXPLICITLY INSTANTIATES in its `instantiations.cpp`.

A C ABI cannot export a template, so -- exactly as random_normal does -- each
EXPLICIT INSTANTIATION becomes its own monomorphised `extern "C"` entry, the type
named by a suffix. The instantiation list in `src/wrappers/<family>/
instantiations.cpp` is AUTHORITATIVE: the generator parses it (no clang, stdlib
`re`, the way fragments.py reads the name fragments) and emits one thunk per
`template RET fn<Targs>(PARAMS);` line, forwarding into the linkable symbol
`wwr::fn<Targs>(...)`. A type the parse adds without this spec classifying it is a
hard error, never a silent drop (the milestone's NO SILENT DROPS rule).

The element-type suffix is sdcz by another name:

  | suffix | T (wwr)              | data POD        |
  |--------|----------------------|-----------------|
  | f32    | float                | float           |
  | f64    | double               | double          |
  | c64    | wwrFloatComplex      | wwr_complex64   |
  | c128   | wwrDoubleComplex     | wwr_complex128  |

blas additionally templates on an INDEX type (int / int64_t), appended as a
second suffix (`_i32` / `_i64`), so a blas entry reads
`wwr_wrappers_blas_gemm_f32_i32`.

Each family is a SELF-CONTAINED surface, like random_normal: it declares its OWN
capi handle typedefs (`wwr_wrappers_<family>_handle` etc. -> void*) and its OWN
stable value enums (`wwr_wrappers_<family>_status` etc.), rather than borrowing
the `wwr_<family>_*` names the vendor-library C surfaces (spec/blas.py and the
rest) declare. That keeps a wrappers unit's `.so` and its wwr-sys feature free of
any dependency on the library surface's unit -- the per-unit isolation the whole
C ABI is built on. The stable enumerator LISTS (and the neutral `wwr::WWR*`
constants the thunk maps them to) are reused 1:1 from those library specs, so the
two surfaces agree on the mapping without sharing a type name.

The neutral vendor types the templates name map across the boundary exactly as
the library surfaces do (§6): a status return and a by-value enum parameter go
through a per-enum `_to_` / `_from_` map that compares only the neutral `wwr::`
constants (so the one source compiles on both backends); an opaque handle /
descriptor is `void*`; a complex pointer is a layout-matched shared POD
(wwr_complex64 / wwr_complex128, §3/§12); every other pointer / scalar passes
through. `ComplexToRealType<T>` -- the real magnitude type a few blas calls
return by pointer (nrm2 / asum / the rot cosines) -- resolves to float for the
float/c64 instantiations and double for the double/c128 ones.

These EXECUTE on the device (a handle binds a device; a GEMM launches), so the C
ABI's gate is COMPILE + LINK on both backends (CI, no GPU); a real run needs a
card and is GPU-gated. A plain dispatch launcher does NOT probe for a device on
load (unlike hipTensor), so merely loading the unit `.so` is safe on a no-GPU box
-- the host-only C / Rust drivers link only complex + runtime_api and are
unaffected. `math` (src/wrappers/math/math.cuh) is the one src/wrappers family
NOT wrapped: it is a `__device__ __forceinline__` header with no host-callable
launcher and no `instantiations.cpp`, so there is no linkable symbol to forward
into -- a logged, explicit skip (see SKIPS below).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import blas as _blas_lib
import fft as _fft_lib
import rand as _rand_lib
import solver as _solver_lib
import sparse as _sparse_lib
from captypes import Handle, PointeePOD, ValueEnum

# The element-type suffix -> (the wwr type the template is instantiated over, the
# shared capi POD a pointer to that type carries, the real magnitude type for a
# ComplexToRealType<T> pointer). `pod` is None for the real types (a float* /
# double* passes straight through, no POD).
ELEMENT = {
    "float": {"suffix": "f32", "pod": None, "real": "float"},
    "double": {"suffix": "f64", "pod": None, "real": "double"},
    "wwrFloatComplex": {"suffix": "c64", "pod": "wwr_complex64", "real": "float"},
    "wwrDoubleComplex": {"suffix": "c128", "pod": "wwr_complex128", "real": "double"},
}

# The blas index-type suffix.
INDEX = {"int": "i32", "int64_t": "i64"}


def _reskin(ve: ValueEnum, family: str) -> ValueEnum:
    """A library-spec ValueEnum re-skinned into this surface's self-contained
    namespace: the capi type and every enumerator get the `wwr_wrappers_<family>_`
    / `WWR_WRAPPERS_<FAMILY>_` stem in place of the library surface's `wwr_<x>_` /
    `WWR_<X>_`, while the neutral `wwr::WWR*` constants (and the vendor_types) are
    kept verbatim -- the mapping is identical, only the C name differs. `wwr_type`
    is carried through so the thunk's `_to_`/`_from_` map names the same neutral
    enum the template's parameter/return is."""
    pfx = f"wwr_wrappers_{family}_"
    cpfx = f"WWR_WRAPPERS_{family.upper()}_"

    def cap(old: str) -> str:
        # wwr_blas_status -> wwr_wrappers_blas_status; wwr_fft_result ->
        # wwr_wrappers_fft_result. Strip the library surface's `wwr_<lib>_` stem
        # (its first two underscore-separated fields) and re-prefix.
        tail = old.split("_", 2)[2]
        return pfx + tail

    def cen(old: str) -> str:
        # WWR_BLAS_OP_N -> WWR_WRAPPERS_BLAS_OP_N; WWR_FFT_SUCCESS ->
        # WWR_WRAPPERS_FFT_SUCCESS. Strip the leading `WWR_` and re-prefix.
        return cpfx + old[len("WWR_"):]

    return ValueEnum(
        # The templates name the NEUTRAL wwr type (wwrblasStatus_t), not the
        # vendor spelling (cublasStatus_t) the library surface's manifest is keyed
        # on -- so the classifier resolves against the neutral name here.
        vendor_types=(ve.wwr_type,),
        capi_type=cap(ve.capi_type),
        wwr_type=ve.wwr_type,
        entries=tuple((cen(e), w) for e, w in ve.entries),
        fallback=cen(ve.fallback),
    )


@dataclass(frozen=True)
class WrapperSpec:
    """One src/wrappers sdcz-dispatch family as a device C-ABI surface, as data.

    `name` is the unit slug (`wrappers_blas`); `family` the src/wrappers
    subdirectory (`blas`); `module` the wwr module the thunk imports and forwards
    into (`wwr.wrappers.blas`); `extra_modules` are the wwr modules that only NAME
    the neutral handle / complex types (so the thunk's BMI resolves) -- the
    family's module imports but does not re-export them. `capi_prefix` is the C
    symbol stem (`wwr_wrappers_blas_`). `handles` / `pointee_pods` / `value_enums`
    are the captypes registrations the classifier resolves the templates' neutral
    types against; `indexed` is True for blas (a second, index template argument)."""

    name: str
    family: str
    module: str
    capi_prefix: str
    extra_modules: list[str] = field(default_factory=list)
    handles: list[Handle] = field(default_factory=list)
    pointee_pods: list[PointeePOD] = field(default_factory=list)
    value_enums: list[ValueEnum] = field(default_factory=list)
    scalar_aliases: dict = field(default_factory=dict)
    indexed: bool = False


# The shared complex pointee PODs every family that names a complex type reuses
# (the same mapping blas.py / fft.py declare). A wwrFloatComplex* / wwrfftComplex*
# parameter keeps the layout-matched wwr_complex64 POD, not void*.
_COMPLEX_PODS = [
    PointeePOD(vendor_types=("wwrFloatComplex", "wwrfftComplex"),
               capi_pod="wwr_complex64"),
    PointeePOD(vendor_types=("wwrDoubleComplex", "wwrfftDoubleComplex"),
               capi_pod="wwr_complex128"),
]

# --- blas ------------------------------------------------------------------
BLAS = WrapperSpec(
    name="wrappers_blas",
    family="blas",
    module="wwr.wrappers.blas",
    capi_prefix="wwr_wrappers_blas_",
    extra_modules=["wwr.blas", "wwr.complex"],
    indexed=True,
    handles=[Handle(vendor_types=("wwrblasHandle_t",),
                    capi_type="wwr_wrappers_blas_handle")],
    pointee_pods=_COMPLEX_PODS,
    value_enums=[_reskin(_blas_lib.STATUS, "blas"),
                 _reskin(_blas_lib.OPERATION, "blas"),
                 _reskin(_blas_lib.FILL_MODE, "blas"),
                 _reskin(_blas_lib.DIAG_TYPE, "blas"),
                 _reskin(_blas_lib.SIDE_MODE, "blas")],
)

# --- fft --------------------------------------------------------------------
# wwrfftHandle is a plan handle that DIVERGES across backends: cufftHandle is a
# plan id (typedef int) on CUDA, but hipfftHandle is an opaque POINTER
# (hipfftHandle_t *) on HIP. A single C ABI cannot be an int on one backend and
# a pointer on the other, so it is carried as the vendor-free opaque void* on
# BOTH (a Handle): the thunk C-style-casts the void* to the neutral wwrfftHandle
# the template expects (a pointer on HIP, the int plan id on CUDA), and a
# consumer round-trips the same bits it got from the plan-create call. (fft.py,
# the host library surface, folds to int because it resolves against the ROCm
# MANIFEST where the plan is an int; the wrappers template names the LIVE
# neutral type, which is the pointer on this ROCm -- so void* is the safe
# representation for the device surface.)
FFT = WrapperSpec(
    name="wrappers_fft",
    family="fft",
    module="wwr.wrappers.fft",
    capi_prefix="wwr_wrappers_fft_",
    extra_modules=["wwr.fft", "wwr.complex"],
    handles=[Handle(vendor_types=("wwrfftHandle",),
                    capi_type="wwr_wrappers_fft_handle")],
    pointee_pods=_COMPLEX_PODS,
    value_enums=[_reskin(_fft_lib.RESULT, "fft")],
)

# --- sparse -----------------------------------------------------------------
SPARSE = WrapperSpec(
    name="wrappers_sparse",
    family="sparse",
    module="wwr.wrappers.sparse",
    capi_prefix="wwr_wrappers_sparse_",
    extra_modules=["wwr.sparse", "wwr.complex"],
    handles=[Handle(vendor_types=("wwrsparseHandle_t",),
                    capi_type="wwr_wrappers_sparse_handle"),
             Handle(vendor_types=("wwrsparseMatDescr_t",),
                    capi_type="wwr_wrappers_sparse_mat_descr")],
    pointee_pods=_COMPLEX_PODS,
    value_enums=[_reskin(_sparse_lib.STATUS, "sparse"),
                 _reskin(_sparse_lib.OPERATION, "sparse"),
                 _reskin(_sparse_lib.DIRECTION, "sparse"),
                 _reskin(_sparse_lib.ACTION, "sparse"),
                 _reskin(_sparse_lib.INDEX_BASE, "sparse")],
)

# --- solver -----------------------------------------------------------------
# The solver templates name the blas Operation / FillMode / SideMode enums (not a
# solver-specific spelling), so their maps reuse the blas library-spec entry
# lists, re-skinned into the solver surface's namespace.
SOLVER = WrapperSpec(
    name="wrappers_solver",
    family="solver",
    module="wwr.wrappers.solver",
    capi_prefix="wwr_wrappers_solver_",
    extra_modules=["wwr.solver", "wwr.blas", "wwr.complex"],
    handles=[Handle(vendor_types=("wwrsolverDnHandle_t",),
                    capi_type="wwr_wrappers_solver_handle"),
             Handle(vendor_types=("wwrsolverGesvdjInfo_t",),
                    capi_type="wwr_wrappers_solver_gesvdj_info"),
             Handle(vendor_types=("wwrsolverSyevjInfo_t",),
                    capi_type="wwr_wrappers_solver_syevj_info")],
    pointee_pods=_COMPLEX_PODS,
    value_enums=[_reskin(_solver_lib.STATUS, "solver"),
                 _reskin(_solver_lib.EIG_MODE, "solver"),
                 _reskin(_solver_lib.EIG_RANGE, "solver"),
                 _reskin(_solver_lib.EIG_TYPE, "solver"),
                 # The blas Operation / FillMode / SideMode the solver templates
                 # take, re-skinned into the solver namespace (same neutral maps).
                 _reskin(_blas_lib.OPERATION, "solver"),
                 _reskin(_blas_lib.FILL_MODE, "solver"),
                 _reskin(_blas_lib.SIDE_MODE, "solver")],
)

# --- rand -------------------------------------------------------------------
RAND = WrapperSpec(
    name="wrappers_rand",
    family="rand",
    module="wwr.wrappers.rand",
    capi_prefix="wwr_wrappers_rand_",
    extra_modules=["wwr.rand"],
    handles=[Handle(vendor_types=("wwrrandGenerator_t",),
                    capi_type="wwr_wrappers_rand_generator")],
    value_enums=[_reskin(_rand_lib.STATUS, "rand")],
)

SPECS = [BLAS, FFT, SPARSE, SOLVER, RAND]

# The one src/wrappers family deliberately NOT wrapped, with its reason -- loud,
# never silent (NO SILENT DROPS). math is a device-only intrinsic header, not an
# sdcz-dispatch wrapper with a host launcher.
SKIPS = [
    ("wrappers_math",
     "src/wrappers/math/math.cuh is a __device__ __forceinline__ intrinsic "
     "header (wwr::exp / sin / fma over float/double), callable only from a "
     "kernel body -- it has no host-callable launcher and no instantiations.cpp, "
     "so there is no explicitly-instantiated linkable symbol for a C thunk to "
     "forward into. Out of scope for a host-boundary C ABI."),
]
