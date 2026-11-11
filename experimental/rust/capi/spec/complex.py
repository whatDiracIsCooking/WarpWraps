"""The neutral `complex` surface, as data — the single source of truth the
generator turns into a C ABI.

Each function carries only what the generator needs now that the thunk calls
*into* `wwr`: its `capi` name, the `wwr` function to call (exported from
`wwr.complex`), the return type and the argument list. There is no per-backend
name and no body: every function is a plain forward into `wwr::<wwr>`, because
the cuComplex-vs-hipComplex divergence (scaled cdiv / hypotf cabs, architecture.md
§3) is already resolved *inside* the `wwr::wwrC*` function, below the thunk — so
the generated thunk is one shape over both backends and the backend is chosen only
by which `wwr` library the capi links.

The one thing a C ABI needs that the `wwr` names do not carry is a stable,
vendor-free POD for every by-value type, so Rust never sees `cuFloatComplex` vs
`hipFloatComplex` (architecture.md §3: on HIP the vendor type is a
`HIP_vector_type` *class*, a different by-value calling convention). The thunk
converts each POD to/from the neutral `wwr::wwrFloatComplex` / `wwr::wwrDoubleComplex`
at the boundary; `fields` is the `#[repr(C)]` / C-struct layout Rust and the
neutral type share.

A real `wwr.capi` target would generate this spec FROM the canonical name surface
(`src/detail/*_names.h`) rather than restate it (architecture.md §25, #328); it is
written out here so the demo is self-contained.

Status surface (#329 increment 2). A status-returning surface forces the one
design problem `complex` never does: the backend error-enum VALUES differ even
where the names agree (architecture.md §6), so the C ABI must NOT pass the raw
backend status int through to Rust. It maps the neutral `wwr::wwrError_t` onto a
stable `wwr_status` C enum with FIXED integer values -- resolved below the
`extern "C"` line, exactly as the neutral POD resolves the by-value divergence for
`complex`. Rust then sees one set of values whichever backend the capi links, and
turns `wwr_status` into a `Result`. `STATUS` below is the single source for both
the generated enum and the thunk's `_map` helper, so they cannot drift; `_map`
compares against the NEUTRAL `wwr::wwr<Name>` constants, never a vendor name.
"""

from __future__ import annotations

# POD types the C ABI exposes, and the neutral `wwr` type each maps onto.
# `fields` is the #[repr(C)] / C-struct layout Rust and the neutral type share.
TYPES = [
    {
        "capi": "wwr_complex64",
        "wwr": "wwr::wwrFloatComplex",
        "fields": [("float", "re"), ("float", "im")],
    },
    {
        "capi": "wwr_complex128",
        "wwr": "wwr::wwrDoubleComplex",
        "fields": [("double", "re"), ("double", "im")],
    },
]

# The stable C status enum, as data. Each pair is (C enumerator, neutral wwr
# constant). The generator emits `wwr_status` with FIXED ascending integer values
# in THIS order -- WWR_STATUS_SUCCESS first (== 0), then the curated error set,
# then WWR_STATUS_UNKNOWN as the catch-all -- and emits the thunk's `_map` from the
# same list, matching each `wwr::wwr<constant>` and returning the paired
# enumerator. The values are stable and backend-independent (that is the whole
# point, architecture.md §6); a `wwr::wwrError_t` the map does not recognise falls
# through to WWR_STATUS_UNKNOWN. SUCCESS and UNKNOWN are generated as the first and
# last enumerators automatically; list only the error constants to map here.
STATUS = [
    ("WWR_STATUS_INVALID_VALUE", "wwrErrorInvalidValue"),
    ("WWR_STATUS_INVALID_DEVICE", "wwrErrorInvalidDevice"),
    ("WWR_STATUS_NO_DEVICE", "wwrErrorNoDevice"),
    ("WWR_STATUS_NOT_READY", "wwrErrorNotReady"),
    ("WWR_STATUS_MEMORY_ALLOCATION", "wwrErrorMemoryAllocation"),
    ("WWR_STATUS_NOT_SUPPORTED", "wwrErrorNotSupported"),
]

# Functions. `wwr` is the function in `wwr.complex` / `wwr.runtime_api` the thunk
# forwards into; `ret`/`args` use capi type names (a POD above, a plain scalar, a
# pointer like `int*` that passes straight through, or `wwr_status` -- which makes
# the generator call the wwr fn and `return _map(...)` the result).
FUNCTIONS = [
    {"capi": "wwr_cadd64", "wwr": "wwr::wwrCaddf",
     "ret": "wwr_complex64", "args": [("wwr_complex64", "a"), ("wwr_complex64", "b")]},
    {"capi": "wwr_cmul64", "wwr": "wwr::wwrCmulf",
     "ret": "wwr_complex64", "args": [("wwr_complex64", "a"), ("wwr_complex64", "b")]},
    {"capi": "wwr_cabs64", "wwr": "wwr::wwrCabsf",
     "ret": "float", "args": [("wwr_complex64", "z")]},
    {"capi": "wwr_cdiv64", "wwr": "wwr::wwrCdivf",
     "ret": "wwr_complex64", "args": [("wwr_complex64", "a"), ("wwr_complex64", "b")]},
    {"capi": "wwr_cadd128", "wwr": "wwr::wwrCadd", "ret": "wwr_complex128",
     "args": [("wwr_complex128", "a"), ("wwr_complex128", "b")]},
    {"capi": "wwr_cmul128", "wwr": "wwr::wwrCmul", "ret": "wwr_complex128",
     "args": [("wwr_complex128", "a"), ("wwr_complex128", "b")]},
    {"capi": "wwr_cabs128", "wwr": "wwr::wwrCabs", "ret": "double",
     "args": [("wwr_complex128", "z")]},
    {"capi": "wwr_cdiv128", "wwr": "wwr::wwrCdiv", "ret": "wwr_complex128",
     "args": [("wwr_complex128", "a"), ("wwr_complex128", "b")]},
    # Status surface -- a thin slice of runtime_api (#329 increment 2). Each
    # returns the stable wwr_status; the generator calls the wwr fn and maps its
    # wwr::wwrError_t through _map. `int*` out-params pass straight through, no POD.
    {"capi": "wwr_runtime_get_version", "wwr": "wwr::wwrRuntimeGetVersion",
     "ret": "wwr_status", "args": [("int*", "version")]},
    {"capi": "wwr_get_device_count", "wwr": "wwr::wwrGetDeviceCount",
     "ret": "wwr_status", "args": [("int*", "count")]},
    {"capi": "wwr_set_device", "wwr": "wwr::wwrSetDevice",
     "ret": "wwr_status", "args": [("int", "device")]},
]
