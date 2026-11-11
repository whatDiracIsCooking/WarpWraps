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

# Functions. `wwr` is the function in `wwr.complex` the thunk forwards into;
# `ret`/`args` use capi type names (a POD above, or a plain scalar).
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
]
