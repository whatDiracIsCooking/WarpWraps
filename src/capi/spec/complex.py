"""The neutral `complex` surface, as data — a DELIBERATE hand-written special
case, the one surface the fragment-sourcing path cannot cover.

Every other capi surface sources its names from `src/detail/*_names.h` and its
signatures from the vendor manifests (spec/runtime_status.py is the model, and
architecture.md §25 is the verdict). `complex` cannot: its thunks forward into
wwr-AUTHORED inline wrappers — `wwr::wwrCaddf` / `wwrCmulf` / `wwrCabsf` /
`wwrCdivf` and the `double` set, exported from `wwr.complex` — which are NOT
vendor functions. They appear in no `vendor/*/*.json` manifest (the manifests
harvest the vendors' own declared symbols), so there is no `decl` to read a
signature from. The cuComplex-vs-hipComplex divergence they resolve (scaled cdiv
/ hypotf cabs, architecture.md §3) lives inside the `wwr::wwrC*` function, below
the thunk, so each thunk is still a plain one-shape-over-both-backends forward —
but the surface is authored, not harvested, so it is spelled out here.

The one thing a C ABI needs that the `wwr` names do not carry is a stable,
vendor-free POD for every by-value type, so Rust never sees `cuFloatComplex` vs
`hipFloatComplex` (architecture.md §3: on HIP the vendor type is a
`HIP_vector_type` *class*, a different by-value calling convention). The thunk
converts each POD to/from the neutral `wwr::wwrFloatComplex` / `wwr::wwrDoubleComplex`
at the boundary; `fields` is the `#[repr(C)]` / C-struct layout Rust and the
neutral type share — the §4/§12 "a name does not pin the bytes" residue, carried
here as the layout sidecar (the fragment-sourced surfaces carry theirs in
ctypes.Struct for the same reason).

This is the ONLY hand-written surface, and it is kept small on purpose; the
status slice that used to live here moved to spec/runtime_status.py once the
fragment path was realized (architecture.md §25, #328).
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
# `ret`/`args` use capi type names (a POD above, or a plain scalar). The status
# slice that used to sit here is now fragment-sourced (spec/runtime_status.py).
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
