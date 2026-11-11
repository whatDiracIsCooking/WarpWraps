"""The neutral `complex` surface, as data — the single source of truth the
generator turns into a C ABI.

This is the same `(wwr_name, cuda_name, hip_name)` tuple the `src/detail/*_names.h`
fragments already carry for the module/`#include` paths; here it carries the two
extra things a *C ABI* needs that the C++ aliases do not:

  - a stable, vendor-free POD for every by-value type, so Rust never sees
    `cuFloatComplex` vs `hipFloatComplex` (architecture.md §3: on HIP the vendor
    type is a `HIP_vector_type` *class*, a different by-value calling convention);
  - a `body` telling the thunk how to implement each call: `forward` renames to
    the backend entry point, `custom` emits a per-backend body — the HIP branch of
    `cdiv`/`cabs` is scaled, exactly as `src/complex.h` does, because ROCm's
    `hipCdiv`/`hipCabs` overflow where cuComplex scales (architecture.md §3).

A real `wwr.capi` target would generate this spec FROM the headers with libclang
rather than restate it; it is written out here so the demo is self-contained.
"""

from __future__ import annotations

# POD types the C ABI exposes, and the vendor type each maps onto per backend.
# `fields` is the #[repr(C)] / C-struct layout Rust and the vendor type share.
TYPES = [
    {
        "capi": "wwr_complex64",
        "cuda": "cuFloatComplex",
        "hip": "hipFloatComplex",
        "fields": [("float", "re"), ("float", "im")],
    },
    {
        "capi": "wwr_complex128",
        "cuda": "cuDoubleComplex",
        "hip": "hipDoubleComplex",
        "fields": [("double", "re"), ("double", "im")],
    },
]

# Functions. `ret`/`args` use capi type names (a POD above, or a plain scalar).
# body == "forward": call `cuda`/`hip` named entry point, converting POD<->vendor
#                     at the boundary.
# body == "custom":  emit the named snippet from CUSTOM below per backend.
FUNCTIONS = [
    {"capi": "wwr_cadd64", "cuda": "cuCaddf", "hip": "hipCaddf",
     "ret": "wwr_complex64", "args": [("wwr_complex64", "a"), ("wwr_complex64", "b")],
     "body": "forward"},
    {"capi": "wwr_cmul64", "cuda": "cuCmulf", "hip": "hipCmulf",
     "ret": "wwr_complex64", "args": [("wwr_complex64", "a"), ("wwr_complex64", "b")],
     "body": "forward"},
    # cabs/cdiv diverge in BEHAVIOUR, not just name — the HIP branch is scaled.
    {"capi": "wwr_cabs64", "cuda": "cuCabsf", "hip": None,
     "ret": "float", "args": [("wwr_complex64", "z")],
     "body": "cabsf"},
    {"capi": "wwr_cdiv64", "cuda": "cuCdivf", "hip": None,
     "ret": "wwr_complex64", "args": [("wwr_complex64", "a"), ("wwr_complex64", "b")],
     "body": "cdivf"},
]

# Per-backend bodies for the behavioural-divergence cases. {A}/{B}/{Z} are the
# vendor-typed locals the thunk has already built from the POD args; {RET} is the
# capi return type. Each snippet must `return` a value of {RET}. CUDA forwards to
# the scaled vendor fn; HIP reimplements cuComplex's scaling (what src/complex.h
# does, verbatim) and returns the POD directly — never naming a vendor type.
CUSTOM = {
    "cabsf": {
        "cuda": "  return ::cuCabsf({Z});",
        "hip": "  return ::hypotf({Z}.x, {Z}.y);",
    },
    "cdivf": {
        "cuda": "  const auto r = ::cuCdivf({A}, {B});\n  return {RET}{{r.x, r.y}};",
        "hip": """\
  const float oos = 1.0f / (::fabsf({B}.x) + ::fabsf({B}.y));
  const float ars = {A}.x * oos, ais = {A}.y * oos;
  const float brs = {B}.x * oos, bis = {B}.y * oos;
  const float oon = 1.0f / ((brs * brs) + (bis * bis));
  return {RET}{{((ars * brs) + (ais * bis)) * oon,
               ((ais * brs) - (ars * bis)) * oon}};""",
    },
}
