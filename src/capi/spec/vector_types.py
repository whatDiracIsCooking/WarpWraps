"""The `vector_types` numeric-type surface, HAND-WRITTEN (the complex / comp shape).

vector_types is a TYPE surface: the CUDA / HIP vector types (float2, int4, ...)
and their make_* constructors, which wwr.vector_types re-exports into namespace
wwr. The constructors are wwr-AUTHORED portable braces (detail/vector_types_names.h),
NOT vendor functions in any manifest -- so no `decl` to source, and the surface is
enumerated here by hand, like complex.py and comp.py.

WHY A POD PER TYPE, not a shared header one. A vector type is a by-value
aggregate whose LAYOUT (N components of one scalar) the C ABI must pin so Rust
and C agree on the bytes (§4/§12). Each is a small #[repr(C)] POD -- float2 is
{float x, y}, int4 is {int x, y, z, w}. They are this unit's OWN PODs (declared
in the unit header, not gen/common.h): no other surface names a float2 by value,
so unlike wwr_half they are not shared numeric vocabulary.

Only CONSTRUCTION is host-wrapped, which is the whole surface: a vector type has
no portable arithmetic to wrap (see vector_types.cppm), and construction is a
host-callable portable brace -- wwr.vector_types binds every make_* `inline`. So
every constructor is host-callable and wrapped; nothing is device-only to skip.
(vector_types.h's device section pastes the same braces with __device__ for a .cu,
which the C ABI never calls.)

The type set mirrors detail/vector_types_names.h EXACTLY -- the 12 scalar bases
in ranks 1-3, rank 4 only for the bases CUDA 13 does not deprecate (no long /
ulong / longlong / ulonglong rank-4, no double4). The (base, ranks, c_element,
rust_element) table below is the one source for the PODs AND the constructors, so
they cannot drift from each other or from the fragment.
"""

from __future__ import annotations

# (vector-type base name, highest rank present, the C element type each component
# is, that element's Rust spelling). Rank R means ranks 1..R all exist. The
# element type is the make_* parameter type the fragment declares (so no brace
# narrows) -- signed char for char*, long for long* etc.; the Rust spelling is
# what ffi.rs uses for the POD field. Mirrors detail/vector_types_names.h.
BASES = [
    ("char", 4, "signed char", "i8"),
    ("uchar", 4, "unsigned char", "u8"),
    ("short", 4, "short", "i16"),
    ("ushort", 4, "unsigned short", "u16"),
    ("int", 4, "int", "i32"),
    ("uint", 4, "unsigned int", "u32"),
    ("long", 3, "long int", "core::ffi::c_long"),
    ("ulong", 3, "unsigned long int", "core::ffi::c_ulong"),
    ("longlong", 3, "long long int", "core::ffi::c_longlong"),
    ("ulonglong", 3, "unsigned long long int", "core::ffi::c_ulonglong"),
    ("float", 4, "float", "f32"),
    ("double", 3, "double", "f64"),
]

# The four component field-name sets, by rank (CUDA / HIP spell them x, y, z, w).
_COMPONENTS = ["x", "y", "z", "w"]


def types():
    """Every (type_name, c_element, rust_element, [field names]) the surface
    declares -- one POD per (base, rank). The type name is the vendor's own
    (base + rank), re-exported by wwr.vector_types, so the thunk's wwr::<name> is
    exactly this name under namespace wwr."""
    out = []
    for base, max_rank, c_elem, rust_elem in BASES:
        for rank in range(1, max_rank + 1):
            out.append({
                "capi": f"wwr_{base}{rank}",
                "name": f"{base}{rank}",  # the wwr:: type / make_ suffix
                "c_elem": c_elem,
                "rust_elem": rust_elem,
                "fields": _COMPONENTS[:rank],
            })
    return out


def functions():
    """Every make_* constructor, one per declared type: wwr::make_<name> takes
    `rank` scalars of the element type and returns the POD by value. The thunk
    builds the wwr vector value and reinterpret_casts it to the capi POD (§3/§12,
    layout-identical) -- generate.py's vector_types thunk emitter handles it."""
    out = []
    for t in types():
        out.append({
            "capi": f"wwr_make_{t['name']}",
            "wwr": f"wwr::make_{t['name']}",
            "pod": t,
        })
    return out
