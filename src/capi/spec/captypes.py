"""Classify a vendor `decl` into the C ABI's vendor-free representation.

This is the extensible core the generator grows: given a vendor function `decl`
from a manifest (ctypes.parse_function), split it into a return type and a
parameter list, then classify each type (classify) into one of a small set of
capi representations:

  SCALAR   int / float / double / size_t / ...   -> passed through unchanged
  POINTER  T* / const T*                          -> passed through as a pointer
                                                     (vendor pointee -> void* when
                                                     it names a vendor type)
  ENUM     a vendor status/enum type               -> a STABLE capi enum with
                                                     fixed, backend-independent
                                                     values + a `_map` translation
                                                     at the boundary (§6)
  HANDLE   an opaque vendor handle (cublasHandle_t,
           cudaStream_t)                           -> typedef void* wwr_<x>_handle
  STRUCT   a by-value vendor struct                -> a capi POD whose LAYOUT is
                                                     carried in a sidecar (§4/§12:
                                                     a name does not pin the bytes)

The status slice this PR ships exercises SCALAR, POINTER and ENUM (the vendor
return type `cudaError_t` / `hipError_t` is the ENUM, mapped to `wwr_status`).
HANDLE and STRUCT are wired here -- with the sidecar hook -- so #349 (blas) is a
thin addition: register the handle/enum/struct types a new surface needs and the
same classifier emits them. Nothing is hard-coded to the runtime surface.

The classifier is DATA-DRIVEN: a surface declares its enum-maps, handle types and
struct layouts; the vendor type names it resolves are supplied by the caller
(sourced from the fragment), never guessed from a string heuristic. The one
string judgement is scalar-vs-pointer, which C spelling makes unambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# A parsed C function declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    type: str  # the C type spelling, e.g. "int", "int *", "const void *"
    name: str


@dataclass(frozen=True)
class Function:
    ret: str
    params: list[Param]


def parse_function(decl: str) -> Function:
    """Split a vendor function `decl` into return type + parameters.

    `decl` is clang's one-line form, e.g.
    `cudaError_t cudaRuntimeGetVersion(int * runtimeVersion)` or
    `hipError_t hipSetDevice(int deviceId)`. The name immediately precedes the
    `(`; everything before it is the return type, and the parenthesised list is
    the parameters (comma-split, each a `type name` where the last identifier is
    the parameter name)."""
    m = re.match(r"^(.*?)\b([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", decl, re.S)
    if not m:
        raise ValueError(f"cannot parse function decl: {decl!r}")
    ret = m.group(1).strip()
    arglist = m.group(3).strip()
    params: list[Param] = []
    if arglist and arglist != "void":
        for raw in _split_args(arglist):
            params.append(_split_type_name(raw.strip()))
    return Function(ret, params)


def _split_args(arglist: str) -> list[str]:
    """Comma-split a parameter list, not splitting inside <...> or (...)."""
    out, depth, start = [], 0, 0
    for i, ch in enumerate(arglist):
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(arglist[start:i])
            start = i + 1
    out.append(arglist[start:])
    return out


def _split_type_name(text: str) -> Param:
    """Split `int * runtimeVersion` into ("int *", "runtimeVersion"). The last
    bare identifier is the name; the rest (including any `*`) is the type. A
    parameter with no name (a bare type) keeps an empty name."""
    m = re.match(r"^(.*?)([A-Za-z_][A-Za-z0-9_]*)\s*$", text, re.S)
    if not m or not m.group(1).strip():
        # bare type, no parameter name (e.g. "int" alone) -- the whole thing is
        # the type, but a trailing identifier that IS the type (e.g. "int") must
        # not be eaten as a name: only treat it as a name if a type precedes it.
        return Param(text.strip(), "")
    return Param(_normalise(m.group(1)), m.group(2))


def _normalise(ty: str) -> str:
    return re.sub(r"\s+", " ", ty).strip()


# ---------------------------------------------------------------------------
# Type classification -- the extensible, data-driven core
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnumMap:
    """A stable capi enum over a vendor enum type (e.g. cudaError_t/hipError_t ->
    wwr_status). `capi_type` is the generated C enum; `success` and `unknown` are
    its bracket enumerators (first == 0, last == catch-all); `entries` is the
    curated (capi_enumerator, neutral wwr constant) set mapped between them.
    The thunk emits a `_map` comparing the NEUTRAL `wwr::<const>` constants only,
    so the same source compiles on both backends (§6)."""

    vendor_types: tuple[str, ...]  # the backend enum type spellings it matches
    capi_type: str
    success: str
    unknown: str
    entries: tuple[tuple[str, str], ...]  # (capi_enumerator, wwr_constant)


@dataclass(frozen=True)
class Handle:
    """An opaque vendor handle -> `typedef void* capi_type;`. `vendor_types` are
    the backend spellings (e.g. cudaStream_t / hipStream_t) the classifier maps
    to the one vendor-free `capi_type`."""

    vendor_types: tuple[str, ...]
    capi_type: str


@dataclass(frozen=True)
class Struct:
    """A by-value vendor struct -> a capi POD. The LAYOUT is the one datum a name
    cannot express and a `decl` must not be trusted to pin for ABI (§4/§12), so
    it is carried here as an explicit field list -- the architecture.md §25
    sidecar, keyed to the type, not a re-listing of names."""

    vendor_types: tuple[str, ...]
    capi_type: str
    fields: tuple[tuple[str, str], ...]  # (c_type, field_name)


# A classified type: `repr` is one of scalar/pointer/enum/handle/struct, and the
# capi-side spelling the header/thunk/Rust use.
@dataclass(frozen=True)
class Classified:
    repr: str
    capi_c: str  # the C spelling in wwr_capi.h / the thunk signature
    detail: object = None  # the EnumMap / Handle / Struct, when applicable


@dataclass
class Registry:
    """What a surface registers so the classifier can resolve its vendor types.
    Empty by default; a surface (the status slice, later blas) fills it. The
    runtime status slice registers exactly one EnumMap (the error enum)."""

    enums: list[EnumMap] = field(default_factory=list)
    handles: list[Handle] = field(default_factory=list)
    structs: list[Struct] = field(default_factory=list)

    def _find(self, items, vendor_base: str):
        for it in items:
            if vendor_base in it.vendor_types:
                return it
        return None


# C scalar types that pass straight through (the capi spelling == the C spelling).
_SCALARS = {
    "int", "unsigned int", "unsigned", "long", "unsigned long",
    "long long", "unsigned long long", "short", "unsigned short",
    "char", "signed char", "unsigned char", "float", "double",
    "size_t", "unsigned long long int", "int64_t", "uint64_t",
    "int32_t", "uint32_t", "bool",
}


def _base(ty: str) -> str:
    """Strip cv-qualifiers and pointer stars to the underlying type name."""
    t = ty.replace("*", " ").replace("const", " ").replace("volatile", " ")
    return _normalise(t)


def classify(ty: str, reg: Registry) -> Classified:
    """Classify one C type spelling into its capi representation, resolving any
    vendor type against the surface's Registry. Pointers pass through, with a
    vendor pointee erased to `void*`; a registered enum/handle/struct maps to its
    capi form; a bare scalar passes through unchanged."""
    ty = _normalise(ty)
    is_ptr = "*" in ty
    base = _base(ty)

    if is_ptr:
        # A pointer passes through. A pointer to a scalar keeps its element type;
        # a pointer to a vendor type is erased to void* (the data is opaque to the
        # ABI). Const-qualification is preserved in the C spelling.
        if base in _SCALARS:
            return Classified("pointer", ty)
        cv = "const " if "const" in ty else ""
        return Classified("pointer", f"{cv}void *")

    enum = reg._find(reg.enums, base)
    if enum is not None:
        return Classified("enum", enum.capi_type, enum)

    handle = reg._find(reg.handles, base)
    if handle is not None:
        return Classified("handle", handle.capi_type, handle)

    struct = reg._find(reg.structs, base)
    if struct is not None:
        return Classified("struct", struct.capi_type, struct)

    if base in _SCALARS:
        return Classified("scalar", ty)

    raise ValueError(
        f"unclassified type {ty!r}: not a known scalar/pointer and no enum, "
        f"handle or struct is registered for the vendor type {base!r}. Register "
        f"it in the surface's ctypes.Registry (an EnumMap / Handle / Struct) so "
        f"the classifier can emit its vendor-free capi form."
    )
