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

The runtime status slice exercises SCALAR, POINTER and the status EnumMap (the
vendor return type `cudaError_t` / `hipError_t`, mapped to `wwr_status`). blas
(#349) exercises the rest: HANDLE (`cublasHandle_t`/`cudaStream_t` -> void*),
a VALUE ENUM carried BOTH ways across the boundary (`cublasOperation_t` and the
rest -- a stable capi enum the thunk maps capi<->neutral, see ValueEnum), and a
POINTEE POD (`cuComplex *` -> `wwr_complex64 *`, `__half *` -> `wwr_half *`: a
vendor pointee that keeps a typed, layout-matched capi POD instead of erasing to
void*, see PointeePOD). Register the types a surface needs and the same
classifier emits them; nothing is hard-coded to one surface.

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
    # A bare multi-word SCALAR type with no parameter name (`unsigned int`,
    # `long long`, `unsigned long long` -- a hand-written forwarder's unnamed
    # argument) must be kept whole: its last word (`int`, `long`) is part of the
    # type, not a parameter name. Only split off a name when what remains is NOT
    # itself a complete scalar spelling.
    if _normalise(text) in _SCALARS:
        return Param(_normalise(text), "")
    m = re.match(r"^(.*?)([A-Za-z_][A-Za-z0-9_]*)\s*$", text, re.S)
    if not m or not m.group(1).strip() or m.group(1).rstrip().endswith("::"):
        # bare type, no parameter name (e.g. "int" alone) -- the whole thing is
        # the type, but a trailing identifier that IS the type (e.g. "int") must
        # not be eaten as a name: only treat it as a name if a type precedes it. A
        # trailing `::` means the identifier is the tail of a qualified TYPE name
        # (`std::size_t` with no parameter name), not a parameter name.
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
class ValueEnum:
    """A stable capi enum over a vendor enum passed/returned BY VALUE (not an
    error status -- cublasOperation_t and the FillMode / DiagType / SideMode /
    PointerMode / compute / datatype / algo family blas forwards). Unlike the
    status EnumMap (neutral -> stable, one direction, a return), this is a 1:1
    map the thunk crosses BOTH ways: capi -> neutral for a parameter, neutral ->
    capi for a return, with an explicit `fallback` enumerator for an unrecognised
    value. `entries` are (capi_enumerator, neutral wwr constant) pairs and the
    stable capi VALUES are assigned fixed ascending integers at emit time (the
    §6 frozen contract); the raw backend value never crosses the boundary, so the
    two maps compare only the neutral `wwr::<const>` constants and the one source
    compiles on both backends."""

    vendor_types: tuple[str, ...]
    capi_type: str
    wwr_type: str  # the neutral wwr type the maps convert to/from (wwrblasOperation_t)
    entries: tuple[tuple[str, str], ...]  # (capi_enumerator, wwr_constant)
    fallback: str  # the capi_enumerator returned for an unrecognised value


@dataclass(frozen=True)
class PointeePOD:
    """A vendor type that, AS A POINTER, keeps a typed capi POD rather than the
    `void*` the generic pointer arm erases a vendor pointee to. blas's complex
    (`cuComplex *` -> `wwr_complex64 *`) and half (`__half *` -> `wwr_half *`)
    pointer parameters: the POD is layout-identical to the neutral type the wwr
    function names, so the thunk `reinterpret_cast`s at the boundary (§3/§12) and
    Rust sees a typed `*mut`/`*const` POD, not an opaque pointer. `capi_pod` is a
    POD the surface also declares (spec-side), so the header/ffi carry it."""

    vendor_types: tuple[str, ...]
    capi_pod: str  # the capi POD struct name (without the `*`)


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
    value_enums: list[ValueEnum] = field(default_factory=list)
    pointee_pods: list[PointeePOD] = field(default_factory=list)
    handles: list[Handle] = field(default_factory=list)
    structs: list[Struct] = field(default_factory=list)
    # A vendor scalar TYPEDEF folded to a plain C scalar BEFORE classification,
    # so no vendor name leaks into the ABI (cuSOLVER's `cusolver_int_t` ->
    # `int`): keyed on the stripped base spelling, value is the C scalar it IS.
    # Applied by classify(), preserving pointer/const (`cusolver_int_t *` ->
    # `int *`). Only for a typedef that is genuinely one C scalar; a struct /
    # handle / enum stays a Handle / Struct / ValueEnum registration.
    scalar_aliases: dict[str, str] = field(default_factory=dict)

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
    "int32_t", "uint32_t", "bool", "long long int",
}


def _base(ty: str) -> str:
    """Strip cv-qualifiers, pointer stars and a leading elaborated-type-specifier
    keyword (`enum` / `struct` / `union`) to the underlying type name. clang
    records a by-value enum / struct parameter with the keyword
    (`enum cudaMemcpyKind`, `struct cudaExtent`); the registry keys on the bare
    type name (`cudaMemcpyKind`), so the keyword must not survive into the base
    or a registered enum/struct would never match."""
    t = ty.replace("*", " ").replace("const", " ").replace("volatile", " ")
    t = re.sub(r"\b(?:enum|struct|union)\b", " ", t)
    return _normalise(t)


def _reptr(ty: str, capi_base: str) -> str:
    """Rebuild a pointer type with its vendor base swapped for `capi_base`,
    preserving the trailing `*`/`const` run so a double pointer stays a double
    pointer. `cuComplex *const *` with base `wwr_complex64` -> `wwr_complex64
    *const *`; `const cuComplex *` with the leading const already in `capi_base`
    -> `const wwr_complex64 *`. The pointer tail is everything from the first `*`
    on, minus a leading `const` the caller folded into `capi_base`."""
    tail = ty[ty.index("*"):]
    tail = re.sub(r"^\s*const\b\s*", "", tail) if capi_base.startswith("const") \
        else tail
    return _normalise(f"{capi_base} {tail}")


def classify(ty: str, reg: Registry) -> Classified:
    """Classify one C type spelling into its capi representation, resolving any
    vendor type against the surface's Registry. Pointers pass through, with a
    vendor pointee erased to `void*`; a registered enum/handle/struct maps to its
    capi form; a bare scalar passes through unchanged."""
    ty = _normalise(ty)
    # A hand-written forwarder writes the C++ std spellings (`std::size_t`,
    # `std::int32_t`); the C ABI knows them only by their stdint/stddef names, and
    # `std::size_t` IS `size_t` by definition. Drop the `std::` qualifier so the
    # scalar table and the Rust map see the plain C name (a vendor manifest decl
    # never carries `std::`, so this only ever fires for a hand-written decl).
    ty = re.sub(r"\bstd::", "", ty)
    base = _base(ty)
    # Fold a registered vendor scalar typedef (`cusolver_int_t` -> `int`) in
    # place, preserving pointer depth / const, so the rest of the classifier
    # (and the ABI) sees only the plain C scalar, never the vendor name.
    if base in reg.scalar_aliases:
        ty = _normalise(re.sub(rf"\b{re.escape(base)}\b", reg.scalar_aliases[base], ty))
        base = _base(ty)
    is_ptr = "*" in ty
    # A CONST POINTEE is a `const` before the first `*` (`const T *`); a `const`
    # AFTER a `*` is a const-qualified POINTER level (`T *const *`), not a const
    # pointee, and must not be folded into the base spelling.
    cv = "const " if "const" in ty.split("*", 1)[0] else ""

    if is_ptr:
        # A pointer to a scalar keeps its element type; a registered POINTEE POD
        # (complex / half) keeps a typed capi POD; a registered handle or value
        # enum keeps a pointer to its capi form (an out-parameter: handle out of
        # Create, stream out of GetStream, mode out of GetPointerMode); any other
        # vendor pointee is erased to void* (opaque to the ABI). Const-qualification
        # and pointer depth are preserved in the C spelling.
        if base in _SCALARS:
            return Classified("pointer", ty)
        pod = reg._find(reg.pointee_pods, base)
        if pod is not None:
            return Classified("pointer_pod",
                              _reptr(ty, f"{cv}{pod.capi_pod}"), pod)
        handle = reg._find(reg.handles, base)
        if handle is not None:
            return Classified("pointer_handle",
                              _reptr(ty, f"{cv}{handle.capi_type}"), handle)
        venum = reg._find(reg.value_enums, base)
        if venum is not None:
            return Classified("pointer_enum",
                              _reptr(ty, f"{cv}{venum.capi_type}"), venum)
        # Any other vendor pointee erases to void*, but the POINTER DEPTH is
        # preserved: a single pointer (`cudaStream_t *`) erases to `void *`, a
        # double pointer (`void **` out-parameter of cudaMalloc / cudaHostAlloc,
        # `T **` of an overload-set forwarder) erases to `void **` -- NOT `void *`,
        # which would silently drop an indirection level and hand Rust a `*mut
        # c_void` where it needs `*mut *mut c_void`. `_reptr` with base `void`
        # rebuilds the exact star run.
        stars = ty.count("*")
        if stars > 1:
            return Classified("pointer", _reptr(ty, f"{cv}void"))
        return Classified("pointer", f"{cv}void *")

    enum = reg._find(reg.enums, base)
    if enum is not None:
        return Classified("enum", enum.capi_type, enum)

    venum = reg._find(reg.value_enums, base)
    if venum is not None:
        return Classified("value_enum", venum.capi_type, venum)

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
