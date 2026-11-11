#!/usr/bin/env python3
"""Generate the `wwr` C ABI by SOURCING its surface, not restating it.

Two inputs, three artefacts out, and nothing hand-maintained can drift:

  NAMES come from `src/detail/*_names.h` -- the repo's one canonical encoding of
       the neutral name surface (architecture.md §25). The C ABI is one more
       consumer of that list: delete `WWR_RT_FUNCTION(RuntimeGetVersion)` from
       the fragment and `wwr_runtime_get_version` leaves the generated header,
       thunk and `ffi.rs`. (fragments.py does the parsing.)
  SIGNATURES + enum VALUES come from the committed `vendor/<pin>/<lib>.json`
       manifests (vendor_harvest.py's output, SDK-free, already in CI). The
       fragment gives a function's selected-backend vendor name; the manifest
       gives that symbol's `decl`, which captypes.py classifies into the ABI's
       vendor-free representation (scalars pass through, pointers pass through /
       erase a vendor pointee to void*, the vendor error enum maps to the stable
       wwr_status -- §6). No signature is hand-written. (manifest.py + captypes.py.)

Artefacts:

  wwr_capi.h    vendor-free C header for a C / bindgen consumer. Opaque PODs,
                the stable wwr_status enum, plain-C prototypes, no vendor name.
  wwr_capi.cpp  the extern "C" thunks. `import wwr.complex; import
                wwr.runtime_api;`, convert at the boundary, forward into the wwr
                function. Backend-neutral: no vendor header, no CUDA-vs-HIP
                split; the divergence is resolved inside the wwr call / the _map
                helper, and the backend is chosen only by which wwr library links.
  ffi.rs        the Rust side: one `extern "C"` block, #[repr(C)] PODs, the
                #[repr(i32)] wwr_status enum, ZERO #[cfg].

`complex` is the ONE hand-written surface (spec/complex.py): its thunks forward
into wwr-AUTHORED inline wrappers (wwr::wwrCaddf), which are not vendor functions
and so are in no manifest -- there is no `decl` to source. It stays a documented
special case; everything else is fragment-sourced (spec/runtime_status.py is the
model, and the shape #349 copies).

Run: python3 src/capi/generate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SPEC = Path(__file__).parent / "spec"
sys.path.insert(0, str(_SPEC))

import blas as blas_spec  # noqa: E402
import captypes  # noqa: E402
import complex as complex_spec  # noqa: E402
import fragments  # noqa: E402
import manifest  # noqa: E402
import runtime_status  # noqa: E402

_SRC = Path(__file__).resolve().parents[1]  # src/
_VENDOR = _SRC.parent / "vendor"

# The canonical pins the rest of the repo builds / harvests against (vendor
# README): the CUDA 13.0.x band and the ROCm 10.0.0 pin. The status slice's
# names and signatures must resolve identically in BOTH -- the ABI is one shape
# over both backends -- so the generator reads the pair and cross-checks.
_CUDA_MANIFEST = _VENDOR / "cuda-13.0.x" / "cuda_runtime_api.json"
_HIP_MANIFEST = _VENDOR / "rocm-10.0.0" / "hip_runtime_api.json"

# The blas surface's manifests (the same pins, the cublas/hipblas libraries).
_CUDA_BLAS_MANIFEST = _VENDOR / "cuda-13.0.x" / "cublas_v2.json"
_HIP_BLAS_MANIFEST = _VENDOR / "rocm-10.0.0" / "hipblas.json"

C_FLOAT = {"float": "float", "double": "double"}
RUST_FIELD = {"float": "f32", "double": "f64"}

# C-scalar / pointer capi types -> their Rust spelling. The generated wwr_status
# enum is added per surface below.
RUST_SCALAR = {
    "float": "f32", "double": "f64", "int": "i32",
    "int *": "*mut i32", "unsigned int": "u32",
    "void *": "*mut core::ffi::c_void",
    "const void *": "*const core::ffi::c_void",
}


# ---------------------------------------------------------------------------
# Resolve each fragment-sourced surface against the manifests, once, up front.
# A resolved function carries its capi name, the wwr fn to forward into, and the
# classified return + parameter types -- everything the three emitters need, with
# no vendor name surviving past this step.
# ---------------------------------------------------------------------------
class ResolvedFunction:
    def __init__(self, capi: str, wwr: str, ret: captypes.Classified,
                 params: list[tuple[captypes.Classified, str]]) -> None:
        self.capi = capi
        self.wwr = wwr
        self.ret = ret
        self.params = params


def _resolve_runtime_status() -> tuple[list[ResolvedFunction], captypes.EnumMap]:
    """The runtime status slice: names from the fragment, signatures from the
    manifests, the error enum registered as the one EnumMap. Cross-checks that
    every function and every mapped error constant resolves in BOTH backend
    manifests (the ABI is one shape over both) -- a mismatch is a hard error,
    not a silent skip."""
    frag = fragments.parse(_SRC / runtime_status.FRAGMENT)
    cuda = manifest.Manifest(_CUDA_MANIFEST)
    hip = manifest.Manifest(_HIP_MANIFEST)

    reg = captypes.Registry(enums=[runtime_status.STATUS])

    resolved: list[ResolvedFunction] = []
    for capi_name, wwr_name in runtime_status.FUNCTIONS:
        entry = frag.get(wwr_name)
        if entry is None or entry.kind != "function":
            raise KeyError(
                f"{wwr_name!r} is not a function entry in "
                f"{runtime_status.FRAGMENT}; the status slice names it but the "
                f"fragment does not (check WWR_RT_FUNCTION / _RAW spelling)."
            )
        # The signature is the selected backend's decl; both must parse to the
        # same shape. Resolve against CUDA, then confirm HIP agrees arity-wise.
        fc = captypes.parse_function(cuda.get(entry.cuda).decl)
        fh = captypes.parse_function(hip.get(entry.hip).decl)
        if len(fc.params) != len(fh.params):
            raise ValueError(
                f"{wwr_name}: CUDA {entry.cuda} and HIP {entry.hip} disagree on "
                f"arity ({len(fc.params)} vs {len(fh.params)}); a fragment entry "
                f"this divergent needs a hand-written forwarder, not the "
                f"fragment-sourced path."
            )
        ret = captypes.classify(fc.ret, reg)
        params = [(captypes.classify(c.type, reg), c.name) for c in fc.params]
        resolved.append(ResolvedFunction(capi_name, f"wwr::{wwr_name}", ret, params))

    # Cross-check the enum map: every mapped neutral constant's SUCCESS/error
    # vendor name must exist in both manifests, and the two backends must agree
    # on SUCCESS == 0 (the frozen anchor, §6). The _map stays constant-based
    # (backend-neutral); this only PROVES the manifest backs the mapping.
    _check_status_enum(runtime_status.STATUS, frag, cuda, hip)
    return resolved, runtime_status.STATUS


def _check_status_enum(enum: captypes.EnumMap, frag: dict, cuda: manifest.Manifest,
                       hip: manifest.Manifest) -> None:
    # Success is the frozen 0 anchor; confirm both backends fold it to 0.
    s = frag.get("wwrSuccess")
    if s is None:
        raise KeyError("wwrSuccess is not in the fragment; the status enum "
                       "anchor WWR_STATUS_SUCCESS == 0 cannot be backed.")
    cv, hv = cuda.get(s.cuda).value, hip.get(s.hip).value
    if cv != 0 or hv != 0:
        raise ValueError(f"success value is not 0 on both backends "
                         f"({s.cuda}={cv}, {s.hip}={hv}); §6 anchor broken.")
    for _enumerator, wwr_const in enum.entries:
        e = frag.get(wwr_const)
        if e is None:
            raise KeyError(f"{wwr_const!r} (mapped by {enum.capi_type}) is not a "
                           f"fragment entry; the map cannot cite a name the "
                           f"fragment does not carry.")
        cuda.get(e.cuda)  # raises if the vendor constant is absent
        hip.get(e.hip)


# ---------------------------------------------------------------------------
# The blas surface: the whole curated fragment, classified from the manifests.
# ---------------------------------------------------------------------------

# A surface's skips, collected while resolving and reported by main() so a
# partial surface is loud, never silent (the issue's NO SILENT DROPS rule).
SKIPPED: list[tuple[str, str]] = []


def _resolve_blas() -> tuple[list[ResolvedFunction], list]:
    """Resolve every blas fragment function against the cublas/hipblas manifest
    pair and classify each with blas's Registry. Cross-checks arity agreement
    between the two backends (a divergence needs a hand-written forwarder, as the
    #else branch in the fragment already provides -- and those keep the fragment's
    cuBLAS signature, which is what we classify). A function the classifier cannot
    represent is appended to SKIPPED with its reason, never silently dropped."""
    frag = fragments.parse(_SRC / blas_spec.FRAGMENT)
    cuda = manifest.Manifest(_CUDA_BLAS_MANIFEST)
    hip = manifest.Manifest(_HIP_BLAS_MANIFEST)

    reg = captypes.Registry(
        value_enums=list(blas_spec.VALUE_ENUMS),
        pointee_pods=list(blas_spec.POINTEE_PODS),
        handles=list(blas_spec.HANDLES),
    )

    resolved: list[ResolvedFunction] = []
    for entry in sorted(frag.values(), key=lambda e: e.wwr):
        if entry.kind != "function":
            continue
        sc = cuda.get(entry.cuda)
        sh = hip.get(entry.hip)  # raises if the HIP backend lacks the symbol
        if not sc.decl:
            SKIPPED.append((entry.wwr, f"no decl for {entry.cuda} in the manifest"))
            continue
        fc = captypes.parse_function(sc.decl)
        fh = captypes.parse_function(sh.decl) if sh.decl else fc
        if len(fc.params) != len(fh.params):
            SKIPPED.append((entry.wwr,
                            f"CUDA {entry.cuda}/{len(fc.params)} and HIP "
                            f"{entry.hip}/{len(fh.params)} disagree on arity"))
            continue
        try:
            ret = captypes.classify(fc.ret, reg)
            params = [(captypes.classify(c.type, reg), c.name) for c in fc.params]
        except ValueError as ex:
            SKIPPED.append((entry.wwr, str(ex).split(":")[0]))
            continue
        resolved.append(ResolvedFunction(
            blas_spec.capi_name(entry.wwr), f"wwr::{entry.wwr}", ret, params))

    _check_blas_enums(reg.value_enums, frag, cuda, hip)
    return resolved, reg.value_enums


def _check_blas_enums(enums: list, frag: dict, cuda: manifest.Manifest,
                      hip: manifest.Manifest) -> None:
    """Prove each value enum's neutral constants are backed by the manifests, the
    way _check_status_enum does for the runtime error enum. A constant that lives
    in another vendor library (the cudaDataType / hipDataType family -- NOT in the
    blas manifest) is exempt: the thunk compares the NEUTRAL wwr:: constant, which
    the fragment carries, and never needs the raw blas-manifest value."""
    exempt = blas_spec.ENUM_CONSTANTS_NOT_IN_MANIFEST
    for enum in enums:
        for _enumerator, wwr_const in enum.entries:
            e = frag.get(wwr_const)
            if e is None:
                raise KeyError(f"{wwr_const!r} (mapped by {enum.capi_type}) is "
                               f"not a blas fragment entry.")
            if e.cuda not in exempt:
                cuda.get(e.cuda)
            if e.hip not in exempt:
                hip.get(e.hip)


def blas_enumerators(enum: captypes.ValueEnum) -> list[tuple[str, int]]:
    """A value enum's (enumerator, fixed stable value) list. Values are ascending
    from 0 in `entries` order, with the fallback folded in at its position if it
    is not already an entry -- a frozen §6 contract, independent of backend."""
    names = [c for c, _ in enum.entries]
    if enum.fallback not in names:
        names.append(enum.fallback)
    return [(n, i) for i, n in enumerate(names)]


def _fallback_wwr_const(enum: captypes.ValueEnum) -> str:
    """The NEUTRAL wwr constant `_to_<e>` returns for an unrecognised capi value.
    If the fallback enumerator is itself a mapped entry, its neutral constant; an
    out-of-band fallback (the status enum's UNKNOWN, with no vendor counterpart)
    falls back to the first entry's neutral constant -- a defined, in-range value
    for a slot that otherwise would not have one."""
    by_cap = {c: w for c, w in enum.entries}
    return by_cap.get(enum.fallback, enum.entries[0][1])


# ---------------------------------------------------------------------------
# C type spelling -> Rust type spelling (generalised, so no hand map to drift)
# ---------------------------------------------------------------------------
_RUST_BASE = {
    "int": "i32", "int64_t": "i64", "long long": "core::ffi::c_longlong",
    "unsigned int": "u32", "float": "f32", "double": "f64",
    "char": "core::ffi::c_char", "void": "core::ffi::c_void",
}


def _rust_c_type(capi_c: str) -> str:
    """Translate a capi C spelling to Rust. A pointer becomes `*const`/`*mut`
    per level (outer `const` -> `*const`), wrapping the Rust base; a bare scalar
    maps through _RUST_BASE; a capi struct/enum keeps its name (Rust declares it).
    `const wwr_complex64 *const *` -> `*const *const wwr_complex64`."""
    s = captypes._normalise(capi_c)
    if "*" not in s:
        return _RUST_BASE.get(s, s)
    head, _, stars = s.partition("*")
    stars = "*" + stars
    base_c = captypes._normalise(head.replace("const", ""))
    base = _RUST_BASE.get(base_c, base_c)
    # Walk pointer levels right-to-left; a `const` immediately left of a `*` (or
    # the leading `const` for the innermost) makes that level `*const`.
    levels = stars.count("*")
    segs = stars.split("*")  # segs[i] is the text left of the i-th star
    out = base
    for i in range(levels):
        qual = segs[i]
        is_const = "const" in qual or (i == 0 and "const" in head)
        out = f"{'*const' if is_const else '*mut'} {out}"
    return out


# ---------------------------------------------------------------------------
# Shared helpers over the hand-written complex surface
# ---------------------------------------------------------------------------
def is_pod(name: str) -> bool:
    return any(t["capi"] == name for t in complex_spec.TYPES)


def pod(name: str) -> dict:
    return next(t for t in complex_spec.TYPES if t["capi"] == name)


def status_enumerators(enum: captypes.EnumMap) -> list[tuple[str, int]]:
    """The full (enumerator, value) list with fixed ascending values: SUCCESS==0,
    the curated error set, UNKNOWN last. One source for the C enum, the Rust enum
    and the _map default -- they cannot drift."""
    names = [enum.success, *(c for c, _ in enum.entries), enum.unknown]
    return [(n, i) for i, n in enumerate(names)]


# ---------------------------------------------------------------------------
# wwr_capi.h — the vendor-free C header
# ---------------------------------------------------------------------------
def gen_header(status_fns: list[ResolvedFunction], enum: captypes.EnumMap,
               blas_fns: list[ResolvedFunction], blas_enums: list) -> str:
    L = ["// GENERATED by src/capi/generate.py — do not edit. Sources: "
         "src/capi/spec/complex.py (hand-written), "
         "src/detail/{runtime_api,blas}_names.h + vendor/*/*.json "
         "(fragment-sourced)",
         "#ifndef WWR_CAPI_H", "#define WWR_CAPI_H", "",
         "#include <stdint.h>", "",
         '#ifdef __cplusplus', 'extern "C" {', "#endif", ""]
    for t in complex_spec.TYPES:
        L.append(f"typedef struct {t['capi']} {{")
        for cty, fn in t["fields"]:
            L.append(f"  {C_FLOAT[cty]} {fn};")
        L.append(f"}} {t['capi']};")
        L.append("")
    # The blas half POD: 16 opaque bits, layout-matched to the vendor __half /
    # hipblasHalf the thunk reinterpret_casts to (§3/§12).
    L.append(f"typedef struct {blas_spec.HALF_POD['capi']} {{")
    for cty, fn in blas_spec.HALF_POD["fields"]:
        L.append(f"  {cty} {fn};")
    L.append(f"}} {blas_spec.HALF_POD['capi']};")
    L.append("")
    # The stable, backend-independent status enum (architecture.md §6). Fixed
    # integer values -- Rust and C agree on them whichever backend the capi links.
    L.append(f"typedef enum {enum.capi_type} {{")
    for name, val in status_enumerators(enum):
        L.append(f"  {name} = {val},")
    L.append(f"}} {enum.capi_type};")
    L.append("")
    # The blas opaque handles (-> void*) and stable value enums (§6).
    for h in blas_spec.HANDLES:
        L.append(f"typedef void *{h.capi_type};")
    L.append("")
    for ve in blas_enums:
        L.append(f"typedef enum {ve.capi_type} {{")
        for name, val in blas_enumerators(ve):
            L.append(f"  {name} = {val},")
        L.append(f"}} {ve.capi_type};")
        L.append("")
    for f in complex_spec.FUNCTIONS:
        args = ", ".join(f"{a[0]} {a[1]}" for a in f["args"])
        L.append(f"{f['ret']} {f['capi']}({args});")
    L.append("")
    # The fragment-sourced status prototypes.
    for f in status_fns:
        args = ", ".join(_c_param(c, n) for c, n in f.params)
        L.append(f"{f.ret.capi_c} {f.capi}({args});")
    L.append("")
    # The fragment-sourced blas prototypes (the full curated surface).
    for f in blas_fns:
        args = ", ".join(_c_param(c, n) for c, n in f.params) or "void"
        L.append(f"{f.ret.capi_c} {f.capi}({args});")
    L += ["", "#ifdef __cplusplus", "}", "#endif", "", "#endif  // WWR_CAPI_H", ""]
    return "\n".join(L)


def _c_param(c: captypes.Classified, name: str) -> str:
    # Pointer C spellings already carry the `*`; join with a space either way.
    sep = "" if c.capi_c.endswith("*") else " "
    return f"{c.capi_c}{sep}{name}"


# ---------------------------------------------------------------------------
# wwr_capi.cpp — the backend-neutral extern "C" thunks, forwarding into wwr
# ---------------------------------------------------------------------------
def gen_cpp(status_fns: list[ResolvedFunction], enum: captypes.EnumMap,
            blas_fns: list[ResolvedFunction], blas_enums: list) -> str:
    L = ["// GENERATED by src/capi/generate.py — do not edit.",
         '#include "wwr_capi.h"', "",
         "#include <cstddef>", "#include <type_traits>", "",
         "import wwr.complex;", "import wwr.runtime_api;", "import wwr.blas;",
         "import wwr.fp16;", ""]

    # The backend status -> stable wwr_status map (architecture.md §6). Internal
    # (not extern "C"), driven by the EnumMap, comparing the NEUTRAL wwr::wwr<Name>
    # constants only -- never a vendor name -- so the same thunk source compiles on
    # either backend. Any unrecognised code falls through to the UNKNOWN catch-all.
    L += ["namespace {",
          f"{enum.capi_type} _map(wwr::wwrError_t e) {{",
          f"  if (e == wwr::wwrSuccess) return {enum.success};"]
    for enumerator, wwr_const in enum.entries:
        L.append(f"  if (e == wwr::{wwr_const}) return {enumerator};")
    L += [f"  return {enum.unknown};",
          "}",
          ""]

    # The blas forwarding core. `_coerce<P>` turns one capi argument into the
    # exact type the wwr function expects at that position (deduced from the
    # function reference, so NO vendor type is named): a pointer is
    # reinterpret_cast (the capi POD / handle pointer is layout-matched, §3/§12),
    # a value is static_cast (an enum already mapped to its neutral type below,
    # an int widened). `_fwd` applies it per parameter and forwards.
    L += ["template <class P, class A> P _coerce(A a) {",
          "  if constexpr (std::is_pointer_v<P>) "
          "return reinterpret_cast<P>(a);",
          "  else return static_cast<P>(a);",
          "}",
          "template <class R, class... P, class... A>",
          "R _fwd(R (&f)(P...), A... a) { return f(_coerce<P>(a)...); }",
          ""]

    # Each blas value enum's capi<->neutral maps (§6). `_to_<e>` (capi -> neutral,
    # for a parameter) and `_from_<e>` (neutral -> capi, for a return / out-param)
    # compare and produce only the NEUTRAL wwr::WWRBLAS_* constants, so the one
    # source compiles on both backends though the raw values differ.
    for ve in blas_enums:
        slug = ve.capi_type[len("wwr_blas_"):]
        default_const = _fallback_wwr_const(ve)
        L.append(f"wwr::{ve.wwr_type} _to_{slug}({ve.capi_type} x) {{")
        L.append("  switch (x) {")
        for enumerator, wwr_const in ve.entries:
            L.append(f"    case {enumerator}: return wwr::{wwr_const};")
        L.append(f"    default: return wwr::{default_const};")
        L.append("  }")
        L.append("}")
        L.append(f"{ve.capi_type} _from_{slug}(wwr::{ve.wwr_type} v) {{")
        for enumerator, wwr_const in ve.entries:
            L.append(f"  if (v == wwr::{wwr_const}) return {enumerator};")
        L.append(f"  return {ve.fallback};")
        L.append("}")
    L += ["}  // namespace",
          "",
          'extern "C" {',
          ""]

    # The hand-written complex thunks.
    for f in complex_spec.FUNCTIONS:
        ret = f["ret"]
        args = ", ".join(f"{a[0]} {a[1]}" for a in f["args"])
        L.append(f"{ret} {f['capi']}({args}) {{")
        wlocals = {}
        for aty, an in f["args"]:
            if is_pod(aty):
                wty = pod(aty)["wwr"]
                wn = f"_w_{an}"
                L.append(f"  {wty} {wn}{{{an}.re, {an}.im}};")
                wlocals[an] = wn
        call_args = ", ".join(wlocals.get(a[1], a[1]) for a in f["args"])
        if is_pod(ret):
            L.append(f"  const auto r = {f['wwr']}({call_args});")
            L.append(f"  return {ret}{{r.x, r.y}};")
        else:
            L.append(f"  return {f['wwr']}({call_args});")
        L += ["}", ""]

    # The fragment-sourced status thunks. Each returns the mapped wwr_status;
    # scalars and pointers pass straight through (the classifier confirmed each
    # is scalar/pointer), so the call is a plain forward wrapped in _map.
    for f in status_fns:
        args = ", ".join(_c_param(c, n) for c, n in f.params)
        call = ", ".join(n for _c, n in f.params)
        L.append(f"{f.ret.capi_c} {f.capi}({args}) {{")
        if f.ret.repr == "enum":
            L.append(f"  return _map({f.wwr}({call}));")
        else:
            L.append(f"  return {f.wwr}({call});")
        L += ["}", ""]

    # The fragment-sourced blas thunks. Each maps every value-enum PARAMETER to
    # its neutral type (_to_<e>), forwards through _fwd (which reinterpret_casts
    # the handle / POD pointers to the wwr function's parameter types), and maps
    # the status return back to the stable enum (_from_status). An out-parameter
    # enum pointer is filled through a neutral local, then mapped -- never
    # reinterpret_cast, which would leak the raw backend value (§6).
    for f in blas_fns:
        args = ", ".join(_c_param(c, n) for c, n in f.params) or "void"
        L.append(f"{f.ret.capi_c} {f.capi}({args}) {{")
        call_terms: list[str] = []
        post: list[str] = []
        for c, n in f.params:
            if c.repr == "value_enum":
                slug = c.capi_c[len("wwr_blas_"):]
                call_terms.append(f"_to_{slug}({n})")
            elif c.repr == "pointer_enum":
                ve = c.detail
                slug = ve.capi_type[len("wwr_blas_"):]
                L.append(f"  wwr::{ve.wwr_type} _o_{n};")
                call_terms.append(f"&_o_{n}")
                post.append(f"  if ({n}) *{n} = _from_{slug}(_o_{n});")
            else:
                call_terms.append(n)
        callexpr = f"_fwd({f.wwr}, {', '.join(call_terms)})" if call_terms \
            else f"{f.wwr}()"
        if f.ret.repr == "value_enum":  # the status return
            slug = f.ret.capi_c[len("wwr_blas_"):]
            if post:
                L.append(f"  const auto _r = {callexpr};")
                L += post
                L.append(f"  return _from_{slug}(_r);")
            else:
                L.append(f"  return _from_{slug}({callexpr});")
        else:  # a non-status return (const char * from GetStatusName/String)
            L.append(f"  return {callexpr};")
        L += ["}", ""]

    L += ["}  // extern \"C\"", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# ffi.rs — the Rust side: one extern "C" block, #[repr(C)] PODs, ZERO cfg
# ---------------------------------------------------------------------------
def gen_rust(status_fns: list[ResolvedFunction], enum: captypes.EnumMap,
             blas_fns: list[ResolvedFunction], blas_enums: list) -> str:
    L = ["// GENERATED by src/capi/generate.py — do not edit.",
         "// Note: NO #[cfg(cuda/hip)] anywhere. The backend is chosen when",
         "// libwwr.capi is built; this Rust is identical for either.", ""]
    for t in complex_spec.TYPES:
        L.append("#[repr(C)]")
        L.append("#[derive(Copy, Clone, Debug)]")
        L.append(f"pub struct {t['capi']} {{")
        for cty, fn in t["fields"]:
            L.append(f"    pub {fn}: {RUST_FIELD[cty]},")
        L.append("}")
        L.append("")
    # The blas half POD (transparent 16 bits).
    L.append("#[repr(C)]")
    L.append("#[derive(Copy, Clone, Debug)]")
    L.append(f"pub struct {blas_spec.HALF_POD['capi']} {{")
    for cty, fn in blas_spec.HALF_POD["fields"]:
        L.append(f"    pub {fn}: {_rust_c_type(cty)},")
    L.append("}")
    L.append("")
    # The blas opaque handles -> raw void pointers (one type alias each).
    for h in blas_spec.HANDLES:
        L.append(f"pub type {h.capi_type} = *mut core::ffi::c_void;")
    L.append("")
    # The stable status enum, #[repr(i32)] so Rust's values match the C enum
    # exactly (architecture.md §6). Rust turns this into a Result.
    L.append("#[repr(i32)]")
    L.append("#[derive(Copy, Clone, Debug, PartialEq, Eq)]")
    L.append(f"pub enum {enum.capi_type} {{")
    for name, val in status_enumerators(enum):
        L.append(f"    {name} = {val},")
    L.append("}")
    L.append("")
    # The blas stable value enums, #[repr(i32)], fixed §6 values.
    for ve in blas_enums:
        L.append("#[repr(i32)]")
        L.append("#[derive(Copy, Clone, Debug, PartialEq, Eq)]")
        L.append(f"pub enum {ve.capi_type} {{")
        for name, val in blas_enumerators(ve):
            L.append(f"    {name} = {val},")
        L.append("}")
        L.append("")
    # The link name is the OUTPUT file name of the shared library, NOT the CMake
    # target name: rustc's -l does a literal libwwr_capi.so lookup, so this must
    # match the `OUTPUT_NAME wwr_capi` set on the wwr.capi target in
    # capi/CMakeLists.txt (which also matches the C ABI's wwr_ symbol prefix).
    # Change one and you must change the other, or the Rust driver fails to link.
    L += ['#[link(name = "wwr_capi")]', 'extern "C" {']
    for f in complex_spec.FUNCTIONS:
        args = ", ".join(f"{a[1]}: {_rust_ty(a[0], enum)}" for a in f["args"])
        ret = "" if f["ret"] == "void" else f" -> {_rust_ty(f['ret'], enum)}"
        L.append(f"    pub fn {f['capi']}({args}){ret};")
    for f in status_fns:
        args = ", ".join(f"{n}: {_rust_classified(c, enum)}" for c, n in f.params)
        L.append(f"    pub fn {f.capi}({args}) -> {_rust_classified(f.ret, enum)};")
    for f in blas_fns:
        args = ", ".join(f"{n}: {_rust_c_type(c.capi_c)}" for c, n in f.params)
        L.append(f"    pub fn {f.capi}({args}) -> {_rust_c_type(f.ret.capi_c)};")
    L += ["}", ""]
    return "\n".join(L)


def _rust_ty(name: str, enum: captypes.EnumMap) -> str:
    if is_pod(name):
        return name
    if name == enum.capi_type:
        return enum.capi_type
    return RUST_SCALAR[name]


def _rust_classified(c: captypes.Classified, enum: captypes.EnumMap) -> str:
    if c.repr == "enum":
        return enum.capi_type
    return RUST_SCALAR[c.capi_c]


def main() -> None:
    status_fns, enum = _resolve_runtime_status()
    blas_fns, blas_enums = _resolve_blas()

    out = Path(__file__).parent / "gen"
    out.mkdir(exist_ok=True)
    (out / "wwr_capi.h").write_text(
        gen_header(status_fns, enum, blas_fns, blas_enums))
    (out / "wwr_capi.cpp").write_text(
        gen_cpp(status_fns, enum, blas_fns, blas_enums))
    rust = gen_rust(status_fns, enum, blas_fns, blas_enums)
    (out / "ffi.rs").write_text(rust)
    print(f"blas: wrapped {len(blas_fns)} functions, skipped {len(SKIPPED)}")
    for name, reason in SKIPPED:
        print(f"  SKIP {name}: {reason}")
    # The wwr-sys crate ships the binding self-contained (it is installed and
    # path/git-depended on away from this tree, so a sibling `../gen/ffi.rs`
    # include! would dangle). Keep ffi.rs the SINGLE SOURCE by writing the same
    # bytes into the crate here -- `src/lib.rs` only include!s this file, never
    # hand-edited. Regenerating always rewrites both, so the crate cannot go
    # stale. driver.rs still include!s gen/ffi.rs via WWR_CAPI_FFI (rustc-direct,
    # in-tree); the crate is the installed/cargo consumer. Identical content.
    crate_ffi = Path(__file__).parent / "wwr-sys" / "src" / "ffi.rs"
    if crate_ffi.parent.is_dir():
        crate_ffi.write_text(rust)
        print(f"wrote wwr_capi.h, wwr_capi.cpp, ffi.rs -> {out}; "
              f"ffi.rs -> {crate_ffi.parent}")
    else:
        print(f"wrote wwr_capi.h, wwr_capi.cpp, ffi.rs -> {out}")


if __name__ == "__main__":
    main()
