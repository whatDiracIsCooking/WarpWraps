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

Artefacts, PER UNIT so each unit's .so links only its surface's vendor dep
(#A: complex NEEDs no math lib; blas NEEDs cuBLAS -- the isolation the one
libwwr_capi.so used to collapse by linking every surface):

  common.h                    the PODs shared by MORE THAN ONE unit: wwr_complex64
                /             and wwr_complex128 (complex declares them, blas uses
                              them by pointer). A POD used by one unit only stays
                              in that unit's header (wwr_half -> blas; wwr_status
                              -> runtime_api).
  <unit>/wwr_capi_<unit>.h    one vendor-free C header per unit -- its own PODs /
                              enums / handles + its prototypes, #include "common.h"
                              for the shared PODs it uses.
  <unit>/wwr_capi_<unit>.cpp  one extern "C" thunk TU per unit, importing ONLY the
                              wwr module(s) its surface needs (complex: wwr.complex;
                              runtime_api: wwr.runtime_api; blas: wwr.blas +
                              wwr.fp16). Backend-neutral: no vendor header, no
                              CUDA-vs-HIP split; the divergence is resolved inside
                              the wwr call / the _map helper, and the backend is
                              chosen only by which wwr library links.
  wwr_capi.h                  the UMBRELLA header: #include "common.h" + every
                              unit header, so an existing <wwr/wwr_capi.h> consumer
                              is unchanged.
  common.rs     the Rust side, SPLIT PER UNIT to mirror the C split (PR #366).
  <unit>/<unit>.rs  common.rs holds the shared PODs (always compiled); each
                <unit>/<unit>.rs is one surface's #[repr(C)] types + extern "C"
                block, carried behind a Cargo feature and linked (by build.rs)
                to only its own libwwr_capi_<unit>.so. A COMBINED gen/ffi.rs
                (one #[link] onto the umbrella) stays for the in-tree driver.rs.

`complex` is the ONE hand-written surface (spec/complex.py): its thunks forward
into wwr-AUTHORED inline wrappers (wwr::wwrCaddf), which are not vendor functions
and so are in no manifest -- there is no `decl` to source. It stays a documented
special case; everything else is fragment-sourced (spec/runtime_status.py is the
model, and the shape #349 copies).

Run: python3 src/capi/generate.py
"""

from __future__ import annotations

import re
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
# Every C scalar / stdint spelling the surfaces can emit, in a POD struct FIELD
# or a function signature, mapped to its Rust type. Fixed-width stdint types map
# to Rust's fixed-width ints; the width-defined C types (char / long long / size_t)
# map to core::ffi aliases so the width tracks the platform exactly. A base NOT in
# here and not a capi struct/enum would leak a C spelling into ffi.rs -- the clang
# that caught uint16_t -- so an unrecognised C-looking base is a hard error below,
# never a silent passthrough.
_RUST_BASE = {
    "int": "i32", "unsigned int": "u32", "unsigned": "u32",
    "short": "i16", "unsigned short": "u16",
    "long": "core::ffi::c_long", "unsigned long": "core::ffi::c_ulong",
    "long long": "core::ffi::c_longlong",
    "unsigned long long": "core::ffi::c_ulonglong",
    "int8_t": "i8", "uint8_t": "u8", "int16_t": "i16", "uint16_t": "u16",
    "int32_t": "i32", "uint32_t": "u32", "int64_t": "i64", "uint64_t": "u64",
    "size_t": "usize", "ptrdiff_t": "isize", "bool": "bool",
    "float": "f32", "double": "f64",
    "char": "core::ffi::c_char", "signed char": "i8", "unsigned char": "u8",
    "void": "core::ffi::c_void",
}

# A capi-declared type (a POD / handle / value enum) keeps its own name in Rust;
# anything else that is not a known C scalar is the leak _RUST_BASE exists to stop.
_CAPI_BASE = re.compile(r"^(wwr_|wwr::)")


def _rust_base(base_c: str) -> str:
    """The Rust spelling of a BARE (pointer-stripped) capi C base type. A C scalar
    maps through _RUST_BASE; a capi struct/enum/handle (wwr_*) keeps its name;
    anything else is an unmapped C spelling and a hard error (it would otherwise
    leak into ffi.rs -- the uint16_t class of bug, #364 CI)."""
    if base_c in _RUST_BASE:
        return _RUST_BASE[base_c]
    if _CAPI_BASE.match(base_c):
        return base_c
    raise ValueError(
        f"no Rust spelling for C type base {base_c!r}: not a known scalar in "
        f"_RUST_BASE and not a capi wwr_* type. Add it to _RUST_BASE so it does "
        f"not leak a C spelling into ffi.rs."
    )


def _rust_c_type(capi_c: str) -> str:
    """Translate a capi C spelling to Rust. A pointer becomes `*const`/`*mut`
    per level (outer `const` -> `*const`), wrapping the Rust base; a bare scalar
    maps through _rust_base; a capi struct/enum keeps its name (Rust declares it).
    `const wwr_complex64 *const *` -> `*const *const wwr_complex64`."""
    s = captypes._normalise(capi_c)
    if "*" not in s:
        return _rust_base(s)
    head, _, stars = s.partition("*")
    stars = "*" + stars
    base_c = captypes._normalise(head.replace("const", ""))
    base = _rust_base(base_c)
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
# The vendor-free C headers, one per unit + a shared common.h + the umbrella.
# ---------------------------------------------------------------------------
# Each header's provenance line is shared; the per-unit sources differ.
_GEN_NOTE = "// GENERATED by src/capi/generate.py — do not edit."


def _c_extern_open() -> list[str]:
    return ['#ifdef __cplusplus', 'extern "C" {', "#endif", ""]


def _c_extern_close(guard: str) -> list[str]:
    return ["", "#ifdef __cplusplus", "}", "#endif", "", f"#endif  // {guard}", ""]


def gen_common_h() -> str:
    """common.h — the PODs more than one unit uses. Only wwr_complex64 /
    wwr_complex128 qualify today (complex declares them, blas takes them by
    pointer); a one-unit POD (wwr_half, wwr_status) stays in that unit's header.
    No extern "C" block: it declares types only, no prototypes."""
    guard = "WWR_CAPI_COMMON_H"
    L = [f"{_GEN_NOTE} Shared PODs (src/capi/spec/complex.py).",
         f"#ifndef {guard}", f"#define {guard}", ""]
    for t in complex_spec.TYPES:
        L.append(f"typedef struct {t['capi']} {{")
        for cty, fn in t["fields"]:
            L.append(f"  {C_FLOAT[cty]} {fn};")
        L.append(f"}} {t['capi']};")
        L.append("")
    L += [f"#endif  // {guard}", ""]
    return "\n".join(L)


def gen_complex_h() -> str:
    """complex/wwr_capi_complex.h — the complex prototypes. Its PODs are the
    shared wwr_complex64 / wwr_complex128, so it carries none of its own and
    #includes common.h for them."""
    guard = "WWR_CAPI_COMPLEX_H"
    L = [f"{_GEN_NOTE} Sources: src/capi/spec/complex.py (hand-written).",
         f"#ifndef {guard}", f"#define {guard}", "",
         '#include "common.h"', ""]
    L += _c_extern_open()
    for f in complex_spec.FUNCTIONS:
        args = ", ".join(f"{a[0]} {a[1]}" for a in f["args"])
        L.append(f"{f['ret']} {f['capi']}({args});")
    L += _c_extern_close(guard)
    return "\n".join(L)


def gen_runtime_api_h(status_fns: list[ResolvedFunction],
                      enum: captypes.EnumMap) -> str:
    """runtime_api/wwr_capi_runtime_api.h — the stable wwr_status enum (used by
    this unit only) and the status prototypes. No shared POD, so no common.h."""
    guard = "WWR_CAPI_RUNTIME_API_H"
    L = [f"{_GEN_NOTE} Sources: src/detail/runtime_api_names.h + vendor/*/*.json.",
         f"#ifndef {guard}", f"#define {guard}", ""]
    L += _c_extern_open()
    # The stable, backend-independent status enum (architecture.md §6). Fixed
    # integer values -- Rust and C agree on them whichever backend the capi links.
    L.append(f"typedef enum {enum.capi_type} {{")
    for name, val in status_enumerators(enum):
        L.append(f"  {name} = {val},")
    L.append(f"}} {enum.capi_type};")
    L.append("")
    for f in status_fns:
        args = ", ".join(_c_param(c, n) for c, n in f.params)
        L.append(f"{f.ret.capi_c} {f.capi}({args});")
    L += _c_extern_close(guard)
    return "\n".join(L)


def gen_blas_h(blas_fns: list[ResolvedFunction], blas_enums: list) -> str:
    """blas/wwr_capi_blas.h — the blas handles, its own wwr_half POD and value
    enums (all blas-only), and the curated prototypes. #includes common.h for the
    shared wwr_complex64 / wwr_complex128 the pointer parameters name."""
    guard = "WWR_CAPI_BLAS_H"
    L = [f"{_GEN_NOTE} Sources: src/detail/blas_names.h + vendor/*/*.json.",
         f"#ifndef {guard}", f"#define {guard}", "",
         "#include <stdint.h>", "",
         '#include "common.h"', ""]
    L += _c_extern_open()
    # The blas half POD: 16 opaque bits, layout-matched to the vendor __half /
    # hipblasHalf the thunk reinterpret_casts to (§3/§12). It is blas-only, so it
    # lives here, not in common.h.
    L.append(f"typedef struct {blas_spec.HALF_POD['capi']} {{")
    for cty, fn in blas_spec.HALF_POD["fields"]:
        L.append(f"  {cty} {fn};")
    L.append(f"}} {blas_spec.HALF_POD['capi']};")
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
    for f in blas_fns:
        args = ", ".join(_c_param(c, n) for c, n in f.params) or "void"
        L.append(f"{f.ret.capi_c} {f.capi}({args});")
    L += _c_extern_close(guard)
    return "\n".join(L)


def gen_umbrella_h() -> str:
    """wwr_capi.h — the umbrella a non-C++ consumer #includes. It pulls in the
    shared PODs plus every unit header, so <wwr/wwr_capi.h> still declares the
    whole ABI and an existing consumer is unchanged."""
    guard = "WWR_CAPI_H"
    L = [f"{_GEN_NOTE} Umbrella: common.h + every per-unit header.",
         f"#ifndef {guard}", f"#define {guard}", "",
         '#include "common.h"',
         '#include "complex/wwr_capi_complex.h"',
         '#include "runtime_api/wwr_capi_runtime_api.h"',
         '#include "blas/wwr_capi_blas.h"', "",
         f"#endif  // {guard}", ""]
    return "\n".join(L)


def _c_param(c: captypes.Classified, name: str) -> str:
    # Pointer C spellings already carry the `*`; join with a space either way.
    sep = "" if c.capi_c.endswith("*") else " "
    return f"{c.capi_c}{sep}{name}"


# ---------------------------------------------------------------------------
# The backend-neutral extern "C" thunk TUs, one per unit. Each #includes only
# its own unit header and `import`s only the wwr module(s) its surface needs, so
# the per-unit .so links only that surface's vendor dep (the PR's whole point).
# ---------------------------------------------------------------------------
def gen_complex_cpp() -> str:
    """complex/wwr_capi_complex.cpp — the hand-written complex thunks. `import
    wwr.complex;` only, so this unit's .so links no vendor math library."""
    L = [_GEN_NOTE,
         '#include "complex/wwr_capi_complex.h"', "",
         "import wwr.complex;", "",
         'extern "C" {', ""]
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
    L += ["}  // extern \"C\"", ""]
    return "\n".join(L)


def gen_runtime_api_cpp(status_fns: list[ResolvedFunction],
                        enum: captypes.EnumMap) -> str:
    """runtime_api/wwr_capi_runtime_api.cpp — the status thunks. `import
    wwr.runtime_api;` only (the runtime library, no vendor math/BLAS)."""
    L = [_GEN_NOTE,
         '#include "runtime_api/wwr_capi_runtime_api.h"', "",
         "import wwr.runtime_api;", ""]

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
          "}  // namespace",
          "",
          'extern "C" {',
          ""]

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
    L += ["}  // extern \"C\"", ""]
    return "\n".join(L)


def gen_blas_cpp(blas_fns: list[ResolvedFunction], blas_enums: list) -> str:
    """blas/wwr_capi_blas.cpp — the blas thunks. `import wwr.blas; import
    wwr.fp16;`, which transitively carry cuBLAS / hipBLAS, so THIS unit's .so is
    the one that NEEDs the vendor BLAS library."""
    L = [_GEN_NOTE,
         '#include "blas/wwr_capi_blas.h"', "",
         "#include <type_traits>", "",
         "import wwr.blas;", "import wwr.fp16;", "",
         "namespace {"]

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
# The Rust side. PR #366 mirrors the C-side per-unit split (PR #365) into the
# crate: the shared PODs live in an always-compiled `common` module, and each
# surface is its own module behind a Cargo feature (src/capi/README.md). So the
# Rust emits are now PER UNIT -- `common` plus one `.rs` per surface -- while a
# COMBINED gen/ffi.rs stays for the in-tree driver.rs (which include!s one file
# and links the umbrella, exactly as the C driver links the umbrella .so).
#
# A per-unit module carries NO #[link(name=...)]: build.rs emits one
# rustc-link-lib per ENABLED feature (libwwr_capi_<unit>.so), so an opted-down
# build links only its surfaces' .so's -- the Rust-side isolation (a
# complex-only build NEEDs no cuBLAS). The combined ffi.rs keeps its single
# #[link(name="wwr_capi")] onto the umbrella for the standalone driver.
# ---------------------------------------------------------------------------
_RUST_NOTE = "// GENERATED by src/capi/generate.py — do not edit."
_RUST_BACKEND_NOTE = (
    "// Note: NO #[cfg(cuda/hip)] anywhere. The backend is chosen when\n"
    "// the linked libwwr_capi*.so is built; this Rust is identical for either."
)
# `cargo fmt --check` must stay clean, but the extern "C" blocks are generated,
# one declaration per line, and rustfmt would reflow the long blas signatures
# into many lines -- churn in a "do not edit" file. #[rustfmt::skip] leaves the
# generated block exactly as emitted; the hand-written lib.rs / build.rs are the
# formatted surface. Emitted on EVERY extern block so a newly long signature in
# any surface cannot reintroduce the drift.
_RUST_FMT_SKIP = "#[rustfmt::skip]"


def _rust_pod_lines(t: dict) -> list[str]:
    L = ["#[repr(C)]", "#[derive(Copy, Clone, Debug)]",
         f"pub struct {t['capi']} {{"]
    for cty, fn in t["fields"]:
        L.append(f"    pub {fn}: {RUST_FIELD[cty]},")
    L += ["}", ""]
    return L


def _rust_status_enum_lines(enum: captypes.EnumMap) -> list[str]:
    # The stable status enum, #[repr(i32)] so Rust's values match the C enum
    # exactly (architecture.md §6). Rust turns this into a Result.
    L = ["#[repr(i32)]", "#[derive(Copy, Clone, Debug, PartialEq, Eq)]",
         f"pub enum {enum.capi_type} {{"]
    for name, val in status_enumerators(enum):
        L.append(f"    {name} = {val},")
    L += ["}", ""]
    return L


def _rust_blas_types_lines(blas_enums: list) -> list[str]:
    # The blas half POD (transparent 16 bits), opaque handles (-> void*) and the
    # stable value enums (#[repr(i32)], fixed §6 values) -- all blas-only.
    L = ["#[repr(C)]", "#[derive(Copy, Clone, Debug)]",
         f"pub struct {blas_spec.HALF_POD['capi']} {{"]
    for cty, fn in blas_spec.HALF_POD["fields"]:
        L.append(f"    pub {fn}: {_rust_c_type(cty)},")
    L += ["}", ""]
    for h in blas_spec.HANDLES:
        L.append(f"pub type {h.capi_type} = *mut core::ffi::c_void;")
    L.append("")
    for ve in blas_enums:
        L += ["#[repr(i32)]", "#[derive(Copy, Clone, Debug, PartialEq, Eq)]",
              f"pub enum {ve.capi_type} {{"]
        for name, val in blas_enumerators(ve):
            L.append(f"    {name} = {val},")
        L += ["}", ""]
    return L


def _rust_complex_fns(enum: captypes.EnumMap) -> list[str]:
    L = []
    for f in complex_spec.FUNCTIONS:
        args = ", ".join(f"{a[1]}: {_rust_ty(a[0], enum)}" for a in f["args"])
        ret = "" if f["ret"] == "void" else f" -> {_rust_ty(f['ret'], enum)}"
        L.append(f"    pub fn {f['capi']}({args}){ret};")
    return L


def _rust_status_fns(status_fns: list[ResolvedFunction],
                     enum: captypes.EnumMap) -> list[str]:
    L = []
    for f in status_fns:
        args = ", ".join(f"{n}: {_rust_classified(c, enum)}" for c, n in f.params)
        L.append(f"    pub fn {f.capi}({args}) -> {_rust_classified(f.ret, enum)};")
    return L


def _rust_blas_fns(blas_fns: list[ResolvedFunction]) -> list[str]:
    L = []
    for f in blas_fns:
        args = ", ".join(f"{n}: {_rust_c_type(c.capi_c)}" for c, n in f.params)
        L.append(f"    pub fn {f.capi}({args}) -> {_rust_c_type(f.ret.capi_c)};")
    return L


def gen_rust_common() -> str:
    """common.rs — the PODs MORE THAN ONE surface uses: wwr_complex64 /
    wwr_complex128 (complex declares them, blas takes them by pointer), mirroring
    gen/common.h. Always compiled (not feature-gated), so any enabled surface can
    name them via `use crate::common::*;`."""
    L = [_RUST_NOTE,
         "// Shared PODs (src/capi/spec/complex.py), used by > 1 surface.", ""]
    for t in complex_spec.TYPES:
        L += _rust_pod_lines(t)
    return "\n".join(L)


def gen_rust_complex(enum: captypes.EnumMap) -> str:
    """complex/complex.rs — the complex extern "C" block. Its PODs are the shared
    wwr_complex64 / wwr_complex128, so it declares none and `use`s common. No
    #[link]: build.rs links libwwr_capi_complex.so when the `complex` feature is on."""
    L = [_RUST_NOTE, _RUST_BACKEND_NOTE, "",
         "#[allow(unused_imports)]", "use crate::common::*;", "",
         _RUST_FMT_SKIP, 'extern "C" {']
    L += _rust_complex_fns(enum)
    L += ["}", ""]
    return "\n".join(L)


def gen_rust_runtime_api(status_fns: list[ResolvedFunction],
                         enum: captypes.EnumMap) -> str:
    """runtime_api/runtime_api.rs — the stable wwr_status enum (this surface only)
    and the status extern "C" block. No shared POD, so no `use common`."""
    L = [_RUST_NOTE, _RUST_BACKEND_NOTE, ""]
    L += _rust_status_enum_lines(enum)
    L += [_RUST_FMT_SKIP, 'extern "C" {']
    L += _rust_status_fns(status_fns, enum)
    L += ["}", ""]
    return "\n".join(L)


def gen_rust_blas(blas_fns: list[ResolvedFunction], blas_enums: list) -> str:
    """blas/blas.rs — the blas-only types (wwr_half, the handles, the value enums)
    and the blas extern "C" block. `use`s common for the shared complex PODs its
    pointer parameters name."""
    L = [_RUST_NOTE, _RUST_BACKEND_NOTE, "",
         "#[allow(unused_imports)]", "use crate::common::*;", ""]
    L += _rust_blas_types_lines(blas_enums)
    L += [_RUST_FMT_SKIP, 'extern "C" {']
    L += _rust_blas_fns(blas_fns)
    L += ["}", ""]
    return "\n".join(L)


def gen_rust_combined(status_fns: list[ResolvedFunction], enum: captypes.EnumMap,
                      blas_fns: list[ResolvedFunction], blas_enums: list) -> str:
    """gen/ffi.rs — the COMBINED binding the in-tree driver.rs include!s: every
    POD / enum / handle and ONE extern "C" block over the umbrella libwwr_capi.so.
    It is NOT the crate (the crate is the per-unit modules); it is the standalone
    driver's self-contained file, mirroring the C driver linking the umbrella .so."""
    L = [_RUST_NOTE, _RUST_BACKEND_NOTE, ""]
    for t in complex_spec.TYPES:
        L += _rust_pod_lines(t)
    L += _rust_blas_types_lines(blas_enums)
    L += _rust_status_enum_lines(enum)
    # The link name is the OUTPUT file name of the UMBRELLA shared library, NOT
    # the CMake target name: rustc's -l does a literal libwwr_capi.so lookup, so
    # this must match the `OUTPUT_NAME wwr_capi` set on the wwr.capi target in
    # capi/CMakeLists.txt (which also matches the C ABI's wwr_ symbol prefix).
    # Change one and you must change the other, or the Rust driver fails to link.
    L += [_RUST_FMT_SKIP, '#[link(name = "wwr_capi")]', 'extern "C" {']
    L += _rust_complex_fns(enum)
    L += _rust_status_fns(status_fns, enum)
    L += _rust_blas_fns(blas_fns)
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
    # The shared PODs, the umbrella and the still-combined ffi.rs at the gen root.
    (out / "common.h").write_text(gen_common_h())
    (out / "wwr_capi.h").write_text(gen_umbrella_h())
    # Each unit's header + thunk TU in its own subdirectory, so CMake can give
    # each its own .so that links only its surface's vendor dep.
    units = {
        "complex": (gen_complex_h(), gen_complex_cpp()),
        "runtime_api": (gen_runtime_api_h(status_fns, enum),
                        gen_runtime_api_cpp(status_fns, enum)),
        "blas": (gen_blas_h(blas_fns, blas_enums),
                 gen_blas_cpp(blas_fns, blas_enums)),
    }
    # Each unit's C header + thunk TU and its Rust module, all in the unit's own
    # subdirectory. CMake gives each unit its own .so (links only its surface's
    # vendor dep); the crate mirrors that into a per-feature Rust module.
    rust_units = {
        "complex": gen_rust_complex(enum),
        "runtime_api": gen_rust_runtime_api(status_fns, enum),
        "blas": gen_rust_blas(blas_fns, blas_enums),
    }
    for unit, (header, cpp) in units.items():
        unit_dir = out / unit
        unit_dir.mkdir(exist_ok=True)
        (unit_dir / f"wwr_capi_{unit}.h").write_text(header)
        (unit_dir / f"wwr_capi_{unit}.cpp").write_text(cpp)
        (unit_dir / f"{unit}.rs").write_text(rust_units[unit])
    # The always-compiled shared Rust module + the combined ffi.rs the in-tree
    # driver.rs include!s (over the umbrella .so).
    rust_common = gen_rust_common()
    (out / "common.rs").write_text(rust_common)
    combined = gen_rust_combined(status_fns, enum, blas_fns, blas_enums)
    (out / "ffi.rs").write_text(combined)
    print(f"blas: wrapped {len(blas_fns)} functions, skipped {len(SKIPPED)}")
    for name, reason in SKIPPED:
        print(f"  SKIP {name}: {reason}")
    # The wwr-sys crate ships the binding self-contained (it is installed and
    # path/git-depended on away from this tree, so a sibling `../gen/*.rs`
    # include! would dangle). Keep each generated module the SINGLE SOURCE by
    # writing the same bytes into the crate here -- common.rs plus one module per
    # surface, each `mod`-included and feature-gated by src/lib.rs, never
    # hand-edited. Regenerating always rewrites both, so the crate cannot go
    # stale. driver.rs still include!s gen/ffi.rs (combined, rustc-direct,
    # in-tree); the crate is the installed/cargo consumer, per-feature.
    crate_src = Path(__file__).parent / "wwr-sys" / "src"
    if crate_src.is_dir():
        (crate_src / "common.rs").write_text(rust_common)
        for unit, body in rust_units.items():
            (crate_src / f"{unit}.rs").write_text(body)
        print(f"wrote common.h, wwr_capi.h, per-unit .h/.cpp/.rs, common.rs, "
              f"ffi.rs -> {out}; common.rs + per-unit .rs -> {crate_src}")
    else:
        print(f"wrote common.h, wwr_capi.h, per-unit .h/.cpp/.rs, common.rs, "
              f"ffi.rs -> {out}")


if __name__ == "__main__":
    main()
