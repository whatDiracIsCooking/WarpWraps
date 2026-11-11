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
import blaslt as blaslt_spec  # noqa: E402
import captypes  # noqa: E402
import comp as comp_spec  # noqa: E402
import complex as complex_spec  # noqa: E402
import fft as fft_spec  # noqa: E402
import fragments  # noqa: E402
import manifest  # noqa: E402
import rand as rand_spec  # noqa: E402
import rtc as rtc_spec  # noqa: E402
import runtime_status  # noqa: E402
import solver as solver_spec  # noqa: E402
import sparse as sparse_spec  # noqa: E402
import tensor as tensor_spec  # noqa: E402

# Every fragment-sourced vendor-library surface, in emit order. blas was the
# first (spec/blas.py); blaslt / solver / rand / rtc and sparse / tensor / fft
# (milestone #6) are seven more of the SAME shape, each a LibrarySpec
# (spec/library.py) the one generic library path below resolves and emits. Append
# a surface here and it joins the gen/ tree, the umbrella, the combined ffi.rs and
# the crate automatically. (comp is the one hand-written library surface -- its
# functions are wwr-authored shims with no vendor decl to source, like complex --
# so it is NOT here; it has bespoke emitters wired into main() below.)
LIBRARY_SPECS = [blas_spec.SPEC, blaslt_spec.SPEC, solver_spec.SPEC,
                 rand_spec.SPEC, rtc_spec.SPEC, sparse_spec.SPEC,
                 tensor_spec.SPEC, fft_spec.SPEC]

# Opaque handle typedefs MORE THAN ONE surface names -> declared ONCE in the
# shared common.h / common.rs (the way the complex PODs are), so the umbrella
# header and the combined ffi.rs do not redefine them. wwr_stream is the stream
# handle blas / blaslt / solver / rand all pass to a Set/GetStream; a per-unit
# header that names it #includes common.h and does NOT redeclare it. (The C side
# tolerates an identical typedef repeat; Rust does not -- a duplicate `pub type`
# in one scope is an error -- so consolidating is required, not merely tidy.)
SHARED_HANDLES = {"wwr_stream": "typedef void *wwr_stream;"}

_SRC = Path(__file__).resolve().parents[1]  # src/
_VENDOR = _SRC.parent / "vendor"

# The canonical pins the rest of the repo builds / harvests against (vendor
# README): the CUDA 13.0.x band and the ROCm 10.0.0 pin. The status slice's
# names and signatures must resolve identically in BOTH -- the ABI is one shape
# over both backends -- so the generator reads the pair and cross-checks.
_CUDA_MANIFEST = _VENDOR / "cuda-13.0.x" / "cuda_runtime_api.json"
_HIP_MANIFEST = _VENDOR / "rocm-10.0.0" / "hip_runtime_api.json"

# The fragment-sourced library surfaces (blas + the milestone-#6 four) name their
# own manifest pair in each LibrarySpec (paths under vendor/), resolved via
# _VENDOR in _resolve_library -- so no per-surface manifest constant lives here.

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

# Each surface's skips, collected while resolving and reported by main() so a
# partial surface is loud, never silent (the issue's NO SILENT DROPS rule).
# Keyed by surface name -> list of (wwr_name, reason).
SKIPPED: dict[str, list[tuple[str, str]]] = {}


class ResolvedLibrary:
    """One resolved LibrarySpec: its wrapped functions and the value enums the
    emitters need. The enums come straight from the spec (the registry's copy)."""

    def __init__(self, spec, functions: list[ResolvedFunction], enums: list):
        self.spec = spec
        self.functions = functions
        self.enums = enums


def _resolve_library(spec) -> ResolvedLibrary:
    """Resolve every fragment function of a LibrarySpec against its vendor
    manifest pair and classify each with the spec's Registry (handles / pointee
    PODs / value enums / scalar aliases). Cross-checks arity agreement between the
    two backends (a divergence needs a hand-written forwarder; we classify the
    CUDA signature). A function the classifier cannot represent is appended to
    SKIPPED[spec.name] with its reason, as are the spec's explicit skips (a
    fragment entry that is a hand-written inline forwarder, not a WWR_FUNCTION_RAW
    macro, so the fragment parser never sees it) -- never a silent drop."""
    frag = fragments.parse(_SRC / spec.fragment)
    cuda = manifest.Manifest(_VENDOR / spec.cuda_manifest)
    hip = manifest.Manifest(_VENDOR / spec.hip_manifest)

    reg = captypes.Registry(
        value_enums=list(spec.value_enums),
        pointee_pods=list(spec.pointee_pods),
        handles=list(spec.handles),
        scalar_aliases=dict(spec.scalar_aliases),
    )

    skips = list(spec.explicit_skips)
    resolved: list[ResolvedFunction] = []
    for entry in sorted(frag.values(), key=lambda e: e.wwr):
        if entry.kind != "function":
            continue
        sc = cuda.get(entry.cuda)
        sh = hip.get(entry.hip)  # raises if the HIP backend lacks the symbol
        if not sc.decl:
            skips.append((entry.wwr, f"no decl for {entry.cuda} in the manifest"))
            continue
        fc = captypes.parse_function(sc.decl)
        fh = captypes.parse_function(sh.decl) if sh.decl else fc
        if len(fc.params) != len(fh.params):
            skips.append((entry.wwr,
                          f"CUDA {entry.cuda}/{len(fc.params)} and HIP "
                          f"{entry.hip}/{len(fh.params)} disagree on arity"))
            continue
        try:
            ret = captypes.classify(fc.ret, reg)
            params = [(captypes.classify(c.type, reg), c.name) for c in fc.params]
        except ValueError as ex:
            skips.append((entry.wwr, str(ex).split(":")[0]))
            continue
        resolved.append(ResolvedFunction(
            spec.capi_name(entry.wwr), f"wwr::{entry.wwr}", ret, params))

    _check_library_enums(spec, reg.value_enums, frag, cuda, hip)
    SKIPPED[spec.name] = skips
    return ResolvedLibrary(spec, resolved, reg.value_enums)


def _check_library_enums(spec, enums: list, frag: dict, cuda: manifest.Manifest,
                         hip: manifest.Manifest) -> None:
    """Prove each value enum's neutral constants are backed by the manifests, the
    way _check_status_enum does for the runtime error enum. A constant that lives
    in another vendor library (blas's cudaDataType / hipDataType family -- NOT in
    the surface manifest) is exempt via the spec's enum_constants_not_in_manifest:
    the thunk compares the NEUTRAL wwr:: constant, which the fragment carries, and
    never needs the raw manifest value."""
    exempt = spec.enum_constants_not_in_manifest
    for enum in enums:
        for _enumerator, wwr_const in enum.entries:
            e = frag.get(wwr_const)
            if e is None:
                raise KeyError(f"{wwr_const!r} (mapped by {enum.capi_type}) is "
                               f"not a {spec.name} fragment entry.")
            if e.cuda not in exempt:
                cuda.get(e.cuda)
            if e.hip not in exempt:
                hip.get(e.hip)


def library_enumerators(enum: captypes.ValueEnum) -> list[tuple[str, int]]:
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
    leak into ffi.rs -- the uint16_t class of bug, #364 CI). A leading `const` on
    a BY-VALUE scalar (tensor's `const uint32_t numModes`, the compute descriptor
    folded to `const uint64_t`) is meaningless to the ABI and dropped -- Rust has
    no const-value parameter."""
    base_c = re.sub(r"^\s*const\b\s*", "", base_c)
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
    L = [f"{_GEN_NOTE} Shared PODs (src/capi/spec/complex.py) + shared handles.",
         f"#ifndef {guard}", f"#define {guard}", ""]
    for t in complex_spec.TYPES:
        L.append(f"typedef struct {t['capi']} {{")
        for cty, fn in t["fields"]:
            L.append(f"  {C_FLOAT[cty]} {fn};")
        L.append(f"}} {t['capi']};")
        L.append("")
    # The opaque handle typedefs more than one unit names (wwr_stream), declared
    # here once so the umbrella does not see a duplicate.
    for decl in SHARED_HANDLES.values():
        L.append(decl)
    if SHARED_HANDLES:
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


def _spec_shared_handles(spec) -> list:
    """The spec's handles whose capi_type is a SHARED handle (declared in
    common.h), in spec order -- these are filtered from the unit's own header."""
    return [h for h in spec.handles if h.capi_type in SHARED_HANDLES]


def _needs_common(spec) -> bool:
    """Whether a surface names anything that lives in the shared common.h -- a
    complex POD by pointer, or a shared handle (wwr_stream) -- so its header must
    #include "common.h"."""
    shared_pods = {t["capi"] for t in complex_spec.TYPES}
    return (any(p.capi_pod in shared_pods for p in spec.pointee_pods)
            or bool(_spec_shared_handles(spec)))


def gen_library_h(lib: ResolvedLibrary) -> str:
    """<unit>/wwr_capi_<unit>.h — a fragment-sourced library surface's own
    handles, any by-value POD it declares (blas's wwr_half -- a POD no other unit
    uses), its stable value enums, and the curated prototypes. #includes common.h
    only when a pointer parameter names a SHARED complex POD (blas / solver);
    #includes <stdint.h> only when a declared POD has a stdint field type."""
    spec = lib.spec
    guard = f"WWR_CAPI_{spec.name.upper()}_H"
    L = [f"{_GEN_NOTE} Sources: src/{spec.fragment} + vendor/*/*.json.",
         f"#ifndef {guard}", f"#define {guard}", ""]
    # <stddef.h> for size_t (buffer-size / length parameters), <stdint.h> for the
    # fixed-width spellings (int64_t dimensions, uint64_t, a uint16_t POD field).
    # Both are standard and guard-safe, so emit them on every library header
    # rather than scan every prototype for which it happens to use.
    L += ["#include <stddef.h>", "#include <stdint.h>", ""]
    if _needs_common(spec):
        L += ['#include "common.h"', ""]
    L += _c_extern_open()
    # The surface's own by-value PODs (blas's wwr_half: 16 opaque bits,
    # layout-matched to the vendor __half the thunk reinterpret_casts to, §3/§12).
    # A POD used by ONE unit lives in that unit's header, not common.h.
    for t in spec.pods:
        L.append(f"typedef struct {t['capi']} {{")
        for cty, fn in t["fields"]:
            L.append(f"  {cty} {fn};")
        L.append(f"}} {t['capi']};")
        L.append("")
    # The surface's OWN opaque handles (-> void*); a SHARED handle (wwr_stream)
    # lives in common.h (included above), not here. Then the stable value enums.
    own_handles = [h for h in spec.handles if h.capi_type not in SHARED_HANDLES]
    for h in own_handles:
        L.append(f"typedef void *{h.capi_type};")
    if own_handles:
        L.append("")
    for ve in lib.enums:
        L.append(f"typedef enum {ve.capi_type} {{")
        for name, val in library_enumerators(ve):
            L.append(f"  {name} = {val},")
        L.append(f"}} {ve.capi_type};")
        L.append("")
    for f in lib.functions:
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
         '#include "runtime_api/wwr_capi_runtime_api.h"']
    for spec in LIBRARY_SPECS:
        L.append(f'#include "{spec.name}/wwr_capi_{spec.name}.h"')
    L.append('#include "comp/wwr_capi_comp.h"')  # hand-written, not a LibrarySpec
    L += ["", f"#endif  // {guard}", ""]
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


def gen_library_cpp(lib: ResolvedLibrary) -> str:
    """<unit>/wwr_capi_<unit>.cpp — a fragment-sourced library surface's thunks.
    `import`s only the wwr module(s) the spec names (which transitively carry the
    surface's vendor library), so THIS unit's .so is the one that NEEDs that
    vendor dep -- the per-unit isolation. The forwarding core (_coerce / _fwd) and
    the per-enum capi<->neutral maps (_to_<e> / _from_<e>) are identical for every
    library surface; the slug strips the spec's capi_prefix."""
    spec = lib.spec
    blas_fns = lib.functions
    blas_enums = lib.enums
    prefix = spec.capi_prefix
    L = [_GEN_NOTE,
         f'#include "{spec.name}/wwr_capi_{spec.name}.h"', "",
         "#include <type_traits>", ""]
    L += [f"import {m};" for m in spec.modules]
    L += ["", "namespace {"]

    # The forwarding core. `_coerce<P>` turns one capi argument into the exact
    # type the wwr function expects at that position (deduced from the function
    # reference, so NO vendor type is named): a pointer is a C-style cast (the
    # capi POD / handle pointer is layout-matched, §3/§12 -- a plain cast so it
    # covers both the reinterpret AND an added inner cv-qualifier, e.g. the
    # scramble-constant out-pointer whose wwr form is const `const unsigned int **`
    # while the capi exposes the manifest's non-const `unsigned int **`, which a
    # reinterpret_cast alone rejects); a value is static_cast (an enum already
    # mapped to its neutral type below, an int widened). `_fwd` applies it per
    # parameter and forwards.
    L += ["template <class P, class A> P _coerce(A a) {",
          "  if constexpr (std::is_pointer_v<P>) return (P)a;",
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
        slug = ve.capi_type[len(prefix):]
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
                slug = c.capi_c[len(prefix):]
                call_terms.append(f"_to_{slug}({n})")
            elif c.repr == "pointer_enum":
                ve = c.detail
                slug = ve.capi_type[len(prefix):]
                L.append(f"  wwr::{ve.wwr_type} _o_{n};")
                call_terms.append(f"&_o_{n}")
                post.append(f"  if ({n}) *{n} = _from_{slug}(_o_{n});")
            else:
                call_terms.append(n)
        callexpr = f"_fwd({f.wwr}, {', '.join(call_terms)})" if call_terms \
            else f"{f.wwr}()"
        if f.ret.repr == "value_enum":  # the status return
            slug = f.ret.capi_c[len(prefix):]
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
# The comp surface: HAND-WRITTEN (spec/comp.py), like complex -- its functions
# are wwr-authored shims with no manifest `decl`. So it has bespoke emitters (not
# the fragment-sourced library path): by-value option PODs, a status/data-type
# value-enum pair, and thunks that BUILD the neutral wwr opts from the capi POD.
# ---------------------------------------------------------------------------
def _c_param_raw(ctype: str, name: str) -> str:
    sep = "" if ctype.endswith("*") else " "
    return f"{ctype}{sep}{name}"


def _comp_enum_c_lines() -> list[str]:
    L: list[str] = []
    for ve in comp_spec.VALUE_ENUMS:
        L.append(f"typedef enum {ve.capi_type} {{")
        for name, val in library_enumerators(ve):
            L.append(f"  {name} = {val},")
        L.append(f"}} {ve.capi_type};")
        L.append("")
    return L


def gen_comp_h() -> str:
    """comp/wwr_capi_comp.h — the comp status / data-type value enums, the three
    by-value option PODs, plus the 18 batched prototypes. #includes common.h for
    the shared wwr_stream handle the async / stream-taking calls name."""
    guard = "WWR_CAPI_COMP_H"
    L = [f"{_GEN_NOTE} Sources: src/detail/comp_names.h (hand-written spec).",
         f"#ifndef {guard}", f"#define {guard}", "",
         "#include <stddef.h>", "#include <stdint.h>", "",
         '#include "common.h"', ""]
    L += _c_extern_open()
    for h in comp_spec.HANDLES:
        if h.capi_type not in SHARED_HANDLES:
            L.append(f"typedef void *{h.capi_type};")
    L.append("")
    L += _comp_enum_c_lines()
    for o in comp_spec.OPTS:
        L.append(f"typedef struct {o['capi']} {{")
        for cty, fn, _how in o["fields"]:
            L.append(f"  {cty} {fn};")
        L.append(f"}} {o['capi']};")
        L.append("")
    for f in comp_spec.functions():
        args = ", ".join(_c_param_raw(ct, n) for _r, ct, n in f["params"])
        L.append(f"wwr_comp_status {f['capi']}({args});")
    L += _c_extern_close(guard)
    return "\n".join(L)


def gen_comp_cpp() -> str:
    """comp/wwr_capi_comp.cpp — the comp thunks. `import wwr.comp;` only -> this
    unit's .so NEEDs the vendor compression library and no other. Each thunk maps
    the data-type enum fields, BUILDS the neutral wwr opts from the capi POD,
    reinterpret_casts the pointer / stream arguments to the wwr parameter types,
    and maps the status return to the stable enum (§6)."""
    L = [_GEN_NOTE,
         '#include "comp/wwr_capi_comp.h"', "",
         "#include <type_traits>", "",
         "import wwr.comp;", "",
         "namespace {"]
    L += ["template <class P, class A> P _coerce(A a) {",
          "  if constexpr (std::is_pointer_v<P>) "
          "return reinterpret_cast<P>(a);",
          "  else return static_cast<P>(a);",
          "}",
          "template <class R, class... P, class... A>",
          "R _fwd(R (&f)(P...), A... a) { return f(_coerce<P>(a)...); }",
          ""]
    # The status and data-type maps (§6), capi <-> neutral, constant-based so the
    # one source compiles on both backends.
    for ve in comp_spec.VALUE_ENUMS:
        slug = ve.capi_type[len("wwr_comp_"):]
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
    L += ["}  // namespace", "", 'extern "C" {', ""]

    for f in comp_spec.functions():
        args = ", ".join(_c_param_raw(ct, n) for _r, ct, n in f["params"])
        L.append(f"wwr_comp_status {f['capi']}({args}) {{")
        call_terms: list[str] = []
        for repr_, _ct, n in f["params"]:
            if repr_ == "opts":
                opts = f["opts"]
                # Build the neutral wwr opts local from the capi POD's fields: an
                # enum field maps through _to_type, a scalar passes, a reserved
                # byte (Snappy) is ignored -- {} builds the empty wwr opts.
                inits = []
                for _cty, fn, how in opts["fields"]:
                    if how == "enum":
                        inits.append(f"_to_type({n}.{fn})")
                    elif how == "scalar":
                        inits.append(f"{n}.{fn}")
                    # "ignore": contributes nothing
                L.append(f"  wwr::{opts['wwr']} _o_{n}{{{', '.join(inits)}}};")
                call_terms.append(f"_o_{n}")
            else:
                call_terms.append(n)
        callexpr = f"_fwd({f['wwr']}, {', '.join(call_terms)})"
        L.append(f"  return _from_status({callexpr});")
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


def _rust_library_types_lines(lib: ResolvedLibrary) -> list[str]:
    # A library surface's own by-value PODs (blas's wwr_half: transparent 16
    # bits), opaque handles (-> void*) and stable value enums (#[repr(i32)], fixed
    # §6 values) -- all surface-only. A surface with no POD / handle emits none.
    L: list[str] = []
    for t in lib.spec.pods:
        L += ["#[repr(C)]", "#[derive(Copy, Clone, Debug)]",
              f"pub struct {t['capi']} {{"]
        for cty, fn in t["fields"]:
            L.append(f"    pub {fn}: {_rust_c_type(cty)},")
        L += ["}", ""]
    own_handles = [h for h in lib.spec.handles
                   if h.capi_type not in SHARED_HANDLES]
    for h in own_handles:
        L.append(f"pub type {h.capi_type} = *mut core::ffi::c_void;")
    if own_handles:
        L.append("")
    for ve in lib.enums:
        L += ["#[repr(i32)]", "#[derive(Copy, Clone, Debug, PartialEq, Eq)]",
              f"pub enum {ve.capi_type} {{"]
        for name, val in library_enumerators(ve):
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
        args = ", ".join(f"{_rust_ident(n)}: {_rust_classified(c, enum)}"
                         for c, n in f.params)
        L.append(f"    pub fn {f.capi}({args}) -> {_rust_classified(f.ret, enum)};")
    return L


# Rust's reserved words -- a C parameter name that is one (cuBLASLt's
# cublasLtMatrixLayoutCreate takes a param literally named `type`) must become a
# RAW identifier `r#<name>` in the extern "C" signature, or rustc rejects it.
# (`self` / `super` / `crate` / `Self` cannot even be raw, but no vendor
# parameter is named those; `type` is the one that occurs.)
_RUST_KEYWORDS = {
    "as", "break", "const", "continue", "crate", "dyn", "else", "enum",
    "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match",
    "mod", "move", "mut", "pub", "ref", "return", "static", "struct", "super",
    "trait", "true", "type", "unsafe", "use", "where", "while", "async", "await",
    "box", "do", "final", "macro", "override", "priv", "typeof",
    "unsized", "virtual", "yield", "try", "gen",
}


def _rust_ident(name: str) -> str:
    """A C identifier as a Rust identifier, raw-escaping a reserved word."""
    return f"r#{name}" if name in _RUST_KEYWORDS else name


def _rust_library_fns(fns: list[ResolvedFunction]) -> list[str]:
    L = []
    for f in fns:
        args = ", ".join(f"{_rust_ident(n)}: {_rust_c_type(c.capi_c)}"
                         for c, n in f.params)
        L.append(f"    pub fn {f.capi}({args}) -> {_rust_c_type(f.ret.capi_c)};")
    return L


def gen_rust_common() -> str:
    """common.rs — the PODs MORE THAN ONE surface uses: wwr_complex64 /
    wwr_complex128 (complex declares them, blas takes them by pointer), mirroring
    gen/common.h. Always compiled (not feature-gated), so any enabled surface can
    name them via `use crate::common::*;`."""
    L = [_RUST_NOTE,
         "// Shared PODs (src/capi/spec/complex.py) + shared handles, used by > 1 "
         "surface.", ""]
    for t in complex_spec.TYPES:
        L += _rust_pod_lines(t)
    for capi_type in SHARED_HANDLES:
        L.append(f"pub type {capi_type} = *mut core::ffi::c_void;")
    if SHARED_HANDLES:
        L.append("")
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


def gen_rust_library(lib: ResolvedLibrary) -> str:
    """<unit>/<unit>.rs — a library surface's own types (its PODs, handles, value
    enums) and its extern "C" block. `use`s common for the shared complex PODs its
    pointer parameters name (harmless #[allow(unused_imports)] when it names none).
    No #[link]: build.rs links libwwr_capi_<unit>.so when the feature is on."""
    L = [_RUST_NOTE, _RUST_BACKEND_NOTE, "",
         "#[allow(unused_imports)]", "use crate::common::*;", ""]
    L += _rust_library_types_lines(lib)
    L += [_RUST_FMT_SKIP, 'extern "C" {']
    L += _rust_library_fns(lib.functions)
    L += ["}", ""]
    return "\n".join(L)


def _rust_comp_types_lines() -> list[str]:
    # The comp status / data-type value enums and the three option PODs
    # (#[repr(C)]). The shared wwr_stream handle lives in crate::common, so it is
    # skipped here and named via the `use`.
    L: list[str] = []
    own_handles = [h for h in comp_spec.HANDLES
                   if h.capi_type not in SHARED_HANDLES]
    for h in own_handles:
        L.append(f"pub type {h.capi_type} = *mut core::ffi::c_void;")
    if own_handles:
        L.append("")
    for ve in comp_spec.VALUE_ENUMS:
        L += ["#[repr(i32)]", "#[derive(Copy, Clone, Debug, PartialEq, Eq)]",
              f"pub enum {ve.capi_type} {{"]
        for name, val in library_enumerators(ve):
            L.append(f"    {name} = {val},")
        L += ["}", ""]
    for o in comp_spec.OPTS:
        L += ["#[repr(C)]", "#[derive(Copy, Clone, Debug)]",
              f"pub struct {o['capi']} {{"]
        for cty, fn, _how in o["fields"]:
            L.append(f"    pub {_rust_ident(fn)}: {_rust_c_type(cty)},")
        L += ["}", ""]
    return L


def _rust_comp_fns() -> list[str]:
    L = []
    for f in comp_spec.functions():
        args = ", ".join(f"{_rust_ident(n)}: {_rust_c_type(ct)}"
                         for _r, ct, n in f["params"])
        L.append(f"    pub fn {f['capi']}({args}) -> wwr_comp_status;")
    return L


def gen_rust_comp() -> str:
    """comp/comp.rs — the comp types and extern "C" block. `use`s common for the
    shared wwr_stream handle. No #[link]: build.rs links libwwr_capi_comp.so under
    the feature."""
    L = [_RUST_NOTE, _RUST_BACKEND_NOTE, "",
         "#[allow(unused_imports)]", "use crate::common::*;", ""]
    L += _rust_comp_types_lines()
    L += [_RUST_FMT_SKIP, 'extern "C" {']
    L += _rust_comp_fns()
    L += ["}", ""]
    return "\n".join(L)


def gen_rust_combined(status_fns: list[ResolvedFunction], enum: captypes.EnumMap,
                      libs: list[ResolvedLibrary]) -> str:
    """gen/ffi.rs — the COMBINED binding the in-tree driver.rs include!s: every
    POD / enum / handle and ONE extern "C" block over the umbrella libwwr_capi.so.
    It is NOT the crate (the crate is the per-unit modules); it is the standalone
    driver's self-contained file, mirroring the C driver linking the umbrella .so.
    Every library surface's types + functions are declared, in LIBRARY_SPECS
    order; no name collides (each surface prefixes its own)."""
    L = [_RUST_NOTE, _RUST_BACKEND_NOTE, ""]
    for t in complex_spec.TYPES:
        L += _rust_pod_lines(t)
    # The shared handles ONCE (each lib's types skip them -- they live in common).
    for capi_type in SHARED_HANDLES:
        L.append(f"pub type {capi_type} = *mut core::ffi::c_void;")
    if SHARED_HANDLES:
        L.append("")
    for lib in libs:
        L += _rust_library_types_lines(lib)
    L += _rust_comp_types_lines()  # the hand-written comp surface's types
    L += _rust_status_enum_lines(enum)
    # NO `#[link(name = ...)]` on the extern block: the in-tree driver.rs (the one
    # consumer that include!s this file) gets its `-l` flags from capi's
    # CMakeLists, which links ONLY the units driver.rs actually CALLS (complex +
    # runtime_api), NOT the umbrella. A `#[link(name = "wwr_capi")]` here would
    # FORCE the umbrella in regardless of those flags -- and the umbrella NEEDs
    # hiptensor, which probes a GPU on load and aborts on no-GPU HIP CI. An
    # UNREFERENCED extern (the device-library surfaces driver.rs declares but
    # never calls) needs no `.so` at link time, so the two host units suffice.
    L += [_RUST_FMT_SKIP, 'extern "C" {']
    L += _rust_complex_fns(enum)
    L += _rust_status_fns(status_fns, enum)
    for lib in libs:
        L += _rust_library_fns(lib.functions)
    L += _rust_comp_fns()
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
    libs = [_resolve_library(spec) for spec in LIBRARY_SPECS]

    out = Path(__file__).parent / "gen"
    out.mkdir(exist_ok=True)
    # The shared PODs, the umbrella and the still-combined ffi.rs at the gen root.
    (out / "common.h").write_text(gen_common_h())
    (out / "wwr_capi.h").write_text(gen_umbrella_h())
    # Each unit's header + thunk TU in its own subdirectory, so CMake can give
    # each its own .so that links only its surface's vendor dep. The hand-written
    # complex surface and the runtime_api slice stay bespoke; every
    # fragment-sourced library surface (blas + the milestone-#6 four) goes through
    # the one generic library path.
    units = {
        "complex": (gen_complex_h(), gen_complex_cpp()),
        "runtime_api": (gen_runtime_api_h(status_fns, enum),
                        gen_runtime_api_cpp(status_fns, enum)),
        "comp": (gen_comp_h(), gen_comp_cpp()),
    }
    rust_units = {
        "complex": gen_rust_complex(enum),
        "runtime_api": gen_rust_runtime_api(status_fns, enum),
        "comp": gen_rust_comp(),
    }
    for lib in libs:
        units[lib.spec.name] = (gen_library_h(lib), gen_library_cpp(lib))
        rust_units[lib.spec.name] = gen_rust_library(lib)
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
    combined = gen_rust_combined(status_fns, enum, libs)
    (out / "ffi.rs").write_text(combined)
    for lib in libs:
        skips = SKIPPED.get(lib.spec.name, [])
        print(f"{lib.spec.name}: wrapped {len(lib.functions)} functions, "
              f"skipped {len(skips)}")
        for name, reason in skips:
            print(f"  SKIP {name}: {reason}")
    print(f"comp: wrapped {len(comp_spec.functions())} functions (hand-written)")
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
