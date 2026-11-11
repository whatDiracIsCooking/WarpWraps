"""The shape EVERY fragment-sourced vendor-library surface shares.

blas (spec/blas.py) was the first large library surface; blaslt / solver / rand /
rtc (milestone #6) are four more of the SAME shape -- a status/result enum
returned by value, opaque handles, by-value value enums, and (for some) complex /
int-typedef pointer parameters. Rather than restate the blas resolve-and-emit
path per surface, a surface declares a `LibrarySpec` here and generate.py's one
generic library path classifies it from the manifests and emits its header, thunk
and Rust the same way for all of them. captypes.py stays the type classifier; this
is only the per-surface DECLARATION bundle plus the one naming helper they share.

A surface's spec/<x>.py builds a LibrarySpec from the registrations captypes
needs (handles / pointee_pods / value_enums, plus any by-value POD it declares)
and the surface facts generate.py needs (its fragment, its capi prefix, its
manifest pair, the wwr module(s) its thunk imports). `scalar_aliases` folds a
vendor scalar TYPEDEF (cuSOLVER's `cusolver_int_t` -> `int`) to a plain C scalar
so no vendor name leaks into the ABI; `explicit_skips` records a fragment entry
deliberately NOT wrapped (a hand-written inline forwarder that is no
WWR_FUNCTION_RAW macro, so the fragment parser never sees it) with its reason, so
a partial surface is loud, never silent (the milestone's NO SILENT DROPS rule).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LibrarySpec:
    """One fragment-sourced vendor-library surface, as data.

    `name` is the unit name (blas / blaslt / solver / rand / rtc); `capi_prefix`
    is its C symbol/enumerator stem (`wwr_blaslt_`); `cuda_manifest` /
    `hip_manifest` are paths UNDER vendor/; `modules` are the wwr module(s) the
    thunk TU imports (so its .so links only that surface's vendor dep). The
    captypes registrations (`handles` / `pointee_pods` / `value_enums`) are what
    the classifier resolves the surface's vendor types against; `pods` are any
    by-value POD the surface declares in its own header (blas's wwr_half -- a POD
    no OTHER unit uses). `scalar_aliases` normalises a vendor scalar typedef to a
    plain C scalar; `enum_constants_not_in_manifest` exempts a value-enum constant
    that lives in a neighbouring vendor library from the manifest value
    cross-check; `explicit_skips` are (wwr_name, reason) pairs for a fragment
    entry deliberately left unwrapped. `capi_name` maps a fragment wwr name to the
    surface's capi symbol name."""

    name: str
    fragment: str
    capi_prefix: str
    cuda_manifest: str
    hip_manifest: str
    modules: list[str]
    handles: list = field(default_factory=list)
    pointee_pods: list = field(default_factory=list)
    pods: list = field(default_factory=list)
    value_enums: list = field(default_factory=list)
    enum_constants_not_in_manifest: set = field(default_factory=set)
    scalar_aliases: dict = field(default_factory=dict)
    explicit_skips: list = field(default_factory=list)
    capi_name: Callable[[str], str] = None


def snake_capi_name(wwr_name: str, prefix: str, capi_prefix: str) -> str:
    """The capi name for a fragment function: strip the wwr `prefix`, snake_case
    the stem, and re-prefix `capi_prefix`. `wwrblasLtMatmul` / `wwrblasLt` ->
    `wwr_blaslt_matmul`; `wwrrtcCreateProgram` / `wwrrtc` ->
    `wwr_rtc_create_program`. Deterministic and collision-free over a surface
    (the same rule spec/blas.py's capi_name uses, lifted so every surface shares
    it)."""
    stem = wwr_name[len(prefix):]
    out = []
    for i, ch in enumerate(stem):
        prev_lower = i > 0 and not stem[i - 1].isupper()
        next_lower = i + 1 < len(stem) and stem[i + 1].islower()
        if ch.isupper() and i > 0 and (prev_lower or next_lower):
            out.append("_")
        out.append(ch.lower())
    return capi_prefix + "".join(out).lstrip("_")
