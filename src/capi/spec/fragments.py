"""Parse a `src/detail/*_names.h` fragment into its `(wwr, cuda, hip)` tuples.

The fragments are the repo's single canonical encoding of the neutral *name*
surface (architecture.md §25): each `WWR_*_TYPE` / `WWR_*_VALUE` /
`WWR_*_FUNCTION` entry names one type, constant or function, and every consumer
binds to that one list. The C ABI is one more consumer (§25), so it sources its
names here instead of restating them -- delete a name from the fragment and it
leaves the generated header/thunk/`ffi.rs` too.

This reads the fragment as TEXT, the way `devtools/header_intersection.py` reads
the vendor manifests: no clang, no `clang.cindex`, stdlib `re` only. That is
sound because the two spellings the fragment uses are a closed, regular set that
mirrors the preprocessor exactly:

  * the convenience form `WWR_RT_<KIND>(Stem)`, defined (src/runtime_api.cppm,
    src/runtime.h) as `WWR_<KIND>_RAW(wwr##Stem, cuda##Stem, hip##Stem)` -- so
    the tuple is the stem with each backend's prefix glued on; and
  * the explicit form `WWR_<KIND>_RAW(wwrX, cudaX, hipX)`, used where a backend
    name diverges beyond the prefix (HIP's `hipErrorLaunchTimeOut`, the
    version-renamed `hipDeviceProp_tR0600`), which carries all three names
    literally.

Reproducing that expansion in `re` -- rather than invoking a compiler -- is the
same text-over-data tradeoff the manifest tooling already makes, and it keeps the
generator SDK-free so it runs in CI on a bare runner. The hand-written forwarders
in a fragment (e.g. `inline wwrError_t wwrGraphInstantiate(...)`) are NOT
`WWR_*` macro entries and so are not parsed; a capi surface that needs one names
it explicitly (none in the status slice this PR ships).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Each fragment entry, after macro expansion, as the three backend spellings.
# `kind` is "type" | "value" | "function"; `wwr` is the neutral name a thunk
# forwards into; `cuda` / `hip` are the vendor names a manifest is keyed on.


@dataclass(frozen=True)
class Entry:
    kind: str
    wwr: str
    cuda: str
    hip: str


_KINDS = {"TYPE": "type", "VALUE": "value", "FUNCTION": "function"}

# WWR_RT_TYPE(Stem) / WWR_RT_VALUE(Stem) / WWR_RT_FUNCTION(Stem)
_CONV = re.compile(r"\bWWR_RT_(TYPE|VALUE|FUNCTION)\s*\(\s*([A-Za-z0-9_]+)\s*\)")
# WWR_TYPE_RAW(wwrX, cudaX, hipX) and the VALUE / FUNCTION variants
_RAW = re.compile(
    r"\bWWR_(TYPE|VALUE|FUNCTION)_RAW\s*\(\s*"
    r"([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)"
)
# The PLAIN three-argument form WWR_TYPE / WWR_VALUE / WWR_FUNCTION(wwrX, cudaX,
# hipX), used by a surface that defines its names IN its module .cppm against an
# IMPORTED raw vendor module (wwr.ccl over wwr.cuda.nccl) rather than in a
# separate detail/*_names.h against the vendor header's `::` names (the blas
# shape). It carries all three spellings literally, exactly like `_RAW`; the ABI
# generator only needs the vendor column to key the manifest and the wwr column
# to forward into, so sourcing a surface straight from its .cppm keeps that file
# the single source (no duplicate detail/ fragment to drift). The `(?!_RAW)`
# guard stops this from also matching the `_RAW` spellings above.
_PLAIN = re.compile(
    r"\bWWR_(TYPE|VALUE|FUNCTION)(?!_RAW)\s*\(\s*"
    r"([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)"
)


def _strip_comments(text: str) -> str:
    """Drop /* ... */ and // ... so a name mentioned only in prose is not parsed
    as an entry (the fragment's comments cite vendor names freely)."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def parse(fragment: Path) -> dict[str, Entry]:
    """Every `WWR_*` entry in `fragment`, keyed by its neutral `wwr` name. The
    convenience form expands exactly as the macro does (stem + per-backend
    prefix); the `_RAW` and PLAIN three-argument forms are taken literally."""
    body = _strip_comments(fragment.read_text())
    out: dict[str, Entry] = {}

    for kind, stem in _CONV.findall(body):
        k = _KINDS[kind]
        out[f"wwr{stem}"] = Entry(k, f"wwr{stem}", f"cuda{stem}", f"hip{stem}")

    for kind, wwr, cuda, hip in _RAW.findall(body):
        out[wwr] = Entry(_KINDS[kind], wwr, cuda, hip)

    for kind, wwr, cuda, hip in _PLAIN.findall(body):
        out[wwr] = Entry(_KINDS[kind], wwr, cuda, hip)

    return out
