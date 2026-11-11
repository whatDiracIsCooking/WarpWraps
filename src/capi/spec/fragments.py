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
generator SDK-free so it runs in CI on a bare runner.

A fragment also carries HAND-WRITTEN forwarders that are NOT `WWR_*` macro
entries -- where a backend diverges beyond the cuda/hip prefix a macro reference
cannot bind (`wwrMalloc` names the `void**` overload of an overload set;
`wwrGraphInstantiate` forwards two differently-named backend functions;
`wwrtensorLoggerSetLevel` wraps a per-backend cast). These used to be invisible:
the parser saw only the macro forms, so a hand-written forwarder was silently
absent from the C ABI -- not wrapped, not even logged. The parser now ALSO
recognises the two closed forms these take, carrying each one's EXPLICIT written
signature as a synthetic `decl` so the generator classifies it exactly as it
classifies a macro function's manifest `decl` (there is no manifest entry -- a
forwarder is not a vendor symbol):

  * the reference-to-function form `inline constexpr RET (&wwrX)(ARGS) = ...;`
    (`wwrMalloc`, `wwrMallocManaged`, `wwrHostAlloc`), and
  * the inline-wrapper form `inline RET wwrX(ARGS) { ... }` (`wwrMallocAsync`,
    `wwrGraphInstantiate`, `wwrtensorLoggerSetLevel`, ...).

The written signature is in NEUTRAL `wwr*` spellings (`wwrError_t`,
`wwrStream_t`, `std::size_t`), the same names the surface's registry already
resolves, so no new type machinery is needed -- a forwarder the classifier cannot
represent (a `dim3` kernel-launch parameter, a by-value enum the surface has not
registered) is a LOGGED skip, never a silent drop. A hand-written form whose name
ALSO has a macro twin in the other backend's `#if` branch (the blas Hgemm family,
the sparse / rand `#if`-guarded pairs) keeps the macro entry -- the macro parse
runs first and the hand-written parse never overwrites an existing name.
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
    # A HAND-WRITTEN forwarder carries its explicit written signature here, as a
    # synthetic `RET name(ARGS)` decl in NEUTRAL wwr spellings -- the generator
    # classifies it directly instead of looking the (absent) vendor symbol up in a
    # manifest. None for an ordinary macro entry, whose decl comes from the vendor
    # manifest keyed on `cuda` / `hip`.
    decl: str | None = None


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


# The two HAND-WRITTEN forwarder forms, each carrying its signature literally.
# A NEUTRAL return type (`wwrError_t`, `wwrtensorStatus_t`, `const char *`), the
# `wwr`-prefixed name, and a parenthesised argument list -- the same data a macro
# entry gets from the manifest, written out here because a forwarder is no vendor
# symbol. Both forms are matched to the OPENING `(` of the arg list; the balanced
# tail (the args, then `= ...;` or `{ ... }`) is taken by _handwritten() so a
# nested `(` / `<` in an argument (a function-pointer param, a template arg) does
# not truncate the list. The name must start with `wwr` so an internal helper in a
# `detail` namespace (comp's `backend_opts`) is never taken as a surface entry.
#
#   ref form:    inline constexpr RET (&wwrX)(ARGS) = ...;
#   inline form: inline [constexpr] RET wwrX(ARGS) [noexcept] { ... }
_HW_REF = re.compile(
    r"\binline\s+constexpr\s+(?P<ret>[A-Za-z_][\w:*&<>,\s]*?)\s*"
    r"\(\s*&\s*(?P<name>wwr[A-Za-z0-9_]*)\s*\)\s*\("
)
# The return type is a base (`wwrError_t`, `const char`, `std::int32_t`) followed
# by a pointer/reference + whitespace run, then the `wwr`-prefixed name. Splitting
# the separator as `[\s*&]*\s|\s[\s*&]*` lets the `*` glue to EITHER side (`const
# char *wwrX` -- star on the name -- and `wwrError_t wwrX` -- a plain space) match.
_HW_INLINE = re.compile(
    r"\binline\s+(?:constexpr\s+)?(?P<ret>[A-Za-z_][\w:<>,\s]*?)"
    r"(?P<stars>[\s*&]*\s|\s[\s*&]*)(?P<name>wwr[A-Za-z0-9_]*)\s*\("
)


def _balanced(text: str, open_paren: int) -> tuple[str, int]:
    """The `( ... )` argument list starting AT `open_paren` (the index of the
    `(`), returned without the outer parens, plus the index just past the `)`.
    Tracks `(` / `<` nesting so a function-pointer parameter or a template
    argument inside the list does not close it early."""
    depth, i = 0, open_paren
    while i < len(text):
        ch = text[i]
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1:i], i + 1
        i += 1
    raise ValueError(f"unbalanced argument list at offset {open_paren}")


def _strip_default_args(arglist: str) -> str:
    """Drop C++ default-argument initialisers (`= 0`, `= nullptr`,
    `= wwrMemcpyHostToDevice`) from each parameter -- a default is meaningless to
    a C ABI, which has no default arguments. Split the list on top-level commas so
    a `<...>` or `(...)` inside one parameter is left intact, then cut each at its
    first top-level `=`."""
    out = []
    for raw in _split_top_level(arglist):
        out.append(raw.split("=", 1)[0].strip())
    return ", ".join(p for p in out if p)


def _split_top_level(arglist: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(arglist):
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(arglist[start:i])
            start = i + 1
    parts.append(arglist[start:])
    return parts


def _handwritten(body: str) -> dict[str, Entry]:
    """Every HAND-WRITTEN forwarder in `body`, keyed by its neutral name, each
    carrying its written signature as a synthetic decl. The reference form and the
    inline-wrapper form are scanned together; for each, the argument list is read
    balanced from the matched `(` and its default arguments stripped. cuda / hip
    are set to the wwr name (a forwarder has no vendor symbol -- the generator
    classifies its `decl` and never keys a manifest on these)."""
    out: dict[str, Entry] = {}
    seen: set[int] = set()
    for rx in (_HW_REF, _HW_INLINE):
        for m in rx.finditer(body):
            if m.start() in seen:
                continue
            seen.add(m.start())
            args, _ = _balanced(body, m.end() - 1)
            args = _strip_default_args(" ".join(args.split()))
            # Re-attach any pointer/ref the _HW_INLINE separator split off the
            # return type (`const char *wwrX` -> ret `const char *`); the ref form
            # has no such group. Normalise whitespace so `const char  *` is tidy.
            ret = m.group("ret")
            if "stars" in m.groupdict() and m.group("stars"):
                ret += " " + m.group("stars").replace(" ", "")
            ret = " ".join(ret.split())
            name = m.group("name")
            decl = f"{ret} {name}({args})"
            out.setdefault(name, Entry("function", name, name, name, decl=decl))
    return out


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

    # The hand-written forwarders, added LAST and only when the name is not
    # already a macro entry (setdefault) -- a form with a macro twin in the other
    # backend's `#if` branch (the blas Hgemm family, the sparse / rand `#if`
    # pairs) keeps the macro entry, which the vendor manifest backs.
    for name, entry in _handwritten(body).items():
        out.setdefault(name, entry)

    return out
