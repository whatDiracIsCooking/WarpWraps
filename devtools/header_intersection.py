#!/usr/bin/env python3
"""Intersect a CUDA vendor manifest with its HIP counterpart to find the shared API.

A ``src/<m>.cppm`` wwr* module is only as complete as the intersection of the two
vendor APIs it bridges: every symbol that BOTH cuRAND and hipRAND DECLARE is a
symbol ``wwr.rand`` could carry a ``wwr*`` name for, and any it skips is a
coverage hole. This script computes that intersection from the committed vendor
**manifests** under ``vendor/`` (``vendor_harvest.py``'s output), so "did we cover
everything the two backends agree on?" becomes a diff over data instead of a
manual read of two headers.

READS MANIFESTS, NOT HEADERS -- and needs NO SDK. It used to regex the vendor
``.h`` files at run time, which meant it only ran where the SDK was installed and
only saw whatever a single header's text exposed. It now reads the committed
``vendor/<pin>/<lib>.json`` manifests, whose ``symbols`` block is the DECLARED
surface clang harvested (see ``vendor_harvest.py``). Two consequences:

  * it runs anywhere the repo is checked out -- no CUDA, no ROCm, no compiler --
    which is what lets ``test/shared/test_header_intersection.py`` gate it in CI
    on a bare runner; and
  * ``--coverage`` is intentionally MORE correct than the old header scan, and
    the numbers DIVERGE from it -- see below.

WHY THE NUMBERS DIVERGE FROM THE OLD HEADER SCAN (and why that is better). The
old scan read ONE header's text per side. Many vendor headers are thin umbrellas:
``cublas_v2.h`` is a wrapper over ``cublas_api.h`` that mostly ``#define``s ~270
``_v2`` macro aliases, so a single-file text scan of ``cublas_v2.h`` was blind to
the bulk of the API that ``cublas_api.h`` actually declares. The manifest, built
from the full transitive AST, sees all of it. So the intersection and the
"missing" list are both larger and truer than the old run reported (blas moved
from ~320 shared / 19 missing under the header scan to a materially larger shared
count under the manifest). This divergence is EXPECTED and ACCEPTED: the manifest
answer is the correct one, and reproducing the old shallow numbers is a non-goal.

WHAT COUNTS AS A SYMBOL. The manifest's ``symbols`` are every prefix-carrying
DECLARATION clang found -- functions, typedefs, enum tags, enum constants,
records. Each side's manifest records its own ``prefix`` (``curand`` / ``hiprand``),
so there is no auto-detection here: the harvest already pinned it. To compare
across the two backends we strip that prefix and lowercase the remainder, then
intersect by that normalised key:

    curandCreateGenerator -> "creategenerator" <- hiprandCreateGenerator   (match)
    CURAND_STATUS_SUCCESS  -> "status_success"  <- HIPRAND_STATUS_SUCCESS   (match)

No tag-folding is applied. The old regex tool folded a C struct/enum tag
(``curandGenerator_st``) into its ``_t`` typedef so the raw tag would not read as
an uncovered shared symbol; that is unnecessary here because ``--coverage`` scores
a shared key covered when EITHER backend's literal name appears in the module
source, and a module that wraps ``curandState_t`` names the tag it aliases too --
so both keys score covered without a folding pass that could mask a real gap.

WHAT THE OUTPUT PROVES, AND WHAT IT DOESN'T. The INTERSECTION is the reliable
list: a key only lands there when both manifests declare a name for it, so
internal noise (a method-enum with no HIP twin) self-filters. The CUDA-only /
HIP-only lists explain the "absent for want of a counterpart" notes a module
header carries -- but they are noisier, since a backend's private declarations
show up there too. And matching is by NAME: it confirms a shared NAME exists, not
that the two share a signature or an enum VALUE (that is what ``vendor_harvest.py``
compares, and see ``rand.cppm`` on why the values differ). Read a green
``--coverage`` run as "every shared name is aliased", not "the aliases are
correct" -- that is what dispatch.py and the compiler are for.

``--coverage`` matches by LITERAL name against the module SOURCE, so read its
"missing" list against the module's CONTRACT, which comes in two kinds.

A WHOLE-SURFACE module (rand, fft, tx) promises to wrap everything the two
backends share and spells both names out (``WWR_FUNCTION(gpu, cu, hip)``).
For these ``--coverage`` is the real completeness gate: a nonempty "missing" is a
genuine hole (or a documented omission the module names in its header).

A CURATED-SUBSET module (blas, solver, sparse, runtime_api) lists only the names
the layer above it uses ("Only the names src/wrappers/blas uses are listed") --
it never promised the full intersection, so its "missing" list is
reachable-but-unused vendor symbols, a discovery menu, not a defect report. For
all four, completeness is enforced elsewhere: the compiler (an unresolved wwr*
name cannot be consumed), ``test/shared/alias_coverage.py`` (every alias defined
has a test), and the dispatch tables that ``test/shared/dispatch.py`` checks.

Pass a module's ``.cuh`` alongside its ``.cppm`` when device-side names live there.

Usage:
    # Intersect the DECLARED surfaces of the two rand backends:
    devtools/header_intersection.py \\
        --cuda vendor/cuda-13.0.x/curand.json \\
               vendor/cuda-13.0.x/curand_kernel.json \\
        --hip  vendor/rocm-7.2.4/hiprand.json \\
               vendor/rocm-7.2.4/hiprand_kernel.json

    # Confirm the wwr* rand layer wraps the whole shared surface:
    devtools/header_intersection.py \\
        --cuda vendor/cuda-13.0.x/curand.json vendor/cuda-13.0.x/curand_kernel.json \\
        --hip  vendor/rocm-7.2.4/hiprand.json vendor/rocm-7.2.4/hiprand_kernel.json \\
        --coverage src/rand.cppm src/rand.cuh
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# The AST declaration kinds vendor_harvest emits that are FUNCTIONS (the rest are
# types / constants). Used only to label a symbol's kind for the report.
_FUNC_KINDS = {"FunctionDecl", "FunctionTemplateDecl"}
_CONST_KINDS = {"EnumConstantDecl", "VarDecl"}


def leading_prefix(ident: str) -> str | None:
    """The library token an identifier leads with, lowercased.

    ``curandFoo`` and ``cudaMalloc`` lead with a lowercase run (``curand`` /
    ``cuda``); ``CURAND_FOO`` leads with an all-caps run up to the first
    separator (``CURAND`` -> ``curand``). Both spellings of one library normalise
    to the same token, which is what lets a function and a macro constant compare
    on equal footing. Mirrors ``vendor_harvest.py``'s function of the same name.
    """
    if m := re.match(r"[a-z]+", ident):
        return m.group(0)
    if m := re.match(r"[A-Z]+", ident):
        return m.group(0).lower()
    return None


def _manifest_kind(decl_kind: str) -> str:
    """Collapse an AST decl kind into the report's coarse func/const/type tag."""
    if decl_kind in _FUNC_KINDS:
        return "func"
    if decl_kind in _CONST_KINDS:
        return "const"
    return "type"


class Side:
    """One backend's declared surface, read from one or more committed manifests.

    Each manifest names its own ``prefix`` under ``harvest`` -- the harvest pinned
    it, so nothing is auto-detected here. The declared ``symbols`` are keyed by
    the prefix-stripped, lowercased remainder; a key maps to the actual name(s)
    that produced it and to a coarse kind tag for the report.
    """

    def __init__(self, name: str, paths: list[Path]):
        self.name = name
        self.paths = paths
        # normalised key -> the actual identifier(s) that produced it
        self.by_key: dict[str, set[str]] = {}
        # normalised key -> coarse kind (func/const/type), first-seen wins by
        # sorted name so it is deterministic across manifests.
        self._kind_src: dict[str, tuple[str, str]] = {}
        self.prefixes: list[str] = []
        for path in paths:
            self._load(path)

    def _load(self, path: Path) -> None:
        data = json.loads(path.read_text())
        prefix = data.get("harvest", {}).get("prefix")
        if not prefix:
            raise SystemExit(
                f"header_intersection: {path} has no harvest.prefix; is it a "
                f"vendor_harvest.py manifest?"
            )
        if prefix not in self.prefixes:
            self.prefixes.append(prefix)
        plen = len(prefix)
        for sym in data.get("symbols", ()):
            ident = sym["name"]
            if leading_prefix(ident) != prefix:
                continue
            key = ident[plen:].lstrip("_").lower()
            if not key:
                continue
            self.by_key.setdefault(key, set()).add(ident)
            kind = _manifest_kind(sym.get("kind", ""))
            # Keep the tag from the alphabetically-first name so a key that spans
            # two manifests (a tag here, its typedef there) resolves the same way
            # every run.
            cur = self._kind_src.get(key)
            if cur is None or ident < cur[0]:
                self._kind_src[key] = (ident, kind)

    @property
    def prefix(self) -> str:
        return "+".join(self.prefixes)

    def names(self, key: str) -> list[str]:
        return sorted(self.by_key.get(key, ()))

    def kind(self, key: str) -> str:
        return self._kind_src[key][1]


def tokens(text: str) -> set[str]:
    return set(_IDENT_RE.findall(text))


def build_report(cuda: Side, hip: Side, coverage: list[Path] | None) -> dict:
    """The full comparison as plain data, ready for text or JSON rendering."""
    ckeys, hkeys = set(cuda.by_key), set(hip.by_key)

    def entry(side: Side, key: str) -> dict:
        return {"names": side.names(key), "kind": side.kind(key)}

    intersection = [
        {"key": k, "cuda": entry(cuda, k), "hip": entry(hip, k)}
        for k in sorted(ckeys & hkeys)
    ]
    report = {
        "cuda": {
            "prefix": cuda.prefix,
            "sources": [str(p) for p in cuda.paths],
            "symbols": len(cuda.by_key),
        },
        "hip": {
            "prefix": hip.prefix,
            "sources": [str(p) for p in hip.paths],
            "symbols": len(hip.by_key),
        },
        "intersection": intersection,
        "cuda_only": [
            {"key": k, **entry(cuda, k)} for k in sorted(ckeys - hkeys)
        ],
        "hip_only": [
            {"key": k, **entry(hip, k)} for k in sorted(hkeys - ckeys)
        ],
    }

    if coverage:
        present: set[str] = set()
        for path in coverage:
            present |= tokens(path.read_text())
        missing = [
            item
            for item in intersection
            if not (present & set(item["cuda"]["names"]))
            and not (present & set(item["hip"]["names"]))
        ]
        report["coverage"] = {
            "module": ", ".join(str(p) for p in coverage),
            "intersection": len(intersection),
            "missing": missing,
        }
    return report


def _fmt(side_entry: dict) -> str:
    return "|".join(side_entry["names"])


def render_text(report: dict, show: str) -> str:
    out: list[str] = []
    c, h = report["cuda"], report["hip"]
    out.append(f"CUDA  prefix={c['prefix']!r}  {c['symbols']} symbols  "
               f"({', '.join(c['sources'])})")
    out.append(f"HIP   prefix={h['prefix']!r}  {h['symbols']} symbols  "
               f"({', '.join(h['sources'])})")
    out.append("")

    inter = report["intersection"]
    if show in ("all", "intersection"):
        out.append(f"== Intersection: {len(inter)} shared symbol(s) ==")
        width = max((len(_fmt(i["cuda"])) for i in inter), default=0)
        for item in inter:
            out.append(f"  [{item['cuda']['kind']:5}] "
                       f"{_fmt(item['cuda']):{width}}  {_fmt(item['hip'])}")
        out.append("")

    if show in ("all", "cuda-only"):
        only = report["cuda_only"]
        out.append(f"== CUDA-only: {len(only)} symbol(s) with no HIP counterpart ==")
        for item in only:
            out.append(f"  [{item['kind']:5}] {'|'.join(item['names'])}")
        out.append("")

    if show in ("all", "hip-only"):
        only = report["hip_only"]
        out.append(f"== HIP-only: {len(only)} symbol(s) with no CUDA counterpart ==")
        for item in only:
            out.append(f"  [{item['kind']:5}] {'|'.join(item['names'])}")
        out.append("")

    if "coverage" in report:
        cov = report["coverage"]
        missing = cov["missing"]
        covered = cov["intersection"] - len(missing)
        out.append(f"== Coverage in {cov['module']} ==")
        out.append(f"  {cov['intersection']} shared symbols, "
                   f"{covered} referenced, {len(missing)} MISSING")
        for item in missing:
            out.append(f"    MISSING [{item['cuda']['kind']:5}] "
                       f"{_fmt(item['cuda'])}  {_fmt(item['hip'])}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cuda", nargs="+", type=Path, required=True,
                        metavar="MANIFEST",
                        help="CUDA vendor manifest(s) under vendor/")
    parser.add_argument("--hip", nargs="+", type=Path, required=True,
                        metavar="MANIFEST",
                        help="HIP vendor manifest(s) under vendor/")
    parser.add_argument("--coverage", nargs="+", type=Path, metavar="SRC",
                        help="src file(s) -- e.g. a wwr* module and its .cuh -- to "
                             "check the intersection against by literal name; a "
                             "shared symbol counts as covered when either backend's "
                             "name appears verbatim (so token-pasted dispatch call "
                             "sites, which name no vendor symbol, won't match)")
    parser.add_argument("--show", choices=["all", "intersection",
                                           "cuda-only", "hip-only"],
                        default="intersection", help="which sections to print")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)

    cuda = Side("cuda", args.cuda)
    hip = Side("hip", args.hip)

    report = build_report(cuda, hip, args.coverage)

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report, args.show), end="")

    # Exit 1 when a coverage gap was found, so this can gate in a script.
    if "coverage" in report and report["coverage"]["missing"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
