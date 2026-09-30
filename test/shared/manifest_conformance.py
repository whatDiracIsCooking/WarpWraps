#!/usr/bin/env python3
"""Floor conformance: assert src/<backend> against the committed vendor manifests.

Each ``src/cuda/<lib>.cppm`` / ``src/hip/<lib>.cppm`` re-exports a vendor library
with ``using ::name;`` lines. This checker asserts that hand-written source stays
buildable at the *floor* SDK, using only the committed ``vendor/<pin>/*.json`` and
``vendor/<floor>/*.json`` manifests -- pure text against JSON, no SDK, no compile,
so the fast Python tier runs it on a bare CI runner. It catches exactly the
regression the floor policy exists to prevent: a symbol used unguarded that exists
at the pin but not the floor, which today only a build against the floor SDK finds
-- and CI builds the pin.

WHAT COUNTS AS THIS LIBRARY'S SURFACE. A manifest captures a name iff
``leading_prefix(name) == <the manifest's prefix>`` -- the SAME rule
``devtools/vendor_harvest.py`` harvests by (mirrored here verbatim). So the check
reasons only about ``using ::`` names whose leading token IS the manifest's
prefix. Everything else -- a cross-library type re-export (``cudaStream_t`` in a
curand module), a variant-library constant whose token differs
(``HIPBLASLT_EPILOGUE_*`` -> ``hipblaslt`` != prefix ``hipblas``) -- is not part
of what this manifest claims to cover, so it is out of scope rather than a false
"absent from the manifest". A genuinely bogus ``using ::`` name fails to *compile*
against any SDK, which is not this checker's job.

THE ASSERTION, per in-scope ``using ::name`` (present in the pin or floor
manifest), by whether it sits inside a version guard (``#if WWR_*_SINCE_*`` /
``#if __has_include(<...>)``):

  * unguarded          -> must NOT be pin-only (if in pin, must be in floor).
                          A pin-only name used unguarded breaks the floor build.
  * in the SINCE branch -> must BE pin-only (in pin, not floor). This validates
                          the guard itself: a guard around a name the floor
                          already has is unnecessary, and one around a name not in
                          the pin is wrong. The guards become a machine-checked
                          spec, not just a comment.
  * in the guard's #else -> must be in the floor manifest (that branch is the
                          floor path and compiles there).

ASSERT, NEVER GENERATE. Like ``alias_coverage.py``, this is one of two
independent statements that must agree -- the hand-written ``using ::`` / guard and
the machine-harvested manifest -- so a bad harvest cannot make a bad source pass.
It never emits ``using ::`` lines from the manifest.

KNOWN BLIND SPOT. ``leading_prefix`` keys a name on its leading token, so a
variant library's own uppercase constants (``HIPBLASLT_*`` -> ``hipblaslt``,
``CUBLASLT_*``, ``NVJITLINK_*``, ``ACTIVITY_DOMAIN_*`` ...) are captured by NO
manifest -- the harvester's own limitation, not this checker's. Two such names are
even ``WWR_HIPBLASLT_SINCE_1_2``-guarded floor-sensitive constants this check
therefore cannot validate. It reports the count so the gap is visible rather than
silent; closing it means widening the harvester's prefix rule for variant
libraries (a vendor_harvest.py change, tracked separately).

CUDA is a single point: its floor (13.0.0) and pin (13.0.x) share one manifest
dir, so pin-only is empty there and the check reduces to "no version guard claims
a CUDA symbol postdates the floor" -- correct until the CUDA pin ever rises above
the floor, when a cuda floor manifest appears and this starts biting.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (backend, pin dir, floor dir). CUDA's floor IS its pin -- one within-major band
# serves both (see vendor/README.md), so the two coordinates coincide.
BACKENDS = [
    ("cuda", ROOT / "vendor" / "cuda-13.0.x", ROOT / "vendor" / "cuda-13.0.x"),
    ("hip", ROOT / "vendor" / "rocm-7.2.4", ROOT / "vendor" / "rocm-7.1.0"),
]

# A version guard: a WWR_*_SINCE_* macro (its name encodes the version), or a
# __has_include that gates a header a newer SDK added (hiptensor.h at ROCm 7.2).
_VERSION_GUARD = re.compile(r"WWR_\w*SINCE\w*|__has_include", re.IGNORECASE)

_IF = re.compile(r"^\s*#\s*if(?:def|ndef)?\b")
_ELIF = re.compile(r"^\s*#\s*elif\b")
_ELSE = re.compile(r"^\s*#\s*else\b")
_ENDIF = re.compile(r"^\s*#\s*endif\b")
# `using ::name;` -- the vendor re-export. Anchored past optional leading space;
# a `using ::` in a comment or string is not this shape in these files.
_USING = re.compile(r"^\s*using\s+::([A-Za-z_]\w*)\s*;")


def leading_prefix(ident: str) -> str | None:
    """The library token an identifier leads with, lowercased.

    Verbatim copy of ``devtools/vendor_harvest.py``'s function: a leading
    lowercase run (``hiprandFoo`` -> ``hiprand``), else a leading all-caps run
    lowercased (``HIPRAND_FOO`` -> ``hiprand``). The harvest keeps a name iff this
    equals the prefix, so the check must key membership the same way.
    """
    if m := re.match(r"[a-z]+", ident):
        return m.group(0)
    if m := re.match(r"[A-Z]+", ident):
        return m.group(0).lower()
    return None


@dataclass(frozen=True)
class UsingSite:
    name: str
    since_guarded: bool  # inside the TRUE branch of a version guard
    else_of_guard: bool  # inside the #else of a version guard (the floor path)


def iter_using_sites(text: str):
    """Yield every ``using ::name`` in a .cppm with its version-guard context.

    Tracks a stack of ``#if`` frames, each flagged as a version guard or not and
    as being in its true branch or its ``#else``. A site is ``since_guarded`` when
    any enclosing frame is a version guard in its true branch; ``else_of_guard``
    when the nearest enclosing version guard is in its ``#else``.
    """
    stack: list[dict] = []  # {"vg": bool, "in_true": bool}
    for line in text.splitlines():
        if _IF.match(line):
            stack.append({"vg": bool(_VERSION_GUARD.search(line)), "in_true": True})
        elif _ELIF.match(line):
            if stack:
                # An #elif of a version guard is not the plain floor #else; treat
                # it conservatively as no longer the since-true branch.
                stack[-1]["in_true"] = False
        elif _ELSE.match(line):
            if stack:
                stack[-1]["in_true"] = False
        elif _ENDIF.match(line):
            if stack:
                stack.pop()
        elif m := _USING.match(line):
            since = any(f["vg"] and f["in_true"] for f in stack)
            els = any(f["vg"] and not f["in_true"] for f in stack) and not since
            yield UsingSite(m.group(1), since, els)


def load_manifest(path: Path) -> tuple[str, set[str]]:
    """(prefix, declared-name set) from a committed manifest."""
    j = json.loads(path.read_text())
    return j["harvest"]["prefix"], {s["name"] for s in j["symbols"]}


@dataclass(frozen=True)
class Violation:
    library: str
    name: str
    kind: str  # human-readable rule that was broken
    floor: str  # the floor SDK label, for the message


def check_source(
    text: str, prefix: str, pin: set[str], floor: set[str],
    library: str, floor_label: str,
) -> tuple[list[Violation], int]:
    """Check one module's source against its pin/floor name sets.

    Returns (violations, blind_spot_count). ``blind_spot_count`` is the number of
    version-guarded sites the manifest cannot see (leading token != prefix).
    """
    violations: list[Violation] = []
    blind = 0
    for site in iter_using_sites(text):
        in_scope = leading_prefix(site.name) == prefix
        if not in_scope:
            if site.since_guarded:
                blind += 1
            continue
        in_pin, in_floor = site.name in pin, site.name in floor
        if not in_pin and not in_floor:
            # A prefix-matching name in neither manifest: a macro, or a harvest
            # gap. Not a floor-delta question; the compiler judges its existence.
            continue
        pin_only = in_pin and not in_floor
        if site.since_guarded:
            if not pin_only:
                why = ("guarded but present at the floor (guard unnecessary)"
                       if in_floor else "guarded but absent from the pin (guard wrong)")
                violations.append(Violation(library, site.name, why, floor_label))
        elif site.else_of_guard:
            if not in_floor:
                violations.append(Violation(
                    library, site.name,
                    "in a guard's #else (floor path) but absent from the floor",
                    floor_label))
        elif pin_only:
            violations.append(Violation(
                library, site.name,
                "used unguarded but absent from the floor (pin-only)", floor_label))
    return violations, blind


@dataclass
class Report:
    violations: list[Violation]
    checked: list[str]      # "backend/lib" modules checked
    skipped: list[str]      # src modules with no manifest (type wrappers, cufile)
    blind_spots: int        # version-guarded sites outside any manifest's prefix


def check_all() -> Report:
    violations: list[Violation] = []
    checked: list[str] = []
    skipped: list[str] = []
    blind = 0
    for backend, pin_dir, floor_dir in BACKENDS:
        floor_label = floor_dir.name
        manifests = {p.stem for p in pin_dir.glob("*.json")}
        for cppm in sorted((ROOT / "src" / backend).glob("*.cppm")):
            stem = cppm.stem
            if stem not in manifests:
                skipped.append(f"{backend}/{stem}")
                continue
            prefix, pin = load_manifest(pin_dir / f"{stem}.json")
            _, floor = load_manifest(floor_dir / f"{stem}.json")
            v, b = check_source(
                cppm.read_text(), prefix, pin, floor, f"{backend}/{stem}",
                floor_label)
            violations.extend(v)
            blind += b
            checked.append(f"{backend}/{stem}")
    return Report(violations, checked, skipped, blind)


def format_report(report: Report) -> str:
    """A summary, not 10k lines: per-library counts, the first few each."""
    lines: list[str] = []
    if not report.violations:
        lines.append(
            f"floor conformance OK: {len(report.checked)} modules checked, "
            f"0 violations "
            f"({len(report.skipped)} no-manifest modules skipped, "
            f"{report.blind_spots} guarded sites outside manifest scope).")
        return "\n".join(lines)
    by_lib: dict[str, list[Violation]] = defaultdict(list)
    for v in report.violations:
        by_lib[v.library].append(v)
    lines.append(
        f"floor conformance FAILED: {len(report.violations)} violations in "
        f"{len(by_lib)} modules.")
    for lib in sorted(by_lib):
        vs = by_lib[lib]
        lines.append(f"  {lib}: {len(vs)}")
        for v in vs[:5]:
            lines.append(f"      {v.name}: {v.kind} [floor {v.floor}]")
        if len(vs) > 5:
            lines.append(f"      ... and {len(vs) - 5} more")
    return "\n".join(lines)


def main() -> int:
    argparse.ArgumentParser(
        description="Assert src/<backend> using:: lines against the floor "
                    "vendor manifests (#117).").parse_args()
    report = check_all()
    print(format_report(report))
    return 1 if report.violations else 0


if __name__ == "__main__":
    sys.exit(main())
