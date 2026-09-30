#!/usr/bin/env python3
"""Manifest conformance: assert src/<backend> against the committed vendor manifests.

Each ``src/cuda/<lib>.cppm`` / ``src/hip/<lib>.cppm`` re-exports a vendor library
with ``using ::name;`` lines. Three assertions (#117, #121), all pure text against
JSON -- no SDK, no compile -- so the fast Python tier runs them on a bare runner:

  1. FLOOR CONFORMANCE (check_all). Hand-written source stays buildable at the
     *floor* SDK: a symbol used unguarded that exists at the pin but not the floor
     fails, and every version guard is validated as a spec (see below). Catches
     the regression the floor policy exists to prevent, which today only a build
     against the floor SDK finds -- and CI builds the pin.
  2. LINK-CHECK COMPLETENESS (check_link_completeness). Every re-exported function
     carries a WWR_LINK_CHECK / WWR_DECLARED_CHECK, so a newly wrapped function
     cannot ship with only "it compiles" behind it. See that function's section.
  3. MACRO CORRECTNESS (check_macro_correctness). Each of those checks uses the
     macro the manifest's LINKABLE data demands: a symbol the .so exports must be
     WWR_LINK_CHECK, one it only declares must be WWR_DECLARED_CHECK. The reverse
     direction is the payoff -- see that function's section.

The rest of this header describes assertion 1.

WHAT COUNTS AS THIS LIBRARY'S SURFACE. A manifest captures a name iff
``leading_prefix(name)`` is the manifest's ``prefix`` OR one of its recorded
``extra_prefixes`` -- the SAME accepted-token set ``devtools/vendor_harvest.py``
harvests by (``leading_prefix`` is mirrored here verbatim, and the token set is
read back from the manifest so the two cannot drift). So the check reasons about
every ``using ::`` name whose leading token the manifest covers, INCLUDING a
variant library's own uppercase constants (``HIPBLASLT_EPILOGUE_*`` ->
``hipblaslt``, admitted alongside base prefix ``hipblas`` -- see #167). Everything
else -- a cross-library type re-export (``cudaStream_t`` in a curand module) -- is
not part of what this manifest claims to cover, so it is out of scope rather than
a false "absent from the manifest". A genuinely bogus ``using ::`` name fails to
*compile* against any SDK, which is not this checker's job.

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

BLIND SPOTS (now a tripwire, not a standing gap). A variant library's own
uppercase constants used to be captured by no manifest -- ``leading_prefix`` keys
on the leading token, so ``HIPBLASLT_*`` -> ``hipblaslt`` fell outside prefix
``hipblas``. #167 closed that by whitelisting each variant's own token as an
``extra_prefix`` in the manifest (``vendor_manifests.sh``), so those names are now
in scope and validated. What remains counted here is the RESIDUAL: a version-
guarded ``using ::`` whose leading token is covered by NO manifest's accepted set
-- a new companion-token constant nobody whitelisted yet. The count is pinned at 0
by ``test_blind_spot_is_bounded``, so the day one appears it is noticed rather
than silently unchecked; the fix is another ``--extra-prefix`` row.

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


def load_manifest(path: Path) -> tuple[set[str], set[str]]:
    """(accepted-token set, declared-name set) from a committed manifest.

    The accepted-token set is the primary ``prefix`` plus any ``extra_prefixes``
    the harvest recorded (a variant library's own token -- #167); membership is
    keyed on it so the checker mirrors exactly what the harvest kept.
    """
    j = json.loads(path.read_text())
    h = j["harvest"]
    accepted = {h["prefix"], *h.get("extra_prefixes", [])}
    return accepted, {s["name"] for s in j["symbols"]}


@dataclass(frozen=True)
class Violation:
    library: str
    name: str
    kind: str  # human-readable rule that was broken
    floor: str  # the floor SDK label, for the message


def check_source(
    text: str, accepted: set[str], pin: set[str], floor: set[str],
    library: str, floor_label: str,
) -> tuple[list[Violation], int]:
    """Check one module's source against its pin/floor name sets.

    ``accepted`` is the manifest's accepted-token set (prefix + extra_prefixes).
    Returns (violations, blind_spot_count). ``blind_spot_count`` is the number of
    version-guarded sites the manifest cannot see (leading token outside
    ``accepted``).
    """
    violations: list[Violation] = []
    blind = 0
    for site in iter_using_sites(text):
        in_scope = leading_prefix(site.name) in accepted
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
    blind_spots: int        # guarded sites outside every manifest's accepted tokens


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
            # The pin is the authority for scope; the floor contributes only its
            # name set (does the floor SDK ship this symbol?).
            accepted, pin = load_manifest(pin_dir / f"{stem}.json")
            _, floor = load_manifest(floor_dir / f"{stem}.json")
            v, b = check_source(
                cppm.read_text(), accepted, pin, floor, f"{backend}/{stem}",
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


# ── Assertion 2: link-check completeness ────────────────────────────────────
#
# Every function wwr re-exports (a ``using ::`` whose name the manifest records as
# a FunctionDecl) must carry a WWR_LINK_CHECK or WWR_DECLARED_CHECK in test/, so a
# newly wrapped function cannot ship with only "it compiles" behind it -- the
# check test/hip/README.md's hipSPARSE-546 note asserts by hand, with nothing
# enforcing it. Assert, never generate: the link check and the using:: are two
# independent statements the linker then judges; this only checks one exists.
#
# A library with a compiled test module IS in CUDA_TEST_LIBRARIES / HIP_TEST_
# LIBRARIES, so the natural scope is "libraries the suite link-checks". But a few
# manifest-backed libraries are deliberately NOT link-checked; they are named here
# with the reason, so removing a library from the suite is caught (its functions
# become unchecked) rather than silently dropping coverage.
_CHECK = re.compile(r"WWR_(?:LINK|DECLARED)_CHECK\(\s*([A-Za-z_]\w*)\s*\)")
# As above, but captures which macro -- assertion 3 needs LINK vs DECLARED.
_MACRO_CHECK = re.compile(r"WWR_(LINK|DECLARED)_CHECK\(\s*([A-Za-z_]\w*)\s*\)")

LINK_CHECK_EXCEPTIONS: dict[tuple[str, str], str] = {
    # hiptensor/rccl compile and link, but their .so's SIGBUS at load in the
    # driverless hip_compile_tests on GPU-less CI -- the problem test/cuda solves
    # with driver stubs + the `gpu` label and test/hip has no mechanism for yet.
    # Tracked in #179; drop these once it lands.
    ("hip", "hiptensor"): "runtime-load fault in driverless CI (#179)",
    ("hip", "rccl"): "runtime-load fault in driverless CI (#179)",
}


def unchecked_functions(
    funcs: set[str], using: set[str], checks: set[str],
) -> list[str]:
    """Wrapped functions with no link/declared check.

    A function is wrapped iff the module re-exports it (``using``) AND the manifest
    records it as a FunctionDecl (``funcs``); it is covered iff a WWR_*_CHECK names
    it (``checks``). The gap is the wrapped set minus the covered set.
    """
    return sorted((funcs & using) - checks)


@dataclass(frozen=True)
class LinkGap:
    library: str
    unchecked: tuple[str, ...]  # wrapped functions with no WWR_*_CHECK


@dataclass
class LinkReport:
    gaps: list[LinkGap]
    checked: list[str]    # "backend/lib" libraries whose surface was verified
    excepted: list[str]   # "backend/lib" libraries skipped with a stated reason


def check_link_completeness() -> LinkReport:
    gaps: list[LinkGap] = []
    checked: list[str] = []
    excepted: list[str] = []
    for backend, pin_dir, _ in BACKENDS:
        manifests = {p.stem for p in pin_dir.glob("*.json")}
        all_checks: set[str] = set()
        for test_cppm in (ROOT / "test" / backend).glob("*.cppm"):
            all_checks |= set(_CHECK.findall(test_cppm.read_text()))
        for cppm in sorted((ROOT / "src" / backend).glob("*.cppm")):
            stem = cppm.stem
            if stem not in manifests:
                continue  # no manifest => no declared-function surface to check
            if (backend, stem) in LINK_CHECK_EXCEPTIONS:
                excepted.append(f"{backend}/{stem}")
                continue
            symbols = json.loads((pin_dir / f"{stem}.json").read_text())["symbols"]
            funcs = {s["name"] for s in symbols if s.get("kind") == "FunctionDecl"}
            using = set(_USING.findall(cppm.read_text()))
            unchecked = unchecked_functions(funcs, using, all_checks)
            checked.append(f"{backend}/{stem}")
            if unchecked:
                gaps.append(LinkGap(f"{backend}/{stem}", tuple(unchecked)))
    return LinkReport(gaps, checked, excepted)


def format_link_report(report: LinkReport) -> str:
    if not report.gaps:
        return (f"link-check completeness OK: {len(report.checked)} libraries "
                f"verified, 0 gaps ({len(report.excepted)} excepted with reason).")
    total = sum(len(g.unchecked) for g in report.gaps)
    lines = [f"link-check completeness FAILED: {total} re-exported functions "
             f"with no WWR_LINK_CHECK/WWR_DECLARED_CHECK, in {len(report.gaps)} "
             f"libraries."]
    for g in sorted(report.gaps, key=lambda g: g.library):
        lines.append(f"  {g.library}: {len(g.unchecked)}")
        for n in g.unchecked[:5]:
            lines.append(f"      {n}")
        if len(g.unchecked) > 5:
            lines.append(f"      ... and {len(g.unchecked) - 5} more")
    return "\n".join(lines)


# ── Assertion 3: macro correctness ──────────────────────────────────────────
#
# The link check comes in two macros, and the manifest's LINKABLE data (#119 --
# nm over the actual .so, not the header) decides which one a site must use: a
# name the .so exports must be WWR_LINK_CHECK (the linker resolves it), one it
# declares but does not export must be WWR_DECLARED_CHECK (WWR_LINK_CHECK on it
# would fail the link of the compile-tests executable). This asserts each site
# picked the right one, in both directions.
#
# The WWR_DECLARED_CHECK -> WWR_LINK_CHECK direction is the payoff. link_check.h
# used to tell the reader to "switch back to WWR_LINK_CHECK when a newer library
# exports it" -- a someday-maybe nobody ever actioned, because noticing meant
# re-testing a symbol already written off as broken. With the manifest it is
# automatic: the day a vendor starts exporting a declared-only symbol, its
# WWR_DECLARED_CHECK becomes a violation here and the build says so.
#
# SCOPE. A test module test/<backend>/<stem>.cppm is judged against the <stem>
# pin manifest's linkable surface (the `linkable` list unioned with
# `declared_not_linkable`). A checked name in neither set is out of scope -- a
# cross-library symbol, or one the harvest never saw -- and only counted. A
# manifest whose .so was absent at harvest (`linkable.available` false --
# nvToolsExt, whose libnvToolsExt.so CUDA dropped at 12) carries no linkability
# truth, so its module is reported UNJUDGED rather than guessed.


def macro_violation(
    kind: str, sym: str, linkable: set[str], declared_only: set[str],
) -> str | None:
    """The macro a site SHOULD use if it used the wrong one, else None.

    ``kind`` is the macro the site uses ("LINK" or "DECLARED"). A name the .so
    exports must be WWR_LINK_CHECK; one it declares but does not export must be
    WWR_DECLARED_CHECK. A name in neither set is out of scope -> None (the
    caller counts it, it is not a violation). ``linkable`` wins a tie, so a name
    in both sets is treated as exported.
    """
    if sym in linkable:
        return "WWR_LINK_CHECK" if kind != "LINK" else None
    if sym in declared_only:
        return "WWR_DECLARED_CHECK" if kind != "DECLARED" else None
    return None


@dataclass(frozen=True)
class Mislabel:
    library: str
    name: str
    used: str    # the macro the source wrote
    should: str  # the macro the manifest's linkable data demands


@dataclass
class MacroReport:
    mislabels: list[Mislabel]
    checked: list[str]   # "backend/lib" modules judged against linkable data
    unjudged: list[str]  # "backend/lib" modules whose .so was absent at harvest
    out_of_scope: int    # checks whose name is outside the manifest's surface


def check_macro_correctness() -> MacroReport:
    mislabels: list[Mislabel] = []
    checked: list[str] = []
    unjudged: list[str] = []
    out_of_scope = 0
    for backend, pin_dir, _ in BACKENDS:
        manifests = {p.stem for p in pin_dir.glob("*.json")}
        for test_cppm in sorted((ROOT / "test" / backend).glob("*.cppm")):
            stem = test_cppm.stem
            if stem not in manifests:
                continue  # no manifest => no linkable surface to judge against
            pairs = _MACRO_CHECK.findall(test_cppm.read_text())
            if not pairs:
                continue
            lib = f"{backend}/{stem}"
            link = json.loads((pin_dir / f"{stem}.json").read_text())["linkable"]
            if not link["available"]:
                unjudged.append(lib)  # .so absent at harvest: no truth to judge by
                continue
            linkable = set(link["linkable"])
            declared_only = set(link.get("declared_not_linkable", []))
            checked.append(lib)
            for kind, sym in pairs:
                should = macro_violation(kind, sym, linkable, declared_only)
                if should is not None:
                    mislabels.append(Mislabel(lib, sym, f"WWR_{kind}_CHECK", should))
                elif sym not in linkable and sym not in declared_only:
                    out_of_scope += 1
    return MacroReport(mislabels, checked, unjudged, out_of_scope)


def format_macro_report(report: MacroReport) -> str:
    if not report.mislabels:
        return (f"macro correctness OK: {len(report.checked)} modules judged, "
                f"0 mislabeled ({len(report.unjudged)} unjudged -- .so absent at "
                f"harvest, {report.out_of_scope} checks out of scope).")
    by_lib: dict[str, list[Mislabel]] = defaultdict(list)
    for m in report.mislabels:
        by_lib[m.library].append(m)
    lines = [f"macro correctness FAILED: {len(report.mislabels)} mislabeled "
             f"checks in {len(by_lib)} modules."]
    for lib in sorted(by_lib):
        ms = by_lib[lib]
        lines.append(f"  {lib}: {len(ms)}")
        for m in ms[:5]:
            verb = ("the .so exports it" if m.should == "WWR_LINK_CHECK"
                    else "the .so does not export it")
            lines.append(f"      {m.name}: {m.used}, but {verb} -> {m.should}")
        if len(ms) > 5:
            lines.append(f"      ... and {len(ms) - 5} more")
    return "\n".join(lines)


def main() -> int:
    argparse.ArgumentParser(
        description="Assert src/<backend> using:: lines against the vendor "
                    "manifests (#117, #121): floor conformance + link-check "
                    "completeness + macro correctness.").parse_args()
    floor = check_all()
    print(format_report(floor))
    link = check_link_completeness()
    print(format_link_report(link))
    macro = check_macro_correctness()
    print(format_macro_report(macro))
    return 1 if (floor.violations or link.gaps or macro.mislabels) else 0


if __name__ == "__main__":
    sys.exit(main())
