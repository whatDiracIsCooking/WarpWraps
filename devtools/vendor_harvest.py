#!/usr/bin/env python3
"""Harvest one vendor library's DECLARED SURFACE into a JSON manifest via clang.

``header_intersection.py`` answers "which NAMES do cuRAND and hipRAND share?" and
is right to do it with a regex over text: the vendor headers pull in device
intrinsics that no host parser digests cleanly, and a name is just a token. This
tool answers the harder question its sibling explicitly disclaims -- "do the two
backends share a SIGNATURE, an ENUM VALUE, a deprecation?" -- and for that a
token is not enough. So it parses with clang instead:

    clang -x c++ -Xclang -ast-dump=json <flags> HEADER

The objection that stops ``header_intersection.py`` from parsing is about DEVICE
code; the HOST API declarations parse fine, and clang-20 ships in every image.
Errors in the device-side transitive includes are tolerated -- clang still emits
a complete AST for the host declarations, which is all we read (see
``--strict``). What lands in the manifest, per symbol: its name, kind, a
NORMALISED declaration string (and its hash, for cheap diffing), the computed
enum value where there is one, a deprecation marker, whether the declaration
carries an exported-visibility attribute, and the originating header.

CONFIG TUPLE. An AST resolves the preprocessor for EXACTLY ONE configuration, so
a manifest is a fact about (backend, sdk_version, arch, defines) -- not about the
library in the abstract. ``hipsparse`` parsed with ``-DCUDART_VERSION=...`` and
without it are two different surfaces; the compat branches that ``src/hip/README.md``
documents by hand are exactly this. Every emitted file records the exact command
and the config tuple that produced it under ``"harvest"``, so a manifest is
reproducible and a diff between two of them is meaningful only when their tuples
agree.

ONE FILE PER LIBRARY. src/cuda and src/hip carry ~10,700 ``using ::`` lines
between them; a single blob manifest would make every diff unreadable. So the
default output name is ``<prefix>.harvest.json`` (``hiprand.harvest.json``), one
per invocation.

THIS TOOL ONLY WRITES MANIFESTS. Nothing it emits is ever fed back into source --
it is a spec/diff artifact, the counterpart to ``header_intersection.py``'s
coverage check, not a code generator.

WHAT IT CANNOT RECOVER. The AST is post-preprocessor, so the SPELLING of an
export macro (``HIPRANDAPI``, ``HIPSPARSE_EXPORT``) is gone by the time we parse
-- what survives is the attribute it expanded to. ``exported`` therefore records
whether a default-visibility / dllexport attribute is present, not the macro's
name. Matching ``header_intersection.py`` is by NAME and is close but not exact:
that script counts every prefixed identifier that appears anywhere in the text
(including in a macro body or a doc ``@see``), whereas an AST only sees actual
DECLARATIONS -- so a name that is only ever referenced, never declared in this
header set, is in the regex list and legitimately not here.

Usage:
    # HIP side (needs the platform define that its headers #error without):
    devtools/vendor_harvest.py --backend HIP \\
        --sdk-version "ROCm 7.2.4" --arch gfx942 \\
        -D __HIP_PLATFORM_AMD__ \\
        -I /opt/rocm/include \\
        /opt/rocm/include/hiprand/hiprand.h

    # CUDA side:
    devtools/vendor_harvest.py --backend CUDA \\
        --sdk-version "CUDA 13.0" \\
        -I /usr/local/cuda/include \\
        /usr/local/cuda/include/curand.h

    # A compat branch, captured by defining the guard the header keys on:
    devtools/vendor_harvest.py --backend HIP --sdk-version "ROCm 7.2.4" \\
        -D __HIP_PLATFORM_AMD__ -D CUDART_VERSION=12000 \\
        -I /opt/rocm/include /opt/rocm/include/hipsparse/hipsparse.h
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Declaration node kinds we surface. Everything else in the AST (statements,
# expressions, the guts of a definition) is traversed but not emitted.
_DECL_KINDS = {
    "FunctionDecl",
    "TypedefDecl",
    "TypeAliasDecl",
    "EnumDecl",
    "EnumConstantDecl",
    "VarDecl",
    "CXXRecordDecl",
    "RecordDecl",
    "FunctionTemplateDecl",
}

# clang-20's canonical DeprecatedAttr; also catch the C++14 spelling variants
# that print with different node kinds on some builds.
_DEPRECATED_ATTRS = {"DeprecatedAttr", "UnavailableAttr"}

# An attribute that marks a symbol as part of the exported ABI. The macro that
# produced it (HIPRANDAPI, HIPSPARSE_EXPORT, ...) is gone post-preprocessor; the
# attribute is what survives, so this is what "exported" reports.
_EXPORT_ATTRS = {"VisibilityAttr", "DLLExportAttr"}

_WS_RE = re.compile(r"\s+")


def leading_prefix(ident: str) -> str | None:
    """The library token an identifier leads with, lowercased.

    ``hiprandFoo`` leads with a lowercase run (``hiprand``); ``HIPRAND_FOO`` with
    an all-caps run up to the first separator (``HIPRAND`` -> ``hiprand``). Both
    normalise to the same token, mirroring ``header_intersection.py`` so the two
    tools agree on what "the library's prefix" means.
    """
    if m := re.match(r"[a-z]+", ident):
        return m.group(0)
    if m := re.match(r"[A-Z]+", ident):
        return m.group(0).lower()
    return None


def run_clang(clang: str, header: Path, flags: list[str], std: str) -> tuple[dict, str]:
    """Dump ``header``'s AST as JSON. Returns (ast, stderr).

    ``-fsyntax-only`` so no object file is written; ``-ferror-limit=0`` so a
    device-side #error does not truncate the host-declaration AST we came for.
    A nonzero exit is EXPECTED (the device includes do not parse) and is not by
    itself fatal -- see ``--strict``.
    """
    cmd = [
        clang,
        "-x", "c++",
        f"-std={std}",
        "-fsyntax-only",
        "-ferror-limit=0",
        "-Xclang", "-ast-dump=json",
        *flags,
        str(header),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not proc.stdout:
        raise SystemExit(
            f"vendor_harvest: clang produced no AST for {header}\n"
            f"  command: {' '.join(cmd)}\n{proc.stderr}"
        )
    return json.loads(proc.stdout), proc.stderr


def _child_attrs(node: dict) -> set[str]:
    return {c.get("kind", "") for c in node.get("inner", ())}


def _params(node: dict) -> list[dict]:
    return [c for c in node.get("inner", ()) if c.get("kind") == "ParmVarDecl"]


def _normalise(text: str) -> str:
    """Collapse whitespace so a reformat is not a diff."""
    return _WS_RE.sub(" ", text).strip()


def declaration_text(node: dict) -> str:
    """A stable, source-independent rendering of a declaration.

    Built from the AST's own type spelling (``type.qualType``) rather than the
    original bytes, so it is immune to the header's formatting and comments while
    still capturing the SIGNATURE -- the thing that makes this a spec and not a
    name list. Function parameters keep their names (a signature change the AST
    would otherwise hide, e.g. a reordering, still shows). Deliberately not a
    compilable declaration; it is a canonical KEY.
    """
    kind = node["kind"]
    name = node.get("name", "")
    qual = (node.get("type") or {}).get("qualType", "")

    if kind in ("FunctionDecl", "FunctionTemplateDecl"):
        # qualType of a function is "RET (ARG, ARG)"; splice the name in.
        ret = qual.split("(", 1)[0].strip() if "(" in qual else qual
        parts = []
        for p in _params(node):
            pn = p.get("name", "")
            pt = (p.get("type") or {}).get("qualType", "")
            parts.append(f"{pt} {pn}".strip())
        return _normalise(f"{ret} {name}({', '.join(parts)})")

    if kind in ("TypedefDecl", "TypeAliasDecl"):
        return _normalise(f"typedef {qual} {name}")

    if kind == "EnumConstantDecl":
        return _normalise(f"{name} = {enum_value(node)}")

    if kind in ("VarDecl",):
        return _normalise(f"{qual} {name}")

    if kind in ("EnumDecl", "CXXRecordDecl", "RecordDecl"):
        tag = {"EnumDecl": "enum", "CXXRecordDecl": "struct",
               "RecordDecl": "struct"}.get(kind, "")
        return _normalise(f"{tag} {name or '(anonymous)'}")

    return _normalise(f"{name} : {qual}")


def enum_value(node: dict) -> str | None:
    """The computed value of an EnumConstantDecl, as clang folded it.

    clang wraps an explicit initialiser in a ``ConstantExpr`` carrying the folded
    ``value`` (so ``= 100`` and ``= PREV + 1`` alike resolve to a number). It can
    sit under an ``ImplicitCastExpr`` when the enum's underlying type differs from
    the literal's (``unsigned int`` enum, ``int`` literal), so search descendants,
    not just direct children. An implicit value (no initialiser) has no
    ConstantExpr; return None so the caller can fall back to positional numbering.
    """
    if node.get("kind") != "EnumConstantDecl":
        return None
    stack = list(node.get("inner", ()))
    while stack:
        c = stack.pop(0)
        if c.get("kind") == "ConstantExpr" and "value" in c:
            return c["value"]
        # Don't descend into the doc comment subtree; only the initialiser.
        if c.get("kind") not in ("FullComment",):
            stack.extend(c.get("inner", ()))
    return None


def harvest(ast: dict, prefix: str) -> list[dict]:
    """Walk the AST, emitting every declaration whose name carries ``prefix``.

    File attribution uses clang's DOCUMENT-ORDER sticky ``loc.file``: a node
    prints ``file`` only when it differs from the previous node's, so the current
    file is carried forward across the whole pre-order walk (NOT scoped to a
    subtree -- restoring it per-subtree loses a file a sibling just entered).
    """
    state = {"file": None}
    # Per enum, a running counter so an implicit-valued constant still gets a
    # number (its ordinal within the enum), matching C's rules.
    enum_counter = {"next": 0}
    out: list[dict] = []

    def visit(node: dict) -> None:
        loc = node.get("loc") or {}
        if loc.get("file"):
            state["file"] = loc["file"]
        rng_begin = (node.get("range") or {}).get("begin") or {}
        if rng_begin.get("file"):
            state["file"] = rng_begin["file"]

        kind = node.get("kind", "")
        name = node.get("name", "")

        if kind == "EnumDecl":
            enum_counter["next"] = 0

        if kind in _DECL_KINDS and name and leading_prefix(name) == prefix:
            attrs = _child_attrs(node)
            entry: dict = {
                "name": name,
                "kind": kind,
                "header": state["file"],
                "decl": declaration_text(node),
            }
            if kind == "EnumConstantDecl":
                v = enum_value(node)
                entry["value"] = int(v) if v is not None else enum_counter["next"]
                entry["value_explicit"] = v is not None
            entry["decl_hash"] = hashlib.sha256(
                entry["decl"].encode()).hexdigest()[:16]
            if attrs & _DEPRECATED_ATTRS:
                entry["deprecated"] = True
            if attrs & _EXPORT_ATTRS:
                entry["exported"] = True
            out.append(entry)

        # Advance the enum ordinal AFTER recording, so the first constant is 0.
        if kind == "EnumConstantDecl":
            v = enum_value(node)
            enum_counter["next"] = (int(v) if v is not None
                                    else enum_counter["next"]) + 1

        for child in node.get("inner", ()):
            visit(child)

    visit(ast)
    # A name can be declared more than once (a forward decl + definition). Keep
    # the richest entry (prefer one with a body-bearing decl / more attributes).
    best: dict[str, dict] = {}
    for e in out:
        key = (e["name"], e["kind"])
        prev = best.get(key)
        if prev is None or len(json.dumps(e)) > len(json.dumps(prev)):
            best[key] = e
    return sorted(best.values(), key=lambda e: (e["kind"], e["name"]))


def detect_prefix(ast: dict, header: Path, family: str | None) -> str | None:
    """Mode of the leading tokens of decls IN THE TARGET HEADER ITSELF.

    Scoping to the input header (not the whole transitive AST) is what tells
    ``curand`` apart from the ``cuda`` runtime it pulls in: a bare ``--backend
    CUDA`` seed matches both ``cuda`` and ``curand``, and the runtime dwarfs the
    library, so an un-scoped mode would pick ``cuda`` and harvest the world.
    Restricting the vote to declarations whose sticky ``loc.file`` is the header
    named on the command line isolates the library's own prefix. A library that
    also declares into a same-name sibling header (``hiprand_rocm.h``) is still
    captured by the resulting prefix during the real harvest.
    """
    target = str(header)
    state = {"file": None}
    counts: Counter[str] = Counter()

    def visit(node: dict) -> None:
        loc = node.get("loc") or {}
        if loc.get("file"):
            state["file"] = loc["file"]
        rb = (node.get("range") or {}).get("begin") or {}
        if rb.get("file"):
            state["file"] = rb["file"]
        name = node.get("name", "")
        if (name and node.get("kind") in _DECL_KINDS
                and state["file"] == target):
            p = leading_prefix(name)
            if p and (family is None or p.startswith(family)):
                counts[p] += 1
        for c in node.get("inner", ()):
            visit(c)

    visit(ast)
    return counts.most_common(1)[0][0] if counts else None


def build_manifest(
    header: Path,
    ast: dict,
    prefix: str,
    backend: str,
    sdk_version: str | None,
    arch: str | None,
    defines: list[str],
    includes: list[str],
    clang_version: str,
    command: list[str],
) -> dict:
    symbols = harvest(ast, prefix)
    return {
        "harvest": {
            "tool": "devtools/vendor_harvest.py",
            "note": "declared surface for ONE (backend, sdk, arch, defines) "
                    "configuration; not a code generator -- see the file header",
            "command": " ".join(command),
            "config": {
                "backend": backend,
                "sdk_version": sdk_version,
                "arch": arch,
                "defines": defines,
                "includes": includes,
            },
            "clang": clang_version,
            "header": str(header),
            "prefix": prefix,
            "symbol_count": len(symbols),
        },
        "symbols": symbols,
    }


def clang_version_string(clang: str) -> str:
    try:
        out = subprocess.run([clang, "--version"], capture_output=True, text=True)
        return out.stdout.splitlines()[0].strip() if out.stdout else clang
    except OSError:
        return clang


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("header", type=Path, help="the vendor header to harvest")
    parser.add_argument("--backend", choices=["CUDA", "HIP"], required=True,
                        help="which backend this configuration targets")
    parser.add_argument("--sdk-version", help="e.g. 'CUDA 13.0' / 'ROCm 7.2.4' "
                        "(free text; recorded in the config tuple)")
    parser.add_argument("--arch", help="GPU arch this config resolves for, if "
                        "the header branches on it (e.g. gfx942, sm_90)")
    parser.add_argument("-D", dest="defines", action="append", default=[],
                        metavar="MACRO[=VAL]", help="a preprocessor define "
                        "(repeatable); the AST resolves branches for exactly "
                        "these")
    parser.add_argument("-I", dest="includes", action="append", default=[],
                        metavar="DIR", help="an include dir (repeatable)")
    parser.add_argument("--prefix", help="pin the vendor prefix (else the mode "
                        "of prefixed identifiers, seeded by --backend)")
    parser.add_argument("--std", default="c++17",
                        help="C++ standard passed to clang (default c++17)")
    parser.add_argument("--clang", default=None,
                        help="clang binary (default: clang-20, then clang)")
    parser.add_argument("-o", "--output", type=Path,
                        help="output path (default <prefix>.harvest.json in cwd)")
    parser.add_argument("--strict", action="store_true",
                        help="fail if clang emitted ANY diagnostic; off by "
                        "default because the device-side transitive includes "
                        "are expected not to parse and do not affect the host "
                        "declarations we read")
    parser.add_argument("--stats", action="store_true",
                        help="print a one-line summary to stderr")
    args = parser.parse_args(argv)

    clang = args.clang or (shutil.which("clang-20") or shutil.which("clang"))
    if not clang:
        print("vendor_harvest: no clang found (need clang-20 or clang on PATH); "
              "pass --clang", file=sys.stderr)
        return 2
    if not args.header.exists():
        print(f"vendor_harvest: header not found: {args.header}", file=sys.stderr)
        return 2

    flags = [f"-D{d}" for d in args.defines] + [f"-I{i}" for i in args.includes]
    ast, stderr = run_clang(clang, args.header, flags, args.std)

    if args.strict and stderr.strip():
        print("vendor_harvest: --strict and clang emitted diagnostics:\n"
              + stderr, file=sys.stderr)
        return 1

    family = {"CUDA": "cu", "HIP": "hip"}[args.backend]
    prefix = args.prefix or detect_prefix(ast, args.header, family)
    if not prefix:
        print(f"vendor_harvest: no {args.backend} vendor prefix detected in "
              f"{args.header}; pass --prefix", file=sys.stderr)
        return 2

    command = [
        clang, "-x", "c++", f"-std={args.std}", "-fsyntax-only",
        "-ferror-limit=0", "-Xclang", "-ast-dump=json", *flags, str(args.header),
    ]
    manifest = build_manifest(
        header=args.header, ast=ast, prefix=prefix, backend=args.backend,
        sdk_version=args.sdk_version, arch=args.arch, defines=args.defines,
        includes=args.includes, clang_version=clang_version_string(clang),
        command=command,
    )

    out_path = args.output or Path(f"{prefix}.harvest.json")
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if args.stats:
        by_kind = Counter(s["kind"] for s in manifest["symbols"])
        dep = sum(1 for s in manifest["symbols"] if s.get("deprecated"))
        print(f"vendor_harvest: {manifest['harvest']['symbol_count']} symbols "
              f"(prefix {prefix!r}) -> {out_path}  {dict(by_kind)}  "
              f"deprecated={dep}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
