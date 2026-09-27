#!/usr/bin/env python3
"""Guard that every src function alias is independently verified.

Each ``src/<m>.cppm`` binds backend-neutral ``gpu*`` names to a backend's own
functions with ``WWR_FUNCTION(gpu_name, cuda_name, hip_name)`` (or the
prefix-pasting ``WWR_RT_FUNCTION(x)`` -> ``gpu##x``). Two independent statements
prove each alias points at the right backend symbol:

  * ``test/gpu/<m>.cppm`` restates it in full with ``WWR_SAME_FUNCTION`` -- for the
    handle/stream/status functions no wrapper dispatches to; and
  * the dispatch table ``test/wrappers/<m>/<m>_dispatch.toml`` names it as the
    expected call target of a wrapper instantiation, which
    ``test/shared/dispatch.py`` verifies against the compiled objects -- for the
    typed entry points.

Between them coverage is total, but nothing *enforced* that: a newly added alias
that no wrapper dispatches to and that no one adds a ``WWR_SAME_FUNCTION`` for would
ship with only "it compiles" behind it -- and a wrong-but-existing backend name
compiles. This checker fails when that happens. It parses source only; it needs
no build. It is deliberately blind to *whether* a ``WWR_SAME_FUNCTION``/table entry
is correct (dispatch.py and the compiler judge that); it only asserts that one
exists for every alias.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

# WWR_FUNCTION(gpu_name, ...) -- anchored at line start so the macro *definition*
# in gpu_backend.h ("#define WWR_FUNCTION ...") and the paste inside
# WWR_RT_FUNCTION's definition are not read as invocations.
_WWR_FUNCTION_RE = re.compile(r"^\s*WWR_FUNCTION\(\s*(\w+)\s*,")
# WWR_RT_FUNCTION(X) expands to WWR_FUNCTION(gpuX, cudaX, hipX); alias is gpuX.
_WWR_RT_FUNCTION_RE = re.compile(r"^\s*WWR_RT_FUNCTION\(\s*(\w+)\s*\)")
_WWR_SAME_FUNCTION_RE = re.compile(r"^\s*WWR_SAME_FUNCTION\(\s*(\w+)\s*,")
_GPU_NAME_RE = re.compile(r"^gpu\w+$")


def parse_aliases(text: str) -> set[str]:
    """The gpu* function aliases a gpu* module defines."""
    aliases: set[str] = set()
    for line in text.splitlines():
        if line.lstrip().startswith("#define"):
            continue
        if m := _WWR_FUNCTION_RE.match(line):
            aliases.add(m.group(1))
        elif m := _WWR_RT_FUNCTION_RE.match(line):
            aliases.add("gpu" + m.group(1))
    return aliases


def parse_same_functions(text: str) -> set[str]:
    """The gpu* names a test/gpu module checks with WWR_SAME_FUNCTION."""
    return {
        m.group(1)
        for line in text.splitlines()
        if (m := _WWR_SAME_FUNCTION_RE.match(line))
    }


def parse_toml_targets(data: dict) -> set[str]:
    """Every gpu*-looking string named anywhere in a dispatch table.

    Dispatch expectations are lists of gpu* call targets; we take any gpu* string
    so the parse does not depend on the table's exact section layout. Non-function
    gpu* strings (a type name in ``[type_names]``) are harmless here -- they can
    only over-cover, never hide a missing function.
    """
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, str):
            if _GPU_NAME_RE.match(node):
                found.add(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return found


def coverage_gap(aliases: set[str], same: set[str], toml_targets: set[str]) -> set[str]:
    """Aliases with neither a WWR_SAME_FUNCTION nor a dispatch-table entry."""
    return aliases - (same | toml_targets)


def check_module(root: Path, module: str) -> set[str]:
    """Uncovered aliases for one module (empty set == fully covered)."""
    src = (root / "src" / f"{module}.cppm").read_text()
    aliases = parse_aliases(src)
    if not aliases:
        return set()

    test_path = root / "test" / "gpu" / f"{module}.cppm"
    same = parse_same_functions(test_path.read_text()) if test_path.exists() else set()

    toml_path = root / "test" / "wrappers" / module / f"{module}_dispatch.toml"
    if toml_path.exists():
        toml_targets = parse_toml_targets(tomllib.loads(toml_path.read_text()))
    else:
        toml_targets = set()

    return coverage_gap(aliases, same, toml_targets)


def discover_modules(root: Path) -> list[str]:
    """The gpu* layer modules (directly under src/) that define at least one alias."""
    gpu_dir = root / "src"
    modules = [
        p.stem for p in sorted(gpu_dir.glob("*.cppm")) if parse_aliases(p.read_text())
    ]
    return modules


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="repository root")
    parser.add_argument("--stamp", type=Path, help="file to touch on success")
    args = parser.parse_args(argv)

    modules = discover_modules(args.root)
    if not modules:
        print("alias_coverage: found no src function aliases -- parser broken?",
              file=sys.stderr)
        return 2

    failed = False
    for module in modules:
        gap = check_module(args.root, module)
        if gap:
            failed = True
            names = ", ".join(sorted(gap))
            print(
                f"alias_coverage: {module}: {len(gap)} gpu* function(s) with no "
                f"WWR_SAME_FUNCTION and no dispatch-table entry: {names}",
                file=sys.stderr,
            )

    if failed:
        print(
            "alias_coverage: add a WWR_SAME_FUNCTION in the matching test/gpu "
            "module, or a dispatch expectation in its *_dispatch.toml.",
            file=sys.stderr,
        )
        return 1

    if args.stamp:
        args.stamp.write_text("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
