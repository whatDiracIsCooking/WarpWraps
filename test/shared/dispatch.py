#!/usr/bin/env python3
"""Check that every wrapper in a wwr.wrappers module calls the right wwr* function.

The wrappers pick their vendor function by token-pasting a type prefix (and, in
BLAS, an optional _64 suffix) onto a basename -- see each module's
dispatch_macros.h. The type system catches a wrong precision (float* does not
convert to double*) but not a wrong integer width (an int widens silently into a
_64 entry point) or a basename typo between two functions with the same
signature (iamax/iamin, dotc/dotu, trmv/trsv, potrf/potri...). Those compile,
link, and run.

So this reads the compiled code instead. Each explicit instantiation of a
wrapper is its own function in the object file, and the relocations inside it
name the vendor function it actually calls. `llvm-objdump -dr` lists them; the
vendor name is mapped back to its wwr* alias through the WWR_FUNCTION table in
the matching wwr* module, which names both backends' functions -- so one
expected table, written in wwr* names, checks a CUDA and a HIP build alike.

Everything module-specific lives in one TOML table file (--table), so this
script is shared by every extension module that dispatches this way:

    [check]
    module     = "wwr.wrappers.blas"   # module name in the mangled symbols
    prefix     = "wwrblas"                 # the wwr* alias prefix to look for
    namespace  = "wwr"       # optional; this is the default
    forwarder_module = "wwr.blas"   # optional; see below

    [type_names.CUDA]                      # demangled spelling -> table spelling
    float2 = "wwrComplex"
    [type_names.HIP]
    "HIP_vector_type<float, 2u>" = "wwrComplex"

    [dispatch]                             # the hand-written expectations
    "iamax<float, int>"    = ["wwrblasIsamax"]
    "rot<wwrComplex, int>" = ["wwrblasCrot", "wwrblasCsrot"]

`[dispatch]` is one key per wrapper<T, ...>, then the wwr* function(s) it must
call. A key with more than one function covers overloads sharing template
arguments (BLAS's complex rot and scal; the solver names that exist in both the
legacy and the modern API); each overload must call exactly one, and together
they must call exactly the listed ones.

`forwarder_module` is for wwr* modules that define ordinary inline functions
alongside the WWR_FUNCTION aliases (wwr.blas does, for the HIP
getrs/getriBatched shims). If one of those is not inlined, the wrapper calls it
by name rather than the vendor symbol, and this is what recognises it. Omit the
key for a module that has none.

Fails on: a wrapper calling the wrong function, calling none (a missing dispatch
branch falls off the end), calling more than one, an instantiation the table
lists but the object lacks, and a wrapper calling a wwr* function that the table
does not list at all.

--print dumps what the objects actually contain, as a `[dispatch]` block ready
to paste, for writing entries for a new wrapper. Never paste its output unread:
it records whatever the code does, bugs included.
"""

import argparse
import re
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

# WWR_FUNCTION(wwrblasSaxpy, cublasSaxpy_v2, hipblasSaxpy) -- the alias is
# constrained to the configured prefix so an unrelated WWR_FUNCTION line in the
# same file (handle create/destroy, stream setters) cannot be mistaken for a
# dispatch target. The _RAW variant (WWR_FUNCTION_RAW, used by src/rand.cppm and
# any other module that binds the vendor header's own global names directly
# rather than importing a raw module) carries the same (wwr, cuda, hip) triple
# and maps identically, so it is matched too.
GPU_FUNCTION_RE_TEMPLATE = (
    r"WWR_FUNCTION(?:_RAW)?\(\s*({prefix}\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)"
)

# A wrapper as llvm-cxxfilt prints it: return type, then
# <namespace>::<name>@<module><<targs>>(<params>).
WRAPPER_RE_TEMPLATE = r"{namespace}::(\w+)@{module}<(.*?)>\("

# An inline forwarder in the wwr* module that was not inlined away.
FORWARDER_RE_TEMPLATE = r"\b({prefix}\w+)@{forwarder_module}\("

FUNC_HEADER_RE = re.compile(r"^[0-9a-f]+ <(.+)>:$")
RELOC_RE = re.compile(r"^\s+[0-9a-f]+:\s+R_\w+\s+(\S+?)(?:[+-]0x[0-9a-f]+)?$")

# A `[dispatch]` key: a wrapper name, then its template arguments.
KEY_RE = re.compile(r"^(\w+)<(.+)>$")

BACKENDS = ("CUDA", "HIP")


class Table:
    """The parsed --table file: the check's configuration and its expectations."""

    def __init__(self, path, backend):
        self.path = path
        self.backend = backend
        try:
            raw = tomllib.loads(Path(path).read_text())
        except (OSError, tomllib.TOMLDecodeError) as exc:
            sys.exit(f"error: {path}: {exc}")

        check = raw.get("check")
        if not isinstance(check, dict):
            sys.exit(f"error: {path}: missing [check] table")
        for key in ("module", "prefix"):
            if not isinstance(check.get(key), str):
                sys.exit(f"error: {path}: [check] needs a string '{key}'")
        module = check["module"]
        prefix = check["prefix"]
        namespace = check.get("namespace", "wwr")
        forwarder_module = check.get("forwarder_module")

        self.module = module
        self.prefix = prefix
        self.gpu_function_re = re.compile(
            GPU_FUNCTION_RE_TEMPLATE.format(prefix=re.escape(prefix))
        )
        self.wrapper_re = re.compile(
            WRAPPER_RE_TEMPLATE.format(
                namespace=re.escape(namespace), module=re.escape(module)
            )
        )
        self.forwarder_re = (
            re.compile(
                FORWARDER_RE_TEMPLATE.format(
                    prefix=re.escape(prefix),
                    forwarder_module=re.escape(forwarder_module),
                )
            )
            if forwarder_module
            else None
        )

        # Only the backend being built is read, but both table names are
        # checked, so [type_names.CUDNN] is rejected on a HIP build too rather
        # than sitting there doing nothing until someone builds the other half.
        type_names = raw.get("type_names", {})
        for name in type_names:
            if name not in BACKENDS:
                sys.exit(
                    f"error: {path}: [type_names.{name}] -- expected one of "
                    f"{', '.join(BACKENDS)}"
                )
        self.type_names = type_names.get(backend, {})

        dispatch = raw.get("dispatch")
        if not isinstance(dispatch, dict) or not dispatch:
            sys.exit(f"error: {path}: missing or empty [dispatch] table")
        self.expected = {}
        for key, callees in dispatch.items():
            if not isinstance(callees, list) or not callees:
                sys.exit(f"error: {path}: [dispatch] {key!r} is not a non-empty list")
            bad = [c for c in callees if not isinstance(c, str)]
            if bad:
                sys.exit(f"error: {path}: [dispatch] {key!r} has non-string entries")
            m = KEY_RE.match(key)
            if not m:
                sys.exit(f"error: {path}: [dispatch] {key!r} is not name<targs>")
            # Canonicalise so the table may space its template arguments however
            # it likes; the actual side is canonicalised the same way.
            canon = canonical_key(m.group(1), split_targs(m.group(2)))
            if canon in self.expected:
                sys.exit(f"error: {path}: [dispatch] {key!r} duplicates {canon!r}")
            self.expected[canon] = callees

    def vendor_aliases(self, gpu_source):
        """vendor symbol -> the wwr* alias(es) naming it, for this backend.

        The value is a LIST because the mapping is legitimately many-to-one: one
        backend can merge two of the other's entry points, and src/blas.cppm
        does exactly that -- wwrblasGetStatusName and wwrblasGetStatusString are
        both hipblasStatusToString under HIP. Returning a bare dict silently kept
        whichever WWR_FUNCTION line came last and dropped the other, so a wrapper
        calling the merged symbol was reported under an arbitrary one of its two
        names. Nothing distinguishes them in the object file -- they ARE one
        symbol there -- so the ambiguity is resolved where it is used, or
        reported; see actual_dispatch.
        """
        col = 2 if self.backend == "CUDA" else 3
        vendor = defaultdict(list)
        for m in self.gpu_function_re.finditer(Path(gpu_source).read_text()):
            alias = m.group(1)
            if alias not in vendor[m.group(col)]:
                vendor[m.group(col)].append(alias)
        if not vendor:
            sys.exit(
                f"error: no WWR_FUNCTION({self.prefix}...) "
                f"lines found in {gpu_source}"
            )
        return dict(vendor)

    def normalize_targs(self, targs):
        """Template arguments as the demangler spells them -> the table's names.

        A wrapper instantiated for a type missing from [type_names] shows up
        under its raw spelling, fails to match the table, and is reported -- add
        the type then.
        """
        return [self.type_names.get(p, p) for p in split_targs(targs)]


def split_targs(targs):
    """Split a template argument list on top-level commas only.

    HIP's complex type is spelled HIP_vector_type<float, 2u>, so a plain
    str.split(",") tears it in half.
    """
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(targs):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(targs[start:i].strip())
            start = i + 1
    parts.append(targs[start:].strip())
    return parts


def canonical_key(name, parts):
    return f"{name}<{', '.join(parts)}>"


def read_calls(objdump, objects):
    """mangled function -> relocation targets inside it, across every object.

    A symbol defined in several objects is ONE instantiation -- the linker keeps
    one copy -- so its target list is recorded once rather than concatenated.
    Accumulating instead made a wrapper emitted into two objects look like it
    called two vendor functions, a confusing failure for a correct tree. Two
    objects that disagree about the same symbol is a real anomaly, so that is
    reported rather than silently resolved either way.
    """
    calls, seen_in = {}, {}
    for obj in objects:
        out = subprocess.run([objdump, "-dr", "--no-show-raw-insn", obj],
                             check=True, capture_output=True, text=True).stdout
        current, local = None, {}
        for line in out.splitlines():
            if m := FUNC_HEADER_RE.match(line):
                current = m.group(1)
                local.setdefault(current, [])  # no relocations is still a definition
            elif current and (m := RELOC_RE.match(line)):
                local[current].append(m.group(1))
        for func, targets in local.items():
            if func in calls and calls[func] != targets:
                sys.exit(
                    f"error: {func} is defined in both {seen_in[func]} and {obj} "
                    f"with different calls ({calls[func]} vs {targets})"
                )
            calls.setdefault(func, targets)
            seen_in.setdefault(func, obj)
    return calls


def demangle(cxxfilt, names):
    names = list(names)
    out = subprocess.run([cxxfilt], input="\n".join(names), check=True,
                         capture_output=True, text=True).stdout.splitlines()
    if len(out) != len(names):
        sys.exit("error: llvm-cxxfilt returned a different number of lines")
    return dict(zip(names, out, strict=True))


def resolve_alias(symbol, candidates, want, key, ambiguous):
    """Which wwr* alias to report for a vendor symbol that several of them name.

    This is NOT the check giving itself the answer. When one backend merges two
    entry points -- hipblasStatusToString is both wwrblasGetStatusName and
    wwrblasGetStatusString -- the object file holds ONE symbol, so no amount of
    disassembly can say which spelling the source used, and both are equally
    true of the compiled code. Preferring the expected spelling is therefore
    exact, not lenient: the check still proves the wrapper calls that symbol and
    not a different one, which is the whole claim. Only when the expectation
    picks none of them is there a real question, and that is reported.
    """
    if len(candidates) == 1:
        return candidates[0]
    preferred = [c for c in candidates if c in want]
    if len(preferred) == 1:
        return preferred[0]
    ambiguous.append(
        f"{key}: calls {symbol}, which {' and '.join(sorted(candidates))} all "
        f"name in this backend -- the object file cannot distinguish them; "
        f"name the intended one in the expected table"
    )
    return sorted(candidates)[0]


def actual_dispatch(table, args):
    """wrapper<T, ...> -> list (one per overload) of the wwr* functions it calls."""
    vendor = table.vendor_aliases(args.gpu_source)
    calls = read_calls(args.objdump, args.objects)
    wanted = set(calls) | {t for ts in calls.values() for t in ts}
    readable = demangle(args.cxxfilt, wanted)

    actual = defaultdict(list)
    ambiguous = []
    for func, targets in calls.items():
        m = table.wrapper_re.search(readable[func])
        if not m:
            continue
        key = canonical_key(m.group(1), table.normalize_targs(m.group(2)))
        want = table.expected.get(key, ())
        called = []
        for t in targets:
            if t in vendor:
                called.append(resolve_alias(t, vendor[t], want, key, ambiguous))
            elif table.forwarder_re and (f := table.forwarder_re.search(readable[t])):
                called.append(f.group(1))
        # Helpers in the module (error_name, the handle wrapper) call no vendor
        # routine and are not wrappers; they drop out here. A wrapper that calls
        # nothing is still caught, by its expected entry going unmatched.
        if called:
            actual[key].append(called)
    return actual, ambiguous


def check(actual, expected):
    errors = []
    for key, want in sorted(expected.items()):
        got = actual.get(key)
        if got is None:
            errors.append(
                f"{key}: no instantiation calls a dispatch target "
                f"(expected {' '.join(want)}) -- missing, or its dispatch "
                f"has no branch"
            )
            continue
        for called in got:
            if len(called) != 1:
                errors.append(
                    f"{key}: one instantiation calls {len(called)} vendor "
                    f"functions: {' '.join(called) or '(none)'}"
                )
        flat = [c for called in got for c in called]
        if Counter(flat) != Counter(want):
            errors.append(
                f"{key}: calls {' '.join(sorted(flat))}, "
                f"expected {' '.join(sorted(want))}"
            )
    for key in sorted(set(actual) - set(expected)):
        calls = " ".join(c for called in actual[key] for c in called)
        errors.append(f"{key}: calls {calls} but has no entry in the expected table")
    return errors


def print_table(actual):
    """The actual dispatch as a [dispatch] block, for pasting into a --table file."""
    if not actual:
        # Silence here reads as "this module dispatches to nothing", which is
        # never true of a module worth checking -- it means the objects, the
        # module name or the prefix are wrong. Say so on stderr, so the empty
        # stdout cannot be mistaken for an answer and pasted as one.
        print(
            "warning: no wrappers found -- check the objects, [check] module "
            "and [check] prefix",
            file=sys.stderr,
        )
        return
    width = max(len(k) for k in actual) + 2  # + the two quotes
    print("[dispatch]")
    for key in sorted(actual):
        # Not deduplicated: two overloads that both call the same function must
        # be listed twice, because the check compares multisets.
        callees = sorted(c for called in actual[key] for c in called)
        rendered = ", ".join(f'"{c}"' for c in callees)
        quoted = f'"{key}"'
        print(f"{quoted:<{width}} = [{rendered}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--objdump", required=True)
    ap.add_argument("--cxxfilt", required=True)
    ap.add_argument("--backend", required=True, choices=list(BACKENDS))
    ap.add_argument("--table", required=True, help="the TOML dispatch table")
    ap.add_argument(
        "--gpu-source", required=True, help="the wwr* module holding WWR_FUNCTION"
    )
    ap.add_argument("--stamp", help="touched on success")
    ap.add_argument(
        "--print", action="store_true", help="print the actual table and exit"
    )
    ap.add_argument("objects", nargs="+")
    args = ap.parse_args()

    table = Table(args.table, args.backend)
    actual, ambiguous = actual_dispatch(table, args)
    if args.print:
        print_table(actual)
        return 0

    errors = ambiguous + check(actual, table.expected)
    if errors:
        # A stamp left by an earlier successful run must not outlive the failure
        # that follows it: the build re-runs this command, but the file it names
        # as its output would otherwise still be sitting there, newer than
        # nothing and explaining nothing.
        if args.stamp:
            Path(args.stamp).unlink(missing_ok=True)
        print(
            f"{table.module} dispatch check FAILED ({args.backend}), "
            f"{len(errors)} problem(s):",
            file=sys.stderr,
        )
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    if args.stamp:
        Path(args.stamp).touch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
