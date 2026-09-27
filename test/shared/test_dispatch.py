"""Tests for dispatch.py, the build-time dispatch checker, which sits beside it.

dispatch.py is the one Python file inside the C++ tree, and the only part of
the C++ build whose logic is testable without a compiler: it is a text pipeline
over `llvm-objdump -dr` and `llvm-cxxfilt` output. The build proves it works on
the real objects; this proves it *fails* when it should, which a passing build
never does. That asymmetry is the whole reason this file exists: dispatch.py is
a checker, so its failure mode is passing when it should fail -- silent, and
identical to success from the outside. This file's failure mode is going red.

It is the only pytest file in the repo, and `testpaths` in pyproject.toml points
at this directory alone. Do not add a pytest.ini or conftest.py here: a second
config wins over the root one for anything run from this directory, and which
markers exist would then depend on where pytest started.

The fakes below stand in for the two LLVM tools. llvm-cxxfilt passes names it
does not recognise through unchanged, so `cat` is a faithful stand-in as long as
the fake objdump emits demangled text in place of mangled symbols -- which is
what `_objdump` does.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "dispatch.py"

# A two-backend WWR_FUNCTION table, as src/blas.cppm spells it.
GPU_SOURCE = """\
WWR_FUNCTION(gpublasSaxpy, cublasSaxpy_v2, hipblasSaxpy)
WWR_FUNCTION(gpublasDaxpy, cublasDaxpy_v2, hipblasDaxpy)
WWR_FUNCTION(gpublasSscal, cublasSscal_v2, hipblasSscal)
// Not a dispatch target: the wrong prefix, and it must be ignored.
WWR_FUNCTION(gpusolverDnCreate, cusolverDnCreate, hipsolverDnCreate)
// Many-to-one, exactly as src/blas.cppm does it: hipBLAS has a single
// status-to-string entry point where cuBLAS has two.
WWR_FUNCTION(gpublasGetStatusName, cublasGetStatusName, hipblasStatusToString)
WWR_FUNCTION(gpublasGetStatusString, cublasGetStatusString, hipblasStatusToString)
"""

TABLE = """\
[check]
module = "wwr.wrappers.blas"
prefix = "gpublas"
forwarder_module = "wwr.blas"

[type_names.CUDA]
float2 = "gpuComplex"
long = "int64_t"

[type_names.HIP]
"HIP_vector_type<float, 2u>" = "gpuComplex"
long = "int64_t"

[dispatch]
"axpy<float, int>" = ["gpublasSaxpy"]
"axpy<double, int>" = ["gpublasDaxpy"]
"""

# Appended to TABLE for the overload cases: one key, two functions.
OVERLOAD_ENTRY = '"scal<gpuComplex, int>" = ["gpublasSscal", "gpublasSaxpy"]\n'


def _wrapper(name, targs, namespace="wwr"):
    """A wrapper as llvm-cxxfilt prints it."""
    return (
        f"gpublasStatus_t {namespace}::{name}"
        f"@wwr.wrappers.blas<{targs}>(int)"
    )


def _problems(stderr):
    """The individual problem lines dispatch.py reported, as a list.

    Asserting only that a message APPEARS is not enough: every "no
    instantiation" test would also pass if the fixture fed dispatch.py an empty
    disassembly, because then *every* expected key is missing. Counting the
    problems pins down that the other instantiations were parsed and accepted,
    which is what makes these tests evidence rather than coincidence.
    """
    header, _, body = stderr.partition("problem(s):\n")
    assert "dispatch check FAILED" in header, f"no failure header in: {stderr!r}"
    return [ln.strip() for ln in body.splitlines() if ln.strip()]


def _objdump(functions):
    """`llvm-objdump -dr --no-show-raw-insn` output for {function: [callees]}.

    Byte-faithful to the real tool, which matters more than it looks: the
    relocation line pads its offset to 16 hex digits and puts TWO spaces after
    the colon, where a hand-waved fake naturally writes `1: `. dispatch.py's
    RELOC_RE absorbs both with `\\s+`, so a fake that got this wrong would keep
    every test below green while tightening that `\\s+` to a literal space broke
    the real check completely -- demonstrated, and the reason
    test_fake_objdump_matches_real_llvm_objdump exists. Keep this faithful, and
    keep that test as the thing that says so.
    """
    out = ["", "fake.o:\tfile format elf64-x86-64", "", "Disassembly of section .text:"]
    for func, targets in functions.items():
        out += ["", f"0000000000000000 <{func}>:", "       0:      \tpushq\t%rbp"]
        for i, t in enumerate(targets):
            out.append(f"\t\t{i + 1:016x}:  R_X86_64_PLT32\t{t}-0x4")
        out.append("       9:      \tretq")
    return "\n".join(out) + "\n"


@pytest.fixture
def env(tmp_path):
    """A fake objdump/cxxfilt pair plus the two input files, wired together."""
    dumpfile = tmp_path / "dump.txt"

    # $3 is the object ("-dr --no-show-raw-insn OBJ"). A per-object dump wins
    # when one exists, so a test can give two objects different contents; the
    # shared dump covers the single-object majority.
    objdump = tmp_path / "objdump"
    objdump.write_text(
        '#!/bin/sh\n'
        f'if [ -f "$3.dump" ]; then cat "$3.dump"; else cat "{dumpfile}"; fi\n'
    )
    objdump.chmod(0o755)

    gpu_source = tmp_path / "blas.cppm"
    gpu_source.write_text(GPU_SOURCE)
    table = tmp_path / "table.toml"
    table.write_text(TABLE)
    obj = tmp_path / "blas.o"
    obj.write_text("")

    def run(functions, *, backend="CUDA", extra=(), objects=()):
        dumpfile.write_text(_objdump(functions))
        return subprocess.run(
            [sys.executable, str(SCRIPT),
             "--objdump", str(objdump),
             "--cxxfilt", shutil.which("cat"),
             "--backend", backend,
             "--table", str(table),
             "--gpu-source", str(gpu_source),
             *extra, str(obj), *[str(o) for o in objects]],
            capture_output=True, text=True,
        )

    def object_with(name, functions):
        """An extra object file whose disassembly is its own."""
        other = tmp_path / name
        other.write_text("")
        (tmp_path / f"{name}.dump").write_text(_objdump(functions))
        return other

    run.table = table
    run.tmp_path = tmp_path
    run.object_with = object_with
    return run


# ---- the passing shape ---------------------------------------------------

def test_correct_dispatch_passes(env):
    r = env({
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    })
    assert r.returncode == 0, r.stderr


def test_stamp_written_only_on_success(env):
    stamp = env.tmp_path / "ok.stamp"
    r = env({
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    }, extra=("--stamp", str(stamp)))
    assert r.returncode == 0 and stamp.exists()

    stamp2 = env.tmp_path / "bad.stamp"
    r = env({
        _wrapper("axpy", "float, int"): ["cublasDaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    }, extra=("--stamp", str(stamp2)))
    assert r.returncode == 1 and not stamp2.exists()


def test_hip_backend_reads_the_other_column(env):
    r = env({
        _wrapper("axpy", "float, int"): ["hipblasSaxpy"],
        _wrapper("axpy", "double, int"): ["hipblasDaxpy"],
    }, backend="HIP")
    assert r.returncode == 0, r.stderr


def test_non_wrapper_functions_are_ignored(env):
    """Helpers in the module call no vendor routine and must not be reported."""
    r = env({
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
        "char const* wwr::error_name@wwr.wrappers.blas(int)": [],
        "void some::other::thing(int)": ["cublasSaxpy_v2"],
    })
    assert r.returncode == 0, r.stderr


# ---- the failures it exists to catch -------------------------------------

def test_wrong_function_fails(env):
    r = env({
        _wrapper("axpy", "float, int"): ["cublasDaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    })
    assert r.returncode == 1
    assert "axpy<float, int>: calls gpublasDaxpy, expected gpublasSaxpy" in r.stderr
    # Exactly one: the double instantiation beside it was read and accepted.
    assert len(_problems(r.stderr)) == 1


def test_missing_instantiation_fails(env):
    r = env({_wrapper("axpy", "float, int"): ["cublasSaxpy_v2"]})
    assert r.returncode == 1
    assert "axpy<double, int>: no instantiation" in r.stderr
    # One, not two -- an empty disassembly would report both keys missing, so
    # this is what distinguishes "found one" from "parsed nothing".
    assert len(_problems(r.stderr)) == 1


def test_dispatch_branch_that_falls_off_the_end_fails(env):
    """A wrapper compiled with no branch taken calls nothing -- and is caught."""
    r = env({
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): [],
    })
    assert r.returncode == 1
    assert "axpy<double, int>: no instantiation" in r.stderr
    assert len(_problems(r.stderr)) == 1


def test_two_calls_from_one_instantiation_fails(env):
    r = env({
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2", "cublasDaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    })
    assert r.returncode == 1
    assert "one instantiation calls 2 vendor functions" in r.stderr
    # Two: the arity complaint, and the multiset mismatch it causes.
    assert len(_problems(r.stderr)) == 2


def test_wrapper_absent_from_the_table_fails(env):
    r = env({
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
        _wrapper("scal", "float, int"): ["cublasSscal_v2"],
    })
    assert r.returncode == 1
    assert "scal<float, int>: calls gpublasSscal but has no entry" in r.stderr
    assert len(_problems(r.stderr)) == 1


def test_unknown_type_does_not_silently_match(env):
    """A type missing from [type_names] shows up raw and fails, rather than
    being quietly normalised onto some other key."""
    r = env({
        _wrapper("axpy", "float2, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    })
    # float2 IS in [type_names.CUDA], so it normalises to gpuComplex -- which
    # the table does not list.
    assert r.returncode == 1
    assert "axpy<gpuComplex, int>" in r.stderr
    assert "axpy<float, int>: no instantiation" in r.stderr
    assert len(_problems(r.stderr)) == 2


# ---- overloads sharing template arguments --------------------------------

def test_overload_pair_passes_when_both_are_right(env):
    env.table.write_text(TABLE + OVERLOAD_ENTRY)
    r = env({
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
        _wrapper("scal", "float2, int") + " #1": ["cublasSscal_v2"],
        _wrapper("scal", "float2, int") + " #2": ["cublasSaxpy_v2"],
    })
    assert r.returncode == 0, r.stderr


def test_overload_pair_fails_when_both_call_the_same_one(env):
    env.table.write_text(TABLE + OVERLOAD_ENTRY)
    r = env({
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
        _wrapper("scal", "float2, int") + " #1": ["cublasSscal_v2"],
        _wrapper("scal", "float2, int") + " #2": ["cublasSscal_v2"],
    })
    assert r.returncode == 1
    assert "scal<gpuComplex, int>: calls gpublasSscal gpublasSscal" in r.stderr
    assert len(_problems(r.stderr)) == 1


# ---- forwarders ----------------------------------------------------------

def test_uninlined_forwarder_counts_as_its_alias(env):
    r = env({
        _wrapper("axpy", "float, int"): ["gpublasSaxpy@wwr.blas(int)"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    })
    assert r.returncode == 0, r.stderr


def test_forwarder_is_not_recognised_without_the_key(env):
    env.table.write_text(TABLE.replace('forwarder_module = "wwr.blas"\n', ""))
    r = env({
        _wrapper("axpy", "float, int"): ["gpublasSaxpy@wwr.blas(int)"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    })
    assert r.returncode == 1
    assert "axpy<float, int>: no instantiation" in r.stderr
    assert len(_problems(r.stderr)) == 1


# ---- the table file itself -----------------------------------------------

def test_template_argument_spacing_is_forgiving(env):
    env.table.write_text(TABLE.replace('"axpy<float, int>"', '"axpy<float,int>"'))
    r = env({
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    })
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(
    ("mangle", "message"),
    [
        (lambda t: t.replace("[check]\n", ""), "missing [check] table"),
        (lambda t: t.replace('prefix = "gpublas"\n', ""), "needs a string 'prefix'"),
        (lambda t: t.replace("[dispatch]\n", "[dispatch]\nx = 1\n"),
         "is not a non-empty list"),
        (lambda t: t.replace('"axpy<float, int>"', '"axpy"'), "is not name<targs>"),
        (lambda t: t.replace("[type_names.CUDA]", "[type_names.OPENCL]"),
         "expected one of CUDA, HIP"),
    ],
)
def test_malformed_table_is_rejected(env, mangle, message):
    env.table.write_text(mangle(TABLE))
    r = env({_wrapper("axpy", "float, int"): ["cublasSaxpy_v2"]})
    assert r.returncode != 0
    assert message in r.stderr


def test_duplicate_key_after_canonicalisation_is_rejected(env):
    env.table.write_text(TABLE + '"axpy<float,int>" = ["gpublasSaxpy"]\n')
    r = env({_wrapper("axpy", "float, int"): ["cublasSaxpy_v2"]})
    assert r.returncode != 0
    assert "duplicates" in r.stderr


def test_gpu_source_without_the_prefix_is_rejected(env, tmp_path):
    """A table pointed at the wrong gpu* module says so, rather than
    reporting every wrapper as calling nothing."""
    empty = tmp_path / "solver.cppm"
    empty.write_text(
        "WWR_FUNCTION(gpusolverDnCreate, cusolverDnCreate, hipsolverDnCreate)\n"
    )
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--objdump", "/bin/true", "--cxxfilt",
         shutil.which("cat"), "--backend", "CUDA", "--table", str(env.table),
         "--gpu-source", str(empty), str(tmp_path / "blas.o")],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "no WWR_FUNCTION(gpublas...) lines found" in r.stderr


# ---- --print -------------------------------------------------------------

def test_print_emits_a_pasteable_dispatch_block(env):
    r = env({
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
        _wrapper("scal", "float2, int"): ["cublasSscal_v2"],
    }, extra=("--print",))
    assert r.returncode == 0, r.stderr
    assert r.stdout == textwrap.dedent("""\
        [dispatch]
        "axpy<double, int>"     = ["gpublasDaxpy"]
        "axpy<float, int>"      = ["gpublasSaxpy"]
        "scal<gpuComplex, int>" = ["gpublasSscal"]
        """)


def test_print_keeps_a_repeated_callee_twice(env):
    """The check compares multisets, so --print must not deduplicate."""
    r = env({
        _wrapper("axpy", "float, int") + " #1": ["cublasSaxpy_v2"],
        _wrapper("axpy", "float, int") + " #2": ["cublasSaxpy_v2"],
    }, extra=("--print",))
    assert '["gpublasSaxpy", "gpublasSaxpy"]' in r.stdout


# ---- template arguments that contain commas --------------------------------

def test_nested_template_argument_is_not_split(env):
    """HIP's complex type is spelled HIP_vector_type<float, 2u>.

    split_targs exists solely because a plain str.split(",") tears that in two,
    turning scal<gpuComplex, int> into a three-argument key that matches nothing
    -- and the failure would read as a missing instantiation, pointing nowhere
    near the parser. It is the only hazard dispatch.py documents that nothing
    used to exercise end to end: every other HIP case here uses scalar types.
    """
    env.table.write_text(TABLE + '"scal<gpuComplex, int>" = ["gpublasSscal"]\n')
    r = env({
        _wrapper("axpy", "float, int"): ["hipblasSaxpy"],
        _wrapper("axpy", "double, int"): ["hipblasDaxpy"],
        _wrapper("scal", "HIP_vector_type<float, 2u>, int"): ["hipblasSscal"],
    }, backend="HIP")
    assert r.returncode == 0, r.stderr


# ---- the optional [check] namespace ---------------------------------------

def test_namespace_override_is_honoured(env):
    """[check] namespace defaults to wwr; the override was dead.

    Both shipped tables take the default, so nothing proved the key does
    anything. A silently ignored override would look like every wrapper having
    vanished.
    """
    env.table.write_text(
        TABLE.replace(
            'prefix = "gpublas"',
            'prefix = "gpublas"\nnamespace = "wwr::other"',
        )
    )
    ns = {"namespace": "wwr::other"}
    r = env({
        _wrapper("axpy", "float, int", **ns): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int", **ns): ["cublasDaxpy_v2"],
    })
    assert r.returncode == 0, r.stderr

    # ...and the default must then NOT match, or the override proves nothing.
    r = env({
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    })
    assert r.returncode == 1
    assert len(_problems(r.stderr)) == 2


# ---- one symbol, several gpu* names ---------------------------------------

def test_merged_backend_symbol_resolves_to_the_expected_name(env):
    """On HIP both status functions ARE hipblasStatusToString -- one symbol.

    The disassembly cannot distinguish them, so reporting the expected spelling
    is exact rather than lenient. Before this, whichever WWR_FUNCTION line came
    last silently won, so the same object could fail or pass depending on the
    order of two unrelated lines in src/blas.cppm.
    """
    for expected in ("gpublasGetStatusName", "gpublasGetStatusString"):
        env.table.write_text(TABLE + f'"name<int, int>" = ["{expected}"]\n')
        r = env({
            _wrapper("axpy", "float, int"): ["hipblasSaxpy"],
            _wrapper("axpy", "double, int"): ["hipblasDaxpy"],
            _wrapper("name", "int, int"): ["hipblasStatusToString"],
        }, backend="HIP")
        assert r.returncode == 0, f"{expected}: {r.stderr}"


def test_merged_backend_symbol_is_reported_when_unresolvable(env):
    """If the expectation names neither, there is a real question -- say so."""
    env.table.write_text(TABLE + '"name<int, int>" = ["gpublasSaxpy"]\n')
    r = env({
        _wrapper("axpy", "float, int"): ["hipblasSaxpy"],
        _wrapper("axpy", "double, int"): ["hipblasDaxpy"],
        _wrapper("name", "int, int"): ["hipblasStatusToString"],
    }, backend="HIP")
    assert r.returncode == 1
    assert "cannot distinguish them" in r.stderr
    assert "gpublasGetStatusName and gpublasGetStatusString" in r.stderr


def test_cuda_keeps_the_two_names_apart(env):
    """The same table under CUDA has two distinct symbols and must not merge."""
    env.table.write_text(TABLE + '"name<int, int>" = ["gpublasGetStatusName"]\n')
    r = env({
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
        _wrapper("name", "int, int"): ["cublasGetStatusString"],
    })
    assert r.returncode == 1
    assert "calls gpublasGetStatusString, expected gpublasGetStatusName" in r.stderr
    assert len(_problems(r.stderr)) == 1


# ---- several object files --------------------------------------------------

def test_same_symbol_in_two_objects_counts_once(env):
    """A module library is passed ALL its objects, and symbols can repeat.

    A wrapper emitted into two of them (weak/COMDAT -- the linker keeps one) was
    read as one instantiation calling two vendor functions, because the
    relocation lists were concatenated. That is a confusing failure on a
    perfectly correct tree, and nothing exercised more than one object before.
    """
    functions = {
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    }
    r = env(functions, objects=[env.object_with("dup.o", functions)])
    assert r.returncode == 0, r.stderr


def test_objects_disagreeing_about_a_symbol_is_reported(env):
    """Two definitions that call different things is a real anomaly, not noise.

    Resolving it silently either way would let the check's answer depend on the
    order the objects happened to arrive in.
    """
    other = env.object_with("odd.o", {
        _wrapper("axpy", "float, int"): ["cublasDaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    })
    r = env({
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    }, objects=[other])
    assert r.returncode != 0
    assert "with different calls" in r.stderr


def test_wrappers_split_across_objects_are_all_found(env):
    """The instantiations of one module can live in different objects."""
    other = env.object_with("more.o", {
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    })
    r = env({_wrapper("axpy", "float, int"): ["cublasSaxpy_v2"]}, objects=[other])
    assert r.returncode == 0, r.stderr


# ---- the stamp and the empty dump ------------------------------------------

def test_stale_stamp_is_removed_by_a_failing_run(env):
    """CMake names the stamp as this command's output; a failure must not leave
    a previous run's stamp sitting there as if the check had passed."""
    stamp = env.tmp_path / "stale.stamp"
    stamp.write_text("")
    r = env({
        _wrapper("axpy", "float, int"): ["cublasDaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
    }, extra=("--stamp", str(stamp)))
    assert r.returncode == 1
    assert not stamp.exists()


def test_print_with_nothing_found_says_so(env):
    """An empty [dispatch] block is indistinguishable from a module that
    dispatches nowhere, and --print's output is meant to be pasted."""
    r = env({"void unrelated::thing(int)": ["cublasSaxpy_v2"]}, extra=("--print",))
    assert r.returncode == 0
    assert r.stdout == ""
    assert "no wrappers found" in r.stderr


def test_non_string_dispatch_entry_is_rejected(env):
    env.table.write_text(TABLE + '"scal<float, int>" = [42]\n')
    r = env({_wrapper("axpy", "float, int"): ["cublasSaxpy_v2"]})
    assert r.returncode != 0
    assert "has non-string entries" in r.stderr


# ---- round trip ----------------------------------------------------------

def test_print_output_is_accepted_as_a_table(env):
    """--print claims to emit a pasteable [dispatch] block. Paste it and check.

    The exact-text test above pins the formatting; this pins the thing the
    formatting is FOR. Without it, --print and the checker could drift into
    disagreeing -- print quoting a key the parser rejects, or flattening the
    overload multiset -- and every other test in this file would stay green,
    because nothing else feeds one side's output to the other.
    """
    functions = {
        _wrapper("axpy", "float, int"): ["cublasSaxpy_v2"],
        _wrapper("axpy", "double, int"): ["cublasDaxpy_v2"],
        _wrapper("scal", "float2, int") + " #1": ["cublasSscal_v2"],
        _wrapper("scal", "float2, int") + " #2": ["cublasSaxpy_v2"],
    }
    printed = env(functions, extra=("--print",))
    assert printed.returncode == 0, printed.stderr
    assert printed.stdout.startswith("[dispatch]")

    head = TABLE.split("[dispatch]")[0]
    env.table.write_text(head + printed.stdout)
    r = env(functions)
    assert r.returncode == 0, r.stderr


# ---- the fakes, against the real tool ------------------------------------

SAMPLE = SCRIPT.parent / "objdump_sample.txt"


def _sample_objdump(tmp_path):
    """A stand-in objdump that replays the recorded real output."""
    fake = tmp_path / "objdump"
    fake.write_text(f'#!/bin/sh\ncat "{SAMPLE}"\n')
    fake.chmod(0o755)
    return fake


def _dispatch_module():
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import dispatch
    finally:
        sys.path.pop(0)
    return dispatch


def test_parser_reads_recorded_real_objdump_output(tmp_path):
    """The fidelity check that needs NO tools, so it never skips.

    objdump_sample.txt is real `llvm-objdump -dr --no-show-raw-insn` output,
    committed so this runs on a bare host too. That matters because the host is
    where dispatch.py gets EDITED -- the live test below only runs where the
    toolchain is, which is where the script runs but not where it changes. A
    tightened regex would otherwise go green for whoever wrote it.

    Two shapes in one sample: a function with several relocations, and one with
    none (a definition the parser must still record).
    """
    dispatch = _dispatch_module()
    calls = dispatch.read_calls(str(_sample_objdump(tmp_path)), [str(tmp_path / "x.o")])

    assert "caller_two_calls" in calls, (
        f"FUNC_HEADER_RE did not match real llvm-objdump output; got {sorted(calls)}"
    )
    assert calls["caller_two_calls"] == ["callee_one", "callee_two"], (
        f"RELOC_RE lost or reordered the callees; got {calls['caller_two_calls']}"
    )
    assert calls.get("caller_no_calls") == [], (
        "a function with no relocations must still be recorded as a definition"
    )


# The source test/shared/gen-objdump-sample.sh compiles. Kept identical here so
# the live tool's output can be compared with the recording, parse for parse.
SAMPLE_SOURCE = """\
void callee_one(void);
void callee_two(void);

void caller_two_calls(void) { callee_one(); callee_two(); }
void caller_no_calls(void) { }
"""


def test_sample_source_matches_the_generator():
    """The comparison below is only meaningful on the SAME program.

    test/shared/gen-objdump-sample.sh compiles the source that produced
    objdump_sample.txt; SAMPLE_SOURCE above is what the live test compiles. If
    they drift, the live test happily compares two different programs and
    reports a mismatch that is nothing to do with the toolchain -- or worse,
    agrees by luck. Comment said so; this enforces it.
    """
    gen = SCRIPT.parent / "gen-objdump-sample.sh"
    generator = gen.read_text()
    for line in SAMPLE_SOURCE.splitlines():
        if line.strip():
            assert line in generator, (
                f"{line!r} is in SAMPLE_SOURCE but not in "
                f"gen-objdump-sample.sh -- the two have drifted"
            )


def test_recorded_sample_still_matches_real_llvm_objdump(tmp_path):
    """Keeps objdump_sample.txt honest, which is the recording's weak point.

    A committed fixture is only evidence while it still resembles what the tool
    emits. The test above cannot notice a toolchain upgrade changing the format
    -- it would keep parsing the stale recording happily -- so this compiles the
    SAME source the sample was generated from, runs the real llvm-objdump, and
    requires the two to parse identically. A ROCm/LLVM bump that moves a column
    now fails here with "regenerate it" rather than leaving the fixture quietly
    obsolete.

    Skipped where the toolchain is absent (llvm-objdump is in
    DOCTOR_OPTIONAL_TOOLS), which is exactly the case the recording covers for.
    `-rs` is what makes the skip visible.
    """
    objdump = shutil.which("llvm-objdump")
    cc = shutil.which("clang") or shutil.which("cc")
    if not objdump or not cc:
        pytest.skip("needs llvm-objdump and a C compiler (both live in the image)")

    src = tmp_path / "sample.c"
    src.write_text(SAMPLE_SOURCE)
    obj = tmp_path / "sample.o"
    subprocess.run([cc, "-c", "-fno-inline", str(src), "-o", str(obj)], check=True)

    dispatch = _dispatch_module()
    live = dispatch.read_calls(objdump, [str(obj)])
    recorded = dispatch.read_calls(str(_sample_objdump(tmp_path)), [str(obj)])

    assert live == recorded, (
        "real llvm-objdump output no longer parses the same as "
        f"{SAMPLE.name} -- regenerate it with test/shared/gen-objdump-sample.sh "
        f"inside the image.\n  live:     {live}\n  recorded: {recorded}"
    )


# ---- the real tables in the tree -----------------------------------------

REPO_ROOT = SCRIPT.resolve().parents[2]

# (table, gpu source) as the BUILD pairs them. Discovered from the CMake call
# sites rather than hard-coded, so a dispatch check added later is covered
# without editing this file -- and so a test that claims to check the shipped
# tables cannot quietly be checking a stale list of them.
DISPATCH_CHECK_RE = re.compile(
    r"wwr_add_dispatch_check\s*\((.*?)\)", re.DOTALL
)


def _shipped_checks():
    found = []
    for cml in sorted(REPO_ROOT.glob("test/**/CMakeLists.txt")):
        for call in DISPATCH_CHECK_RE.findall(cml.read_text()):
            args = call.split()
            table = args[args.index("TABLE") + 1]
            gpu_source = args[args.index("GPU_SOURCE") + 1]
            found.append(
                pytest.param(
                    cml.parent / table,
                    REPO_ROOT / gpu_source,
                    id=f"{cml.parent.name}/{table}",
                )
            )
    return found


SHIPPED_CHECKS = _shipped_checks()


def test_shipped_checks_were_discovered():
    """The discovery above must not quietly find nothing.

    A typo in the regex would empty the parametrize list, and pytest reports a
    parametrize over an empty list as a skip -- so the table tests below would
    vanish rather than fail. This is the floor that stops that.
    """
    assert len(SHIPPED_CHECKS) >= 2, (
        "expected at least the blas and solver dispatch checks; "
        "wwr_add_dispatch_check call sites may have moved"
    )


@pytest.mark.parametrize(("table_path", "gpu_source"), SHIPPED_CHECKS)
def test_shipped_tables_load(table_path, gpu_source):
    """The tables are only read during a C++ build, which needs a container.

    This is the cheap half: they parse, their [check] blocks are complete, and
    every key canonicalises without collapsing onto another -- caught here
    rather than after a multi-minute module build.
    """
    assert table_path.is_file(), f"{table_path} does not exist"
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import dispatch
    finally:
        sys.path.pop(0)

    type_names = {}
    for backend in ("CUDA", "HIP"):
        table = dispatch.Table(str(table_path), backend)
        assert table.module.startswith("wwr.wrappers.")
        assert table.expected
        type_names[backend] = table.type_names

    # A table needs [type_names] only when its keys spell a type the demangler
    # writes differently -- a complex/vector type, or int64_t (emitted as 'long').
    # fft's keys are float/double alone, so it correctly declares none; a wholesale
    # missing mapping surfaces as a dispatch mismatch at build time. What this
    # cannot see and so pins here is asymmetry: declaring the mapping for one
    # backend but not the other silently under-canonicalises on the omitted one.
    assert bool(type_names["CUDA"]) == bool(type_names["HIP"]), (
        f"{table_path.name}: [type_names] present for one backend but not the "
        f"other: { {b: bool(v) for b, v in type_names.items()} }"
    )


@pytest.mark.parametrize(("table_path", "gpu_source"), SHIPPED_CHECKS)
def test_shipped_tables_name_real_functions(table_path, gpu_source):
    """Every gpu* function a table expects must exist in its gpu* module.

    The tables are hand-written on purpose, which is exactly why a typo in one
    is plausible. Left to the build, `gpublasIsamx` surfaces as "calls
    gpublasIsamax, expected gpublasIsamx" after several minutes of compiling --
    a message that reads like a dispatch bug rather than a spelling mistake.
    Both backends' columns are checked, so a HIP-only typo is caught on a CUDA
    box and vice versa.
    """
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import dispatch
    finally:
        sys.path.pop(0)

    assert gpu_source.is_file(), f"{gpu_source} does not exist"
    for backend in ("CUDA", "HIP"):
        table = dispatch.Table(str(table_path), backend)
        known = {
            alias
            for aliases in table.vendor_aliases(str(gpu_source)).values()
            for alias in aliases
        }
        expected = {fn for callees in table.expected.values() for fn in callees}
        unknown = sorted(expected - known)
        assert not unknown, (
            f"{table_path.name} ({backend}) expects {unknown}, which "
            f"WWR_FUNCTION({table.prefix}...) in {gpu_source.name} does not define"
        )
