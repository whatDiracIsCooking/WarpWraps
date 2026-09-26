#!/bin/sh
# Regenerate test/shared/objdump_sample.txt from the real llvm-objdump.
#
#   docker run --rm -v "$PWD:$PWD" -w "$PWD" -u "$(id -u):$(id -g)" \
#     gpumod:latest test/shared/gen-objdump-sample.sh > test/shared/objdump_sample.txt
#
# Must run where llvm-objdump and clang are, i.e. inside the image -- the point
# of the sample is that it is the REAL tool's output, committed so the parser
# test needs neither. test_recorded_sample_still_matches_real_llvm_objdump
# compiles this same source with the live tool and fails if the two stop
# agreeing, which is the signal to re-run this.
#
# SAMPLE_SOURCE in test/shared/test_dispatch.py must stay identical to the
# source below, or that comparison is against a different program.
set -e
d=$(mktemp -d)
cat > "$d/sample.c" <<'C'
/* Two callers so the sample covers a function with several relocations and a
   function with none -- both shapes dispatch.py's parser has to handle. */
void callee_one(void);
void callee_two(void);

void caller_two_calls(void) { callee_one(); callee_two(); }
void caller_no_calls(void) { }
C
clang -c -fno-inline "$d/sample.c" -o "$d/sample.o"
# Normalise the leading "<path>: file format ..." banner, which carries a
# mktemp path and would otherwise make the sample differ on every run.
llvm-objdump -dr --no-show-raw-insn "$d/sample.o" | sed "s#^.*/sample\.o:#sample.o:#"
rm -rf "$d"
