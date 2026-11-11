#!/usr/bin/env bash
# Prove the installed package works, by consuming it the way a stranger would.
#
#   devtools/install-check.sh [--preset NAME] [--prefix DIR] [--keep] [--no-run]
#
# Four steps, and the third is the only one that proves anything:
#
#   1. configure wwr
#   2. cmake --install it into a throwaway prefix
#   3. configure example/consumer against ONLY that prefix, and build it
#   4. run the resulting binary
#
# Step 3 is the point. An install rule that runs without error still says
# nothing about whether the result is usable -- a missing module source, a
# header that did not travel next to the .cppm that includes it, a usage
# requirement that was PRIVATE and so never exported: all of those install
# perfectly cleanly and fail only in a consumer. example/consumer is a
# standalone project that knows nothing about this source tree and reaches
# wwr through find_package alone, so building it is a real answer and
# `cmake --install` succeeding is not.
#
# WHY THIS EXISTS AS ITS OWN TIER. devtools/cpp-tier.sh builds and ctests the
# tree in place; nothing in it ever installs, so nothing in it can catch an
# export-set regression. A module added to src/ is picked up by the install
# sweep automatically (cmake/wwr_install.cmake reads the buildsystem back),
# but a module added with a PRIVATE compile requirement is exactly the change
# that passes cpp-tier.sh and breaks consumers. This is the tier that notices.
#
# THE TOOLCHAIN IS NOT ON YOUR HOST -- clang-20 with libc++'s module manifest
# and CMake 4.2 live in docker/Dockerfile.base, same as for cpp-tier.sh:
#
#   devtools/devcontainer.sh shell -c devtools/install-check.sh
#
# Step 4 needs a GPU; steps 1-3 do not need one to BUILD, but the `default`
# preset's CMAKE_CUDA_ARCHITECTURES=native queries a device at configure time,
# so on a GPU-less box pass --preset ci-cuda, which pins 86. The consumer
# binary exits 77 when it finds no device, which this script reports as a skip
# rather than a failure; --no-run stops before that and is what
# .github/workflows/ci.yml uses, since its runner has no card at all.
#
# Flags:
#   --preset NAME   wwr configure preset (default: CMAKE_PRESET from
#                   devtools/config.sh).
#   --prefix DIR    install prefix (default: a mktemp -d, removed on exit).
#   --keep          keep the prefix and the consumer build dir, and print
#                   where they are. For inspecting what actually got installed.
#   --no-run        stop after building the consumer. Use on a box with no
#                   device when you only want the compile-and-link answer.
#   --extension     configure wwr with -DWWR_INSTALL_EXTENSION=ON so the
#                   package also ships src/extension, and let example/consumer
#                   consume it (WWR_HAS_EXTENSION drives that automatically).
#                   Without this the extension layer is not installed and the
#                   consumer builds against the core package alone.
#   --no-wrappers   configure wwr with -DWWR_INSTALL_WRAPPERS=OFF so the package
#                   ships the wwr* core alone, without src/wrappers. The
#                   consumer then builds against the core (WWR_HAS_WRAPPERS
#                   drives that automatically) -- its gemm runs on the raw wwr*
#                   layer. The default ships the wrappers and the consumer links
#                   them, which is what proves the wrapper install rules.
#   --capi          configure wwr with -DWWR_INSTALL_CAPI=ON so the package also
#                   ships the C ABI (src/capi, the prebuilt libwwr_capi.so + the
#                   plain-C wwr_capi.h), then build+run example/capi-consumer --
#                   a PURE-C project that find_package()s wwr, links wwr::capi and
#                   calls the C functions against the INSTALLED .so + header. This
#                   is the only gate that proves the capi install rules (the .so
#                   landed, the header travelled, wwr::capi exports, no C++ leak),
#                   the same way the C++ consumer above proves the module layers.
#
# Exit status is the first failing step's.
#: -- help stops here --
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

preset=$CMAKE_PRESET
prefix=""
keep=0
run=1
extension=0
wrappers=1
capi=0

while [ $# -gt 0 ]; do
  case $1 in
    --preset) preset=${2:?--preset needs a value}; shift 2 ;;
    --prefix) prefix=${2:?--prefix needs a value}; shift 2 ;;
    --keep) keep=1; shift ;;
    --no-run) run=0; shift ;;
    --extension) extension=1; shift ;;
    --no-wrappers) wrappers=0; shift ;;
    --capi) capi=1; shift ;;
    -h | --help)
      sed -n '2,/^#: -- help stops here --$/p' "${BASH_SOURCE[0]}" |
        sed 's/^# \{0,1\}//; $d'
      exit 0
      ;;
    *) echo "install-check.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

# A build directory of its own. Sharing one with cpp-tier.sh would mean the two
# tiers reconfigure each other's cache back and forth -- a full rebuild each
# way -- which is the same reason every preset in CMakePresets.json has its own
# binaryDir.
build_dir=$REPO_ROOT/build-install-check
consumer_build=$REPO_ROOT/build-install-check-consumer
capi_consumer_build=$REPO_ROOT/build-install-check-capi-consumer

if [ -z "$prefix" ]; then
  prefix=$(mktemp -d -t wwr-install-XXXXXX)
  created_prefix=1
else
  mkdir -p "$prefix"
  prefix=$(cd "$prefix" && pwd)
  created_prefix=0
fi

cleanup() {
  if [ "$keep" -eq 1 ]; then
    echo
    echo "kept: install prefix   $prefix"
    echo "kept: consumer build   $consumer_build"
    [ "$capi" -eq 1 ] && echo "kept: capi consumer    $capi_consumer_build"
    return
  fi
  [ "$created_prefix" -eq 1 ] && rm -rf "$prefix"
  rm -rf "$consumer_build" "$capi_consumer_build"
}
trap cleanup EXIT

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
step "1/4  configure wwr (preset: $preset)"
# ---------------------------------------------------------------------------
# -B overrides the preset's own binaryDir so this tier keeps its cache separate
# from cpp-tier.sh's, per the comment above.
# Pass WWR_INSTALL_EXTENSION, WWR_INSTALL_WRAPPERS and WWR_INSTALL_CAPI
# explicitly BOTH ways, never just when flipped: this tier reuses $build_dir
# across runs, and a cached value from an earlier --extension / --no-wrappers /
# --capi run would otherwise persist into a plain run and quietly install a
# different layer set than the caller asked for.
install_args=(-DWWR_INSTALL=ON)
if [ "$extension" -eq 1 ]; then
  install_args+=(-DWWR_INSTALL_EXTENSION=ON)
else
  install_args+=(-DWWR_INSTALL_EXTENSION=OFF)
fi
if [ "$wrappers" -eq 1 ]; then
  install_args+=(-DWWR_INSTALL_WRAPPERS=ON)
else
  install_args+=(-DWWR_INSTALL_WRAPPERS=OFF)
fi
if [ "$capi" -eq 1 ]; then
  install_args+=(-DWWR_INSTALL_CAPI=ON)
else
  install_args+=(-DWWR_INSTALL_CAPI=OFF)
fi
cmake --preset "$preset" -B "$build_dir" "${install_args[@]}"

# ---------------------------------------------------------------------------
step "2/4  build and install into $prefix"
# ---------------------------------------------------------------------------
build_args=()
[ -n "${BUILD_JOBS:-}" ] && build_args+=(-j "$BUILD_JOBS")
cmake --build "$build_dir" "${build_args[@]}"
cmake --install "$build_dir" --prefix "$prefix"

echo
echo "installed:"
echo "  $(find "$prefix" -name '*.cppm' | wc -l) module interface sources"
echo "  $(find "$prefix" -name '*.a' | wc -l) static libraries"
echo "  $(find "$prefix" \( -name '*.h' -o -name '*.cuh' \) | wc -l) headers"

# The package config is what a consumer finds first; if it is missing, nothing
# downstream can work and the error there would not say so.
for required in \
  "lib/cmake/wwr/wwrConfig.cmake" \
  "lib/cmake/wwr/wwr-targets.cmake"; do
  if [ ! -f "$prefix/$required" ]; then
    echo "install-check.sh: FAIL -- $required was not installed" >&2
    exit 1
  fi
done

# With --extension, prove the three distinctive kinds of extension artifact
# actually landed -- a mirrored src/ header, a module interface source, and a
# device-kernel archive -- before the consumer (step 3) is asked to compile
# against them. These are the pieces an install-rule regression is most likely to
# drop, and a missing one gives a clearer message here than a compile or link
# error there.
if [ "$extension" -eq 1 ]; then
  ext_missing=0
  ext_bridge="include/wwr/extension/init_state/init_state_bridge.h"
  ext_module="include/wwr/modules/extension/runtime/interface.cppm"
  [ -f "$prefix/$ext_bridge" ] ||
    { echo "install-check.sh: FAIL -- extension header not installed: $ext_bridge" >&2; ext_missing=1; }
  [ -f "$prefix/$ext_module" ] ||
    { echo "install-check.sh: FAIL -- extension module source not installed: $ext_module" >&2; ext_missing=1; }
  # LIBDIR is lib or lib64 depending on the distro, so match on name rather than
  # a fixed path.
  find "$prefix" -name 'libwwr.extension.random_normal.device.a' | grep -q . ||
    { echo "install-check.sh: FAIL -- extension device archive not installed" >&2; ext_missing=1; }
  [ "$ext_missing" -eq 0 ] || exit 1
  echo "  extension layer installed (headers, module sources, device archives)"
fi

# The wrappers layer ships by default and is dropped with --no-wrappers. Prove
# both directions from the prefix before the consumer is built: the two most
# telling artifacts are a per-target module source and the dispatch header that
# has to travel next to it (rules 2 and 3). Present when shipped, absent when
# dropped -- catching an install rule that ignored WWR_INSTALL_WRAPPERS in
# either direction.
wrap_module="include/wwr/modules/wrappers/blas/interface.cppm"
wrap_header="include/wwr/wrappers/common/dispatch_sdcz.h"
if [ "$wrappers" -eq 1 ]; then
  wrap_missing=0
  [ -f "$prefix/$wrap_module" ] ||
    { echo "install-check.sh: FAIL -- wrapper module source not installed: $wrap_module" >&2; wrap_missing=1; }
  [ -f "$prefix/$wrap_header" ] ||
    { echo "install-check.sh: FAIL -- wrapper dispatch header not installed: $wrap_header" >&2; wrap_missing=1; }
  [ "$wrap_missing" -eq 0 ] || exit 1
  echo "  wrappers layer installed (module sources, dispatch headers)"
else
  wrap_leaked=0
  [ -e "$prefix/$wrap_module" ] &&
    { echo "install-check.sh: FAIL -- wrapper module source installed despite --no-wrappers: $wrap_module" >&2; wrap_leaked=1; }
  [ -e "$prefix/$wrap_header" ] &&
    { echo "install-check.sh: FAIL -- wrapper dispatch header installed despite --no-wrappers: $wrap_header" >&2; wrap_leaked=1; }
  [ "$wrap_leaked" -eq 0 ] || exit 1
  echo "  wrappers layer omitted (core-only install)"
fi

# The C ABI ships only with --capi. Prove its two distinctive artifacts landed
# before the C consumer (the extra step below) is asked to link them -- the
# prebuilt shared library (matched by name, LIBDIR is lib or lib64) and the
# plain-C header in the wwr include root. A missing one gives a clearer message
# here than a link or #include error in the consumer.
capi_header="include/wwr/wwr_capi.h"
if [ "$capi" -eq 1 ]; then
  capi_missing=0
  find "$prefix" -name 'libwwr_capi.so*' | grep -q . ||
    { echo "install-check.sh: FAIL -- capi shared library not installed" >&2; capi_missing=1; }
  [ -f "$prefix/$capi_header" ] ||
    { echo "install-check.sh: FAIL -- capi header not installed: $capi_header" >&2; capi_missing=1; }
  [ "$capi_missing" -eq 0 ] || exit 1
  echo "  C ABI installed (libwwr_capi.so, wwr_capi.h)"
fi

# ---------------------------------------------------------------------------
step "3/4  configure and build example/consumer against the install"
# ---------------------------------------------------------------------------
# CMAKE_PREFIX_PATH is the ONLY thing connecting the consumer to wwr. No
# source path, no build directory, nothing from this tree -- if find_package
# cannot work from the install prefix alone, this step is where it shows.
#
# But it must ALSO keep finding the libraries the CUDA and HIP images place off
# any default search path (cuTENSOR, nvCOMP, cuGraph, ROCm-DS), which they point
# at through CMAKE_PREFIX_PATH in the image ENV. A -D cache entry SHADOWS the
# environment variable of the same name rather than extending it, so passing
# only "$prefix" would blind the consumer to every image-provided prefix the
# moment a wrapper links one. Extend instead of replace: the temp prefix first,
# then the image's. The two forms differ -- a -D value is a CMake list
# (';'-separated) while the POSIX ENV is ':'-separated -- so translate ':' to
# ';' on the inherited value before appending it.
prefix_path=$prefix
[ -n "${CMAKE_PREFIX_PATH:-}" ] && prefix_path="$prefix;${CMAKE_PREFIX_PATH//:/;}"
rm -rf "$consumer_build"
cmake -S "$REPO_ROOT/example/consumer" -B "$consumer_build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$prefix_path"
cmake --build "$consumer_build" "${build_args[@]}"

# example/consumer builds one `consumer` target unconditionally, so a
# successful build (guaranteed by `set -e`) that leaves no binary is not a
# "disabled example" -- it is a real defect in the consumer build, and step 3
# is the only step that proves anything. Fail on it rather than passing.
if [ ! -x "$consumer_build/consumer" ]; then
  echo
  echo "install-check.sh: FAIL -- consumer build succeeded but produced no" >&2
  echo "  '$consumer_build/consumer' binary." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# With --capi, the SAME proof for the C ABI: configure example/capi-consumer --
# a standalone PURE-C project -- against the install prefix alone and build it.
# It find_package(wwr COMPONENTS capi), links wwr::capi and #includes the
# installed <wwr/wwr_capi.h>, so a successful build here means the prebuilt
# libwwr_capi.so, its header and its export entry all re-attach through
# find_package with no C++ toolchain in sight. Same prefix_path as the C++
# consumer, so it keeps finding any image-provided prefixes too.
if [ "$capi" -eq 1 ]; then
  step "3b/4  configure and build example/capi-consumer against the install"
  rm -rf "$capi_consumer_build"
  cmake -S "$REPO_ROOT/example/capi-consumer" -B "$capi_consumer_build" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$prefix_path"
  cmake --build "$capi_consumer_build" "${build_args[@]}"
  if [ ! -x "$capi_consumer_build/capi_consumer" ]; then
    echo
    echo "install-check.sh: FAIL -- capi consumer build succeeded but produced" >&2
    echo "  no '$capi_consumer_build/capi_consumer' binary." >&2
    exit 1
  fi
  # Prove the consumer links the INSTALLED libwwr_capi.so, not some in-tree
  # build: its sole wwr dependency must resolve inside the throwaway prefix.
  # ldd reads the dynamic deps; a resolution pointing anywhere but $prefix means
  # the consumer is not actually exercising the install, which is the one thing
  # this tier exists to prove.
  capi_so=$(ldd "$capi_consumer_build/capi_consumer" 2>/dev/null |
    awk '/libwwr_capi\.so/ { print $3 }')
  case "${capi_so:-}" in
    "$prefix"/*) linked_install=1 ;;
    *) linked_install=0 ;;
  esac
  if [ "$linked_install" -eq 0 ]; then
    echo
    echo "install-check.sh: FAIL -- capi consumer does not link the INSTALLED" >&2
    echo "  libwwr_capi.so (resolved to '${capi_so:-<none>}', not under $prefix)." >&2
    exit 1
  fi
  echo "  capi consumer links the installed lib: $capi_so"
fi

# ---------------------------------------------------------------------------
if [ "$run" -eq 0 ]; then
  step "4/4  skipped (--no-run)"
  echo
  echo "install-check.sh: PASS (built, not run)"
  exit 0
fi

step "4/4  run the consumer(s)"
# ---------------------------------------------------------------------------
# A consumer exiting 77 means "built and linked, but no reachable GPU" -- a
# skip, not a failure (see example/README.md). Run each consumer that was built,
# treat 77 as a pass, and fail on any other non-zero.
run_consumer() {
  local bin=$1 name=$2
  set +e
  "$bin"
  local status=$?
  set -e
  if [ "$status" -eq 77 ]; then
    echo
    echo "install-check.sh: $name built and linked; not run -- no GPU"
    return 0
  fi
  if [ "$status" -ne 0 ]; then
    echo
    echo "install-check.sh: FAIL -- $name exited $status" >&2
    exit "$status"
  fi
}

run_consumer "$consumer_build/consumer" "consumer"
[ "$capi" -eq 1 ] && run_consumer "$capi_consumer_build/capi_consumer" "capi consumer"

echo
echo "install-check.sh: PASS"
