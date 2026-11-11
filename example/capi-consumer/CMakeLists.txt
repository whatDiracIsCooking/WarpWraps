# A standalone PURE-C project that consumes an INSTALLED wwr through its C ABI.
#
# The sibling consumer/ is a C++23 project: it COMPILES wwr's installed module
# interface sources, so it needs the whole clang + libc++ + `import std`
# toolchain set up before project(). This one needs none of that -- the capi
# ships as a PREBUILT libwwr_capi.so plus a plain-C header, exactly so a non-C++
# FFI consumer (Rust, ctypes, ...) can link it with no C++ toolchain at all.
# Keeping this project C-only, separate from consumer/, is what proves that
# promise: a C++ leak in the capi install would show up as this failing to
# configure or link.
#
# Deliberately not part of the main build (not added by example/CMakeLists.txt),
# same as consumer/: the point is that it reaches wwr only through
# find_package(wwr). devtools/install-check.sh builds it against a throwaway
# install prefix, which is the only way the capi install rules can be verified
# rather than assumed.

cmake_minimum_required(VERSION 4.2)

project(wwr_capi_consumer_example LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# wwrConfig.cmake's toolchain check compares the consumer's CXX compiler against
# the clang + libc++ wwr was built with, because a MODULE consumer recompiles
# wwr's interface sources. This consumer compiles none -- it links a prebuilt
# .so and includes a plain-C header -- so that check is irrelevant here and,
# with no CXX language enabled, would only warn about an empty compiler id. Opt
# out of it, which is exactly the switch wwrConfig documents for a consumer who
# knows their toolchain is compatible. This is also the proof that the capi
# needs NO C++ toolchain to consume.
set(WWR_SKIP_TOOLCHAIN_CHECK ON)

# CONFIG mode: wwr ships a package config (wwrConfig.cmake), not a Find module.
# Point CMAKE_PREFIX_PATH at the install prefix to locate it. The `capi`
# component is REQUIRED, so an install that did not ship the C ABI
# (-DWWR_INSTALL_CAPI left OFF) is refused here by name rather than later as an
# unresolved wwr::capi -- see wwrConfig.cmake.in section 1g. install-check.sh
# only builds this when it installed the capi, so the component is always
# satisfiable here.
find_package(wwr 0.1 REQUIRED COMPONENTS capi)

# The one line that is actually about wwr. wwr::capi is the exported target for
# the prebuilt libwwr_capi.so; linking it brings in the installed
# <wwr/wwr_capi.h> include root. The capi's own wwr module dependencies are
# PRIVATE on the shared library, so none leak here -- this C project links the
# .so and nothing else.
add_executable(capi_consumer main.c)
target_link_libraries(capi_consumer PRIVATE wwr::capi)
