# A standalone project that consumes an INSTALLED wwr.
#
# Deliberately not part of the main build: the whole point is that it knows
# nothing about wwr's source tree and reaches it only through
# find_package(wwr). devtools/install-check.sh builds it that way -- install
# wwr to a throwaway prefix, then configure this against that prefix -- which
# is the only way an install rule can actually be verified rather than assumed.
#
# Everything above project() is the toolchain setup a consumer needs too. It is
# not wwr boilerplate leaking out; it is the consequence of consuming a C++23
# module package. The consumer's build COMPILES wwr's module interface
# sources (a BMI is not portable, so the package ships sources -- see
# cmake/wwr_install.cmake in the wwr tree), and those sources `import
# std;`. So the flags that make `import std` work have to be set here, before
# project(), exactly as wwr's own top-level CMakeLists.txt sets them.

cmake_minimum_required(VERSION 4.2)

# The experimental UUID for `import std`, tied to the CMake version -- look the
# current value up in that release's Help/dev/experimental.rst. A stale UUID is
# ignored silently and every `import std` then fails to resolve.
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")

set(CMAKE_CXX_STANDARD_LIBRARY "libc++")
set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")
set(CMAKE_CXX_COMPILER "clang++")

project(wwr_consumer_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

add_link_options($<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>)
add_link_options($<$<COMPILE_LANGUAGE:CXX>:-lc++abi>)

# The one line that is actually about wwr.
#
# CONFIG mode: wwr ships a package config (wwrConfig.cmake), not a Find
# module. Point CMAKE_PREFIX_PATH at the install prefix to locate it.
find_package(wwr 0.1 REQUIRED)

message(STATUS "wwr backend:   ${WWR_GPU_BACKEND}")
message(STATUS "wwr warp size: ${WWR_WARP_SIZE}")

add_executable(consumer main.cpp)

# The :: spellings are the exported names. Linking them brings in wwr's
# module sources, which this build compiles -- `import wwr.wrappers.blas;`
# in main.cpp resolves because of these lines and nothing else.
#
# The consumer imports two gpu* modules directly (wwr.runtime_api for device
# memory, wwr.blas for the handle), so it names them here. The wrapper
# targets each LINK_PUBLIC the gpu* module they wrap, so those names would
# arrive transitively too, but the example uses runtime_api/blas in its own
# right and so depends on them in its own right.
target_link_libraries(
  consumer
  PRIVATE wwr::wwr.runtime_api
          wwr::wwr.blas
          wwr::wwr.wrappers.blas
          wwr::wwr.wrappers.solver
          wwr::wwr.wrappers.fft
          wwr::wwr.wrappers.sparse
          wwr::wwr.wrappers.tx
)

# The extension layer is optional in the package (installed with
# -DWWR_INSTALL_EXTENSION=ON). WWR_HAS_EXTENSION comes from wwrConfig.cmake and
# says whether this install shipped it. When it did, link two extension modules
# and let main.cpp exercise them under WWR_CONSUMER_HAS_EXTENSION, which is the
# only thing that proves the extension install rules (module sources, the src/
# header tree, and the WHOLE_ARCHIVE device archives) actually re-attach in a
# find_package consumer, exactly as the gpu*/wrapper half above does.
if(WWR_HAS_EXTENSION)
  message(STATUS "wwr extension: available")
  target_link_libraries(
    consumer PRIVATE wwr::wwr.extension.runtime wwr::wwr.extension.random_normal
  )
  target_compile_definitions(consumer PRIVATE WWR_CONSUMER_HAS_EXTENSION)
else()
  message(STATUS "wwr extension: not in this install")
endif()
