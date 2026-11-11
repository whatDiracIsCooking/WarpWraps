# wwr_install.cmake — install rules, the export set, and the package config.
#
# A C++23 module package ships module interface SOURCES (.cppm), not BMIs (a BMI
# is not portable), and the consumer's own build recompiles them. Three rules
# follow, each enforced below and explained in full in cmake/README.md, "Install
# and the CMake package":
#
#   1. Every compile requirement of a .cppm must reach the consumer — a PRIVATE
#      requirement is not exported, so it becomes a broken install.
#   2. Module sources need per-target destinations (six modules are each rooted
#      at a file named interface.cppm; one shared dir would collide).
#   3. A header a module unit #includes is installed next to those sources.
#
# The consumer-facing half of the contract (same compiler, standard library and
# backend) is stated and enforced in wwrConfig.cmake.in.

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Where the pieces of the installed package live, all relative to the install
# prefix. `modules` holds module interface SOURCES, not BMIs (see above); it is
# under include/ because that is where a "things the consumer's compiler reads"
# directory belongs, not because anything #includes from it.
set(WWR_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/wwr")
set(WWR_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/wwr")
set(WWR_INSTALL_MODULEDIR "${WWR_INSTALL_INCLUDEDIR}/modules")
# share/wwr -- arch-independent sidecar data, not headers or libs. Only the
# wwr-sys Rust crate skeleton lives here today (under rust/wwr-sys, gated on
# WWR_INSTALL_CAPI); it is crate SOURCE a Rust consumer path/git-depends on.
set(WWR_INSTALL_DATADIR "${CMAKE_INSTALL_DATADIR}/wwr")

# ---------------------------------------------------------------------------
# Internal helpers (underscore-prefixed -- not part of the public API)
# ---------------------------------------------------------------------------

# Collect every library target defined in `dir`, and in `dir`'s subdirectories
# when RECURSE is given.
#
# This is deliberately a sweep rather than a hand-maintained list. The list it
# would replace is exactly the kind that goes stale silently: a module added to
# src/ and forgotten here would not fail anything, it would just be missing from
# the package, and the first report would come from a consumer. Reading the
# buildsystem back means "installed" and "defined under src/" cannot drift.
#
# UTILITY targets (the compile_time_tests umbrella) and ALIAS targets (the ::
# spellings, which are not in BUILDSYSTEM_TARGETS at all) fall out of the TYPE
# filter on their own.
function(_wwr_collect_library_targets dir out_var)
  cmake_parse_arguments(
    _c
    "RECURSE"
    ""
    ""
    ${ARGN}
  )

  set(_found "")
  get_property(
    _targets
    DIRECTORY "${dir}"
    PROPERTY BUILDSYSTEM_TARGETS
  )
  foreach(_target IN LISTS _targets)
    get_target_property(_type ${_target} TYPE)
    if(_type MATCHES "^(STATIC|SHARED|MODULE|OBJECT|INTERFACE)_LIBRARY$")
      list(APPEND _found ${_target})
    endif()
  endforeach()

  if(_c_RECURSE)
    get_property(
      _subdirs
      DIRECTORY "${dir}"
      PROPERTY SUBDIRECTORIES
    )
    foreach(_subdir IN LISTS _subdirs)
      _wwr_collect_library_targets("${_subdir}" _from_subdir RECURSE)
      list(APPEND _found ${_from_subdir})
    endforeach()
  endif()

  set(${out_var}
      "${_found}"
      PARENT_SCOPE
  )
endfunction()

# The installed module-source directory for `target`, mirroring its position
# under src/ -- see rule 2 in this file's header comment.
function(_wwr_module_destination target out_var)
  get_target_property(_source_dir ${target} SOURCE_DIR)
  file(RELATIVE_PATH _relative "${PROJECT_SOURCE_DIR}/src" "${_source_dir}")
  # The wwr* layer lives directly in src/, so its relative path is empty and its
  # module sources install at the MODULEDIR root; everything else mirrors to a
  # subdirectory (cuda/, wrappers/blas, ...).
  if(_relative STREQUAL "")
    set(_destination "${WWR_INSTALL_MODULEDIR}")
  else()
    set(_destination "${WWR_INSTALL_MODULEDIR}/${_relative}")
  endif()
  set(${out_var}
      "${_destination}"
      PARENT_SCOPE
  )
endfunction()

# Install the non-module headers that a target's module units #include by
# relative path, next to the installed module sources -- see rule 3.
#
# Nothing here has to name them: a header sitting in a module target's own
# source directory is, by this project's layout, exactly a header that
# directory's module units include by relative path. Globbing at configure time
# is safe for the same reason it is usually not: the set is not an input to any
# build rule, only to an install rule, so a stale glob costs a reconfigure and
# never a wrong build.
function(_wwr_install_module_adjacent_headers target destination)
  get_target_property(_source_dir ${target} SOURCE_DIR)
  file(GLOB _headers "${_source_dir}/*.h" "${_source_dir}/*.cuh")
  if(_headers)
    install(FILES ${_headers} DESTINATION "${destination}")
  endif()
endfunction()

# The exported name for a target: its in-tree `::` alias without the `wwr::`
# namespace. Strip the leading `wwr.`/`wwr_` and turn the remaining dots into
# `::`, so the dotted target `wwr.extension.fft` gets EXPORT_NAME
# `extension::fft` and, under install(EXPORT ... NAMESPACE wwr::) below, an
# installed consumer links the SAME `wwr::extension::fft` that in-tree
# CMakeLists already link -- one spelling everywhere (issue #183). `wwr_backend`
# and `wwr_module_flags` fold in by the same rule (the leading separator is `_`)
# to `wwr::backend` / `wwr::module_flags`; the device archives (`wwr.*.device`,
# which carry no in-tree alias) to `wwr::…::device`, which the WHOLE_ARCHIVE
# $<INSTALL_INTERFACE:> genexes in src/extension name by hand.
function(_wwr_export_name target out_var)
  string(REGEX REPLACE "^wwr[._]" "" _name "${target}")
  string(REPLACE "." "::" _name "${_name}")
  set(${out_var}
      "${_name}"
      PARENT_SCOPE
  )
endfunction()

# Install one collected target into the wwr-targets export set. Three shapes:
#
#   * an INTERFACE library has no artifact -- it is in the set only for its
#     usage requirements and because targets linking it cannot be exported;
#   * a module library ships its FILE_SET CXX_MODULES interface sources to a
#     per-target destination (rule 2) plus its module-adjacent headers (rule 3);
#   * a plain archive (a wwr_add_gpu_device_library .device target under
#     src/extension) has no module file set, so it gets NO FILE_SET clause --
#     naming a file set the target does not have is an error, not a no-op -- and
#     only installs its archive; init_state/random_normal pull it in
#     WHOLE_ARCHIVE.
function(_wwr_install_target target)
  # Unify the exported name with the in-tree `::` alias (issue #183). Set before
  # any install(TARGETS ... EXPORT), which is what reads EXPORT_NAME into the
  # generated wwr-targets.cmake.
  _wwr_export_name(${target} _export_name)
  set_target_properties(${target} PROPERTIES EXPORT_NAME "${_export_name}")

  get_target_property(_type ${target} TYPE)
  if(_type STREQUAL "INTERFACE_LIBRARY")
    install(TARGETS ${target} EXPORT wwr-targets)
    return()
  endif()

  get_target_property(_module_sets ${target} CXX_MODULE_SETS)
  if(_module_sets)
    _wwr_module_destination(${target} _module_dir)
    # cmake-lint: disable=E1122
    # One DESTINATION per artifact kind is how install(TARGETS) is spelled:
    # ARCHIVE, LIBRARY, RUNTIME and FILE_SET each take their own. cmake-lint
    # models the command as a flat argument list and reads the repeats as a
    # duplicated keyword.
    install(
      TARGETS ${target}
      EXPORT wwr-targets
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
              FILE_SET CXX_MODULES
              DESTINATION "${_module_dir}"
    )
    _wwr_install_module_adjacent_headers(${target} "${_module_dir}")
  else()
    install(
      TARGETS ${target}
      EXPORT wwr-targets
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
  endif()
endfunction()

# Install the C ABI (src/capi) as prebuilt binaries: the per-unit shared objects
# and the umbrella.
#
# The capi is the one wwr layer that ships as PREBUILT shared objects
# (libwwr_capi.so + the per-unit libwwr_capi_<unit>.so) rather than as module
# source -- a non-C++ consumer links them, they import the wwr modules but
# export none. So they are NOT part of the src/-sweep (whose non-recursive src/
# collection never reaches a target defined in src/capi, like src/thrust); they
# are installed here explicitly, gated on WWR_INSTALL_CAPI (effect only when
# WWR_INSTALL is ON, mirroring WWR_INSTALL_EXTENSION). The targets themselves
# build unconditionally with the rest of src/; only this install is gated.
#
# Each install(TARGETS ... EXPORT wwr-targets) puts a .so (+ soname) in the lib
# dir and adds its wwr::capi[::<unit>] alias to the export set, so a consumer's
# find_package(wwr) + target_link_libraries(... wwr::capi) (the umbrella, the
# drop-in for the old single .so) OR ... wwr::capi::blas (one unit, the isolated
# dep) resolves. EXPORT_NAME is set the same way the sweep sets it, so each
# exports under its in-tree alias, one spelling everywhere. The capi's wwr deps
# are PRIVATE on each shared lib, so none leak into its INTERFACE_LINK_LIBRARIES
# -- the consumer just links the .so.
#
# Every capi target carries the INCLUDES DESTINATION so wwr::capi (the umbrella)
# and wwr::capi::<unit> each resolve <wwr/wwr_capi.h> and the per-unit headers
# from the install's include roots. The headers install below, mirroring the
# gen/ tree flat under include/wwr/, so the umbrella's own #include "common.h" /
# "complex/wwr_capi_complex.h" spellings (root-relative to its own dir) resolve
# unchanged -- the historical <wwr/wwr_capi.h> keeps working AND still reaches
# its siblings, with no forwarder.
#
# TWO include roots are exported: include/ for the <wwr/...> spellings, AND
# include/wwr/ so a UNIT header's bare #include "common.h" resolves -- a unit
# header lives one dir down (include/wwr/complex/), so common.h is not its
# sibling and the quote-include falls through to the -I roots, exactly as
# in-tree against the gen/ root. Without the second root, <wwr/wwr_capi.h> ->
# complex/... -> common.h fails in the installed consumer (install-check catches
# this precisely).
#
# Called from wwr_install_package() BEFORE its install(EXPORT), which these
# install(TARGETS ... EXPORT) must precede.
function(_wwr_install_capi)
  if(NOT WWR_INSTALL_CAPI)
    return()
  endif()
  # The umbrella + every per-unit target. EXPORT_NAME per target gives the
  # in-tree alias (wwr::capi, wwr::capi::complex, ...). INCLUDES DESTINATION
  # adds the include root (include/, for <wwr/...>) and the wwr/ subdir (for a
  # unit header's bare common.h include) to each INTERFACE_INCLUDE_DIRECTORIES.
  # sparse / fft build unconditionally like blas; tensor / comp only under their
  # WWR_WITH_* gates (the wwr.tensor / wwr.comp modules they import exist only
  # then), so install them only when built -- the same gating the module targets
  # carry in src/CMakeLists.txt.
  # The umbrella plus every per-unit target. sparse / fft build unconditionally
  # like blas / blaslt / solver / rand / rtc; tensor / comp only under their
  # WWR_WITH_* gates (the wwr.tensor / wwr.comp modules they import exist only
  # then), so install them only when built -- the same gating src/CMakeLists.txt
  # carries on the module targets.
  # tx builds unconditionally like sparse / fft (wwr.tx has no WWR_WITH_* gate);
  # ccl only under WWR_WITH_CCL (the wwr.ccl module it imports exists only
  # then), the same gating as tensor / comp.
  set(_capi_targets
      wwr.capi wwr.capi.complex wwr.capi.runtime_api wwr.capi.blas
      wwr.capi.blaslt wwr.capi.solver wwr.capi.rand wwr.capi.rtc
      wwr.capi.sparse wwr.capi.fft wwr.capi.tx
  )
  if(WWR_WITH_TENSOR)
    list(APPEND _capi_targets wwr.capi.tensor)
  endif()
  if(WWR_WITH_COMP)
    list(APPEND _capi_targets wwr.capi.comp)
  endif()
  if(WWR_WITH_CCL)
    list(APPEND _capi_targets wwr.capi.ccl)
  endif()
  foreach(_capi_target IN LISTS _capi_targets)
    _wwr_export_name(${_capi_target} _capi_export_name)
    set_target_properties(
      ${_capi_target} PROPERTIES EXPORT_NAME "${_capi_export_name}"
    )
    install(
      TARGETS ${_capi_target}
      EXPORT wwr-targets
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
      INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
                           "${WWR_INSTALL_INCLUDEDIR}"
    )
  endforeach()
  # The vendor-free C headers: the whole gen/ header tree mirrored FLAT under
  # include/wwr/ (common.h -> include/wwr/common.h, the umbrella ->
  # include/wwr/wwr_capi.h, each unit -> include/wwr/<unit>/wwr_capi_<unit>.h).
  # So <wwr/wwr_capi.h> resolves exactly as before AND its root-relative
  # #includes of common.h / complex/... resolve against include/wwr/ -- the
  # whole ABI from one umbrella #include, same as the old single header.
  # install(DIRECTORY) with a .h filter is safe here: the set is an input only
  # to an install rule.
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/src/capi/gen/"
    DESTINATION "${WWR_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.h"
  )
  # The wwr-sys Rust crate skeleton, so a Rust user can path/git-depend on the
  # installed tree (#340). It is SOURCE, not a built artifact -- cargo cannot
  # build a C++23-module target, so the crate only links the prebuilt
  # libwwr_capi.so installed above; its build.rs reads WWR_CAPI_LIB_DIR to find
  # it (the lib dir of this very prefix). Cargo.toml, build.rs and src/ (lib.rs
  # + the generated ffi.rs) are everything the crate needs; the end-to-end
  # consume-from-the-installed-prefix check is #341.
  install(FILES "${PROJECT_SOURCE_DIR}/src/capi/wwr-sys/Cargo.toml"
                "${PROJECT_SOURCE_DIR}/src/capi/wwr-sys/build.rs"
          DESTINATION "${WWR_INSTALL_DATADIR}/rust/wwr-sys"
  )
  install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/capi/wwr-sys/src"
          DESTINATION "${WWR_INSTALL_DATADIR}/rust/wwr-sys"
  )
endfunction()

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Emit every install rule for the project, plus the CMake package.
#
# Call this from the top-level CMakeLists.txt AFTER every add_subdirectory that
# defines a library, because it reads the buildsystem back to decide what to
# install and a target that does not exist yet cannot be found.
function(wwr_install_package)
  # The backend's own wrapper directory is the only part of src/ that is not
  # added unconditionally, so the sweep covers whichever one this build chose.
  if(WWR_GPU_BACKEND STREQUAL "CUDA")
    set(_backend_dir "${PROJECT_SOURCE_DIR}/src/cuda")
  else()
    set(_backend_dir "${PROJECT_SOURCE_DIR}/src/hip")
  endif()

  _wwr_collect_library_targets("${_backend_dir}" _backend_targets RECURSE)
  # The wwr* layer is defined directly in src/CMakeLists.txt, so it is collected
  # from src/ NON-recursively; recursing would re-collect src/cuda, src/hip and
  # src/wrappers, which are swept separately above and below.
  _wwr_collect_library_targets("${PROJECT_SOURCE_DIR}/src" _gpu_targets)
  # The wwr* Thrust layer (src/thrust) is the one core-layer target in its own
  # subdirectory, so the non-recursive src/ sweep above misses it. Collected
  # explicitly. Ships always, like the rest of the core wwr* layer -- it is not
  # gated like the wrappers/extension sweeps below. wwrConfig.cmake.in already
  # resolves the CCCL / rocthrust dependency wwr.thrust links.
  _wwr_collect_library_targets(
    "${PROJECT_SOURCE_DIR}/src/thrust" _thrust_targets
  )
  # The wrappers layer (src/wrappers) ships only when opted into -- see
  # WWR_INSTALL_WRAPPERS in the top-level CMakeLists.txt. Gated like the
  # extension sweep below so a core-only install omits its targets from the
  # export set. Nothing else in src/ links a wwr.wrappers.* target, so dropping
  # them leaves the rest of the export set intact.
  set(_wrapper_targets "")
  if(WWR_INSTALL_WRAPPERS)
    _wwr_collect_library_targets(
      "${PROJECT_SOURCE_DIR}/src/wrappers" _wrapper_targets RECURSE
    )
  endif()

  # The extension layer (src/extension) ships only on request -- see
  # WWR_INSTALL_EXTENSION in the top-level CMakeLists.txt. Swept RECURSE so both
  # its module libraries AND the wwr_add_gpu_device_library archives
  # (wwr.extension.*.device) join the export set: init_state and random_normal
  # name those archives under $<INSTALL_INTERFACE:> (WHOLE_ARCHIVE), so
  # install(EXPORT) refuses the whole set unless they are members too.
  set(_extension_targets "")
  if(WWR_INSTALL_EXTENSION)
    _wwr_collect_library_targets(
      "${PROJECT_SOURCE_DIR}/src/extension" _extension_targets RECURSE
    )
  endif()

  # wwr_module_flags is defined in the top-level CMakeLists.txt rather than
  # under src/, so the sweep above does not reach it -- but three src/cuda
  # targets link it PUBLIC, which puts it in their INTERFACE_LINK_LIBRARIES and
  # so makes install(EXPORT) refuse the whole export set until it is a member
  # too. Collected non-recursively: recursing from the top would pull in test/
  # and deps/ (GoogleTest), none of which belongs in this package.
  _wwr_collect_library_targets("${PROJECT_SOURCE_DIR}" _root_targets)
  list(
    FILTER
    _root_targets
    INCLUDE
    REGEX
    "^wwr_module_flags$"
  )

  set(_targets ${_root_targets} ${_backend_targets} ${_gpu_targets}
               ${_thrust_targets} ${_wrapper_targets} ${_extension_targets}
  )
  list(REMOVE_DUPLICATES _targets)

  list(LENGTH _targets _count)
  message(STATUS "Install: ${_count} targets in the wwr package")

  # Each target is installed on its own rather than in one install(TARGETS ...)
  # call, because FILE_SET CXX_MODULES needs a per-target DESTINATION (rule 2).
  # They all name the same EXPORT set, which is what makes them one package.
  # _wwr_install_target picks the right install shape per target type.
  foreach(_target IN LISTS _targets)
    _wwr_install_target(${_target})
  endforeach()

  # The .h/.cuh headers that are reached by include path rather than by sitting
  # next to a module source. Installing src/'s shape under include/wwr/ is
  # what makes the include spellings resolve unchanged -- see the
  # INSTALL_INTERFACE include directories on the targets themselves.
  #
  # The wwr* layer's backend-switch headers (backend.h, selected_backend.h,
  # device_guard.h) and its device-side .cuh headers are included bare (e.g.
  # "backend.h") and live directly in src/, so a NON-recursive glob is
  # exactly this set and they install at the include root. install(DIRECTORY)
  # is wrong here: it would recurse into src/cuda, src/hip and src/wrappers,
  # whose headers are handled separately (wrappers', below; the backends have
  # none). A configure-time glob is safe here: the set is an input only to an
  # install rule, never to a build rule -- a stale glob costs a reconfigure, not
  # a wrong build.
  file(GLOB _gpu_layer_headers "${PROJECT_SOURCE_DIR}/src/*.h"
       "${PROJECT_SOURCE_DIR}/src/*.cuh"
  )
  install(FILES ${_gpu_layer_headers}
          DESTINATION "${WWR_INSTALL_INCLUDEDIR}"
  )

  # The wwr* Thrust layer's device headers (src/thrust/*.cuh), included through
  # the src/ root ("thrust/sort.cuh", ...) so the subtree is mirrored under
  # include/wwr/thrust for those spellings to resolve unchanged after install --
  # the same shape as the wrappers/ and extension/ subtrees below. It is
  # unconditional: the layer is part of the core wwr* layer, not opt-in.
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/src/thrust/"
    DESTINATION "${WWR_INSTALL_INCLUDEDIR}/thrust"
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.cuh"
  )

  # The src/detail/ shared include fragments (runtime_api_names.h, ...) -- the
  # macro-driven name lists pasted (never imported) by the module's purview,
  # runtime.h's device section AND wwr/runtime_api.h, each `#include`d
  # root-relative as "detail/...". Mirrored under include/wwr/detail/ so that
  # spelling resolves against the package include root for the two header paths
  # that travel (runtime.h and wwr/runtime_api.h); the module compiles in-tree
  # only, so its copy never needs the installed fragment.
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/src/detail/"
    DESTINATION "${WWR_INSTALL_INCLUDEDIR}/detail"
    FILES_MATCHING
    PATTERN "*.h"
  )

  # The src/wwr/ header-only consumption path (wwr/runtime_api.h, ...) -- the
  # self-contained .h a non-module consumer #includes instead of importing.
  # Mirrored under include/wwr/wwr/ so `#include "wwr/runtime_api.h"` resolves
  # against the package include root, exactly as the wrappers/ and extension/
  # subtrees below are mirrored. It pulls the shared name list via
  # "detail/runtime_api_names.h" (installed just above) and the flag-dance
  # preamble + selected_backend.h, which the src/*.h glob above installs.
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/src/wwr/"
    DESTINATION "${WWR_INSTALL_INCLUDEDIR}/wwr"
    FILES_MATCHING
    PATTERN "*.h"
  )

  # The shared dispatch header under src/wrappers, included through src/ (e.g.
  # "wrappers/common/dispatch_sdcz.h"), mirrored to include/wwr/wrappers.
  # Shipped only when WWR_INSTALL_WRAPPERS added the wrapper targets to the
  # sweep above -- kept in step with that sweep, both keyed off the same option,
  # so a core-only install carries neither the modules nor their headers.
  if(WWR_INSTALL_WRAPPERS)
    install(
      DIRECTORY "${PROJECT_SOURCE_DIR}/src/wrappers/"
      DESTINATION "${WWR_INSTALL_INCLUDEDIR}/wrappers"
      FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.cuh"
    )
  endif()

  # The extension layer's headers, shipped only when WWR_INSTALL_EXTENSION added
  # its targets to the sweep above. parallel_for.cuh and the two *_bridge.h are
  # #included by the extension module units and by parallel_for.cuh through the
  # src/-root spelling ("extension/parallel_for/parallel_for.cuh", ...), so the
  # extension subtree is mirrored under include/wwr/extension for those
  # spellings to resolve unchanged after install -- the same shape, and the
  # same reasoning, as the wrappers directory above. (The state type those
  # bridges name, wwrrandState, rides rand.h -- a src/-root header installed
  # by the src/*.h glob above, not part of this extension subtree.) Kept next
  # to the target sweep that needs them, so install-check.sh --extension
  # actually compiles a consumer against them rather than assuming the rule.
  if(WWR_INSTALL_EXTENSION)
    install(
      DIRECTORY "${PROJECT_SOURCE_DIR}/src/extension/"
      DESTINATION "${WWR_INSTALL_INCLUDEDIR}/extension"
      FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.cuh"
    )
  endif()

  # The C ABI (src/capi -> wwr.capi), installed as a prebuilt binary outside the
  # src/-sweep. Gated on WWR_INSTALL_CAPI (effect only when WWR_INSTALL is ON);
  # must precede the install(EXPORT) below, which it does. See the helper.
  _wwr_install_capi()

  # Install the export set and generate + install the CMake package config.
  _wwr_install_package_config()
endfunction()

# Install the wwr-targets export set and the generated package config/version
# files. Split out of wwr_install_package() so that function stays within the
# cmake-lint statement budget; it is the final phase and runs after every
# install(TARGETS ... EXPORT wwr-targets) above (including the capi's), which
# install(EXPORT) requires.
function(_wwr_install_package_config)
  # CXX_MODULES_DIRECTORY is what makes this an installable module package
  # rather than a broken one: without it the export names targets whose module
  # file sets no consumer can see, and `import wwr.wrappers.blas;` fails to
  # resolve against a package that otherwise installed cleanly.
  install(
    EXPORT wwr-targets
    FILE wwr-targets.cmake
    NAMESPACE wwr::
    DESTINATION "${WWR_INSTALL_CMAKEDIR}"
    CXX_MODULES_DIRECTORY cxx-modules
  )

  # SameMinorVersion, not SameMajorVersion: at 0.x there is no major-version
  # promise to make, and C++23 module packages have a narrower compatibility
  # story than ordinary libraries anyway (see wwrConfig.cmake.in).
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/wwrConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
  )

  # Recorded into the config so a consumer's build is checked against the
  # configuration this package was actually built with, rather than discovering
  # the mismatch as a link error or, worse, a wrong warp size at runtime.
  set(WWR_PACKAGE_BACKEND "${WWR_GPU_BACKEND}")
  set(WWR_PACKAGE_WARP_SIZE "${WWR_WARP_SIZE}")
  # Booleans (ON/OFF), not strings: wwrConfig.cmake.in tests them directly and
  # exposes each as a package component (`wrappers`, `extension`). Each is kept
  # in step with its own sweep and header rule above, all keyed off the same
  # option.
  set(WWR_PACKAGE_HAS_WRAPPERS "${WWR_INSTALL_WRAPPERS}")
  set(WWR_PACKAGE_HAS_EXTENSION "${WWR_INSTALL_EXTENSION}")
  # WWR_WITH_CCL is a BUILD flag, not an install flag: when OFF its targets were
  # never defined, so the target sweeps above already omit them from the export
  # set with no special-casing (the "read the buildsystem back" design). This
  # only tells wwrConfig.cmake.in to skip the NCCL/RCCL find_dependency and to
  # refuse the `ccl` component. No install(DIRECTORY) EXCLUDE is needed: the ccl
  # dirs ship no .h/.cuh, so the extension header glob finds nothing in them.
  # comp/tensor are identical -- their extension dirs are header-free too, so
  # the extension header glob needs no EXCLUDE for them either.
  set(WWR_PACKAGE_HAS_CCL "${WWR_WITH_CCL}")
  set(WWR_PACKAGE_HAS_COMP "${WWR_WITH_COMP}")
  set(WWR_PACKAGE_HAS_TENSOR "${WWR_WITH_TENSOR}")
  # The C ABI ships only under WWR_INSTALL_CAPI (same shape as HAS_EXTENSION
  # above): _wwr_install_capi() is the install gate, and this tells
  # wwrConfig.cmake.in whether wwr::capi is in the export set so the `capi`
  # component and example/capi-consumer can gate on it.
  set(WWR_PACKAGE_HAS_CAPI "${WWR_INSTALL_CAPI}")
  set(WWR_PACKAGE_CXX_COMPILER_ID "${CMAKE_CXX_COMPILER_ID}")
  set(WWR_PACKAGE_CXX_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
  set(WWR_PACKAGE_CXX_STANDARD_LIBRARY "${CMAKE_CXX_STANDARD_LIBRARY}")

  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/wwrConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/wwrConfig.cmake"
    INSTALL_DESTINATION "${WWR_INSTALL_CMAKEDIR}"
  )

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/wwrConfig.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/wwrConfigVersion.cmake"
          DESTINATION "${WWR_INSTALL_CMAKEDIR}"
  )
endfunction()
