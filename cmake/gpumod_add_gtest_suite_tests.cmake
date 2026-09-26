# gpumod_add_gtest_suite_tests — register one ctest entry per GoogleTest suite,
# instead of one per binary or one per case, plus a drift guard.
#
#   gpumod_add_gtest_suite_tests(
#       TARGET       extension_math_tests
#       SUITES       PlainSuiteA PlainSuiteB      # TEST() / TEST_F() suites
#       TYPED_SUITES FooTests BarTests            # TYPED_TEST_SUITE() suites
#       TYPES        float double cuComplex cuDoubleComplex
#       TIMEOUT      120)
#
# See cmake/README.md, "Tests", for why per suite (the cost measurement), how
# TYPES maps typed suites, and why a target may need more than one call.

# Internal: registers the drift guard once per target, after every
# gpumod_add_gtest_suite_tests() call in the directory has contributed its
# suite names. Not meant to be called directly.
function(_gpumod_register_gtest_suite_guard target)
  get_property(_suites GLOBAL PROPERTY _gpumod_gtest_suites_${target})
  add_test(
    NAME ${target}.SuiteListIsComplete
    COMMAND
      ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:${target}> "-DEXPECTED=${_suites}"
      -P ${CMAKE_SOURCE_DIR}/cmake/gpumod_check_gtest_suites.cmake
  )
  set_tests_properties(${target}.SuiteListIsComplete PROPERTIES TIMEOUT 60)
endfunction()

# Register a GoogleTest binary with ctest as one entry per suite, plus the
# <target>.SuiteListIsComplete drift guard. TYPED_SUITES x TYPES expands to one
# entry per instantiation. See cmake/README.md, "Tests", for the why.
function(gpumod_add_gtest_suite_tests)
  cmake_parse_arguments(
    _GST
    ""
    "TARGET;TIMEOUT"
    "SUITES;TYPED_SUITES;TYPES"
    ${ARGN}
  )

  _gpumod_require_args("gpumod_add_gtest_suite_tests" _GST TARGET)
  if(NOT _GST_TIMEOUT)
    set(_GST_TIMEOUT 120)
  endif()
  if(_GST_TYPED_SUITES AND NOT _GST_TYPES)
    message(
      FATAL_ERROR
        "gpumod_add_gtest_suite_tests: TYPED_SUITES given without TYPES"
    )
  endif()

  # Plain suites: one entry, filter `Suite.*`.
  foreach(_suite IN LISTS _GST_SUITES)
    add_test(NAME ${_suite} COMMAND ${_GST_TARGET} --gtest_filter=${_suite}.*)
    set_tests_properties(${_suite} PROPERTIES TIMEOUT ${_GST_TIMEOUT})
  endforeach()

  # Typed suites: one entry per (suite, type), filter `Suite/<index>.*`.
  list(LENGTH _GST_TYPES _type_count)
  foreach(_suite IN LISTS _GST_TYPED_SUITES)
    set(_index 0)
    while(_index LESS _type_count)
      list(GET _GST_TYPES ${_index} _type_name)
      add_test(NAME "${_suite}<${_type_name}>"
               COMMAND ${_GST_TARGET} --gtest_filter=${_suite}/${_index}.*
      )
      set_tests_properties(
        "${_suite}<${_type_name}>" PROPERTIES TIMEOUT ${_GST_TIMEOUT}
      )
      math(EXPR _index "${_index} + 1")
    endwhile()
  endforeach()

  # Drift guard: suite names accumulate on a global property and the guard is
  # deferred to end-of-directory-scope, so a target split across several calls
  # is checked against the union (cmake/README.md, "Tests").
  set_property(
    GLOBAL APPEND PROPERTY _gpumod_gtest_suites_${_GST_TARGET} ${_GST_SUITES}
                           ${_GST_TYPED_SUITES}
  )

  get_property(
    _guard_scheduled GLOBAL PROPERTY _gpumod_gtest_guard_${_GST_TARGET}
  )
  if(NOT _guard_scheduled)
    set_property(GLOBAL PROPERTY _gpumod_gtest_guard_${_GST_TARGET} TRUE)
    # EVAL CODE bakes the target name into the deferred call as a literal.
    # A plain `DEFER CALL f("${_GST_TARGET}")` does not work: deferred
    # arguments are re-evaluated when the call finally runs, by which point
    # this function's scope is gone and the argument expands to nothing --
    # which surfaces as `$<TARGET_FILE:>` failing to parse.
    set(_guard_fn _gpumod_register_gtest_suite_guard)
    cmake_language(
      EVAL CODE "cmake_language(DEFER CALL ${_guard_fn} \"${_GST_TARGET}\")"
    )
  endif()
endfunction()
