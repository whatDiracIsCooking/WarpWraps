# wwr.extension.tensor -- link-check only (NO_RUN). See link_check.cpp's header
# for the full why: cuTENSOR / hipTENSOR abort in a library global constructor
# on a GPU-less host, so no binary linking this module can be RUN -- or even
# `--gtest_list_tests`'d -- on the GPU-less CI runner. A GoogleTest suite is
# therefore impossible here (its SuiteListIsComplete guard would launch the
# binary and abort), so this follows the vendor module's precedent
# (test/gpu/tensor_link_check.cpp, test/hip's hiptensor NO_RUN executable): it
# is BUILT so every wwrtensor* symbol resolves at link -- the link is the proof
# -- with the move/copy contract pinned by compile-time static_asserts, but
# carrying no ctest entry and never launched. success_code's zero-enumerator
# contract is pinned by the static_assert in tensor_error.cppm.
wwr_add_test_executable(
  NAME
  extension_tensor_link_check
  MAIN
  link_check.cpp
  LINK_PRIVATE
  wwr::extension::tensor
  wwr::extension::common
  wwr::extension::handle
  NO_RUN
)

# Built as part of the compile-time tier too, so a tensor-binding regression is
# caught by the cheap check and not only by a full build.
if(TARGET compile_time_tests)
  add_dependencies(compile_time_tests extension_tensor_link_check)
endif()
