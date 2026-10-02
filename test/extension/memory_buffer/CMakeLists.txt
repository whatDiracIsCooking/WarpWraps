# Memory buffer tests
# RAII contract, views, bounds/overflow rejection, stream-ordered alloc/free
#
# First suite migrated to GoogleTest. Note what is gone relative to the
# test_module version, because the rest of the migration follows this shape:
#
#   * no wwr_add_cxx_module_library() wrapping the test source -- the
#     test TU is a plain .cpp compiled straight into the executable
#   * no WHOLE_ARCHIVE, because a TU linked directly into an executable cannot
#     have its self-registering static objects dropped; that hack only existed
#     to keep them alive across a static-library boundary
#   * no main.cpp -- GTest::gtest_main supplies main()
#   * one ctest entry per suite instead of one per binary, so `ctest -R` can
#     select one suite and a skip no longer reads as a pass

wwr_add_gtest_executable(
  NAME
  extension_memory_buffer_tests
  SOURCES
  buffer_tests.cpp
  host_memory_tests.cpp
  allocation_failure_tests.cpp
  LINK
  wwr::runtime_api
  wwr::extension::common
  wwr::extension::handle
  wwr::extension::runtime
  wwr::extension::memory_buffer
  wwr::test::shared::device_handle
)

# One ctest entry per suite. Add a suite here when you add one to the test file,
# or the SuiteListIsComplete guard test will fail and say so.
#
# HostAccessibleOwningBufferTest is a TYPED_TEST_SUITE over the two owning,
# host-accessible, bare-count kinds; TYPES names them in the same order as the
# ::testing::Types list so its ctest entries read HostAccessibleOwningBufferTest
# <HostBuffer> rather than .../0. Unified is handle-backed, so it is not in that
# typed suite -- its own contract lives in the plain UnifiedBufferTests suite.
wwr_add_gtest_suite_tests(
  TARGET
  extension_memory_buffer_tests
  SUITES
  AbortPolicyDeathTest
  AbortPolicyTests
  AllocationFailureTests
  BufferViewTests
  CopyAndMemsetTests
  DeviceBufferTests
  DeviceHandleTests
  GpuCheckDeathTest
  GpuCheckTests
  HostAccessibleBufferTests
  HostBufferTests
  HostMemoryAllocFailureTests
  HostMemoryAllocTests
  HostMemoryErrorTests
  SizeOverflowRejectionTests
  UnifiedBufferTests
  TYPED_SUITES
  HostAccessibleOwningBufferTest
  TYPES
  HostBuffer
  PinnedBuffer
  TIMEOUT
  60
  REQUIRES_GPU
)

# The no_sanitizer-labeled suites deliberately force a real allocation failure
# to prove the failure path stays coherent, and every sanitizer treats that
# intentional failure as an error of its own: compute-sanitizer's memcheck
# counts the cudaErrorMemoryAllocation, and ASAN's allocator aborts on the
# oversized host malloc (allocation-size-too-big) -- so under either sanitizer's
# error-exit they would fail an otherwise-clean tree. The no_sanitizer label is
# the ctest analog of pytest's no_sanitizer marker: the asan ctest preset and
# both compose sanitizer services (asan, compute-sanitizer) exclude it with
# `ctest -LE no_sanitizer`, while every non-sanitized ctest run still executes
# the suites.
#
# The label is confined to the two suites that actually force the failure --
# AllocationFailureTests (the buffer wrappers, in allocation_failure_tests.cpp)
# and HostMemoryAllocFailureTests (the raw std_malloc SIZE_MAX case, split off
# from HostMemoryAllocTests in host_memory_tests.cpp). Their siblings allocate
# nothing (SizeOverflowRejectionTests) or allocate successfully
# (HostMemoryAllocTests), so they are safe under every sanitizer and stay
# unlabeled to keep running there.
#
# APPEND, not `set_tests_properties(... PROPERTIES LABELS no_sanitizer)`. That
# form REPLACES the property, and REQUIRES_GPU above has already put `gpu` on
# every entry in this target -- so the plain form would silently strip it from
# exactly these two suites, and `ctest -LE gpu` on a GPU-less runner would then
# try to run them and fail. A test carries two independent labels here; each one
# has to append.
set_property(
  TEST AllocationFailureTests HostMemoryAllocFailureTests
  APPEND
  PROPERTY LABELS no_sanitizer
)
