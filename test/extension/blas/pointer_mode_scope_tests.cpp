// pointer_mode_scope_tests.cpp - Runtime contract of wwr.extension.blas's PointerModeScope
//
// The build-time half (test/extension/build_time/pointer_mode_scope.cppm) pins
// the types: that the guard's handle axis is the blas_handle concept, that it is
// a non-movable RAII value, and that it accepts both a co-owning shared_ptr and
// -- for a raw pointee -- a bare wwrblasHandle_t. What it cannot prove without a
// device is that the guard *does the thing*: record the mode live on a real
// handle, switch it to the target for the scope, and restore the recorded one on
// destruction. That is these tests, which need a live cuBLAS/hipBLAS handle and
// so live beside BlasHandleTests.
//
// The handle under test is a BlasHandleWrapper built from the reference
// DeviceHandle stream owner (wwr.test.shared), exactly as handle_tests.cpp builds
// it. Runtime, device-requiring, backend-neutral -- built and run for either
// WWR_GPU_BACKEND.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;        // wwrError_t, for the handle's device-access policy
import wwr.extension.common;   // the error_policy concept
import wwr.extension.handle;   // StreamBoundHandle (BlasHandleWrapper's base)
import wwr.extension.runtime;  // StreamWrapper, so owner->stream() has a complete type
import wwr.extension.blas;     // PointerModeScope, BlasHandleWrapper; re-exports wwr.blas
import wwr.test.shared.abort_policy;   // AbortPolicy for this file's instantiations
import wwr.test.shared.device_handle;  // the reference stream owner

namespace wwr::extension::test {

using Abort = AbortPolicy<wwrblasStatus_t>;
using AbortDev = AbortPolicy<wwrError_t>;
using BlasHandle = BlasHandleWrapper<Abort, Abort, DeviceHandle, AbortDev>;
using Scope = PointerModeScope<BlasHandle, Abort>;            // co-owns a wrapper pointee
using RawScope = PointerModeScope<wwrblasHandle_t, Abort>;    // raw pointee (borrow)

// Read the pointer mode live off the handle. EXPECT (not ASSERT) because this
// returns a value -- a failed get still yields a mode the caller's own EXPECT_EQ
// then reports against.
wwrblasPointerMode_t read_mode(wwrblasHandle_t h) {
  wwrblasPointerMode_t mode{};
  EXPECT_EQ(wwrblasGetPointerMode(h, &mode), WWRBLAS_STATUS_SUCCESS);
  return mode;
}

// Put the handle in a known starting mode, so a test never leans on the vendor
// default (HOST for both cuBLAS and hipBLAS, but set explicitly here regardless).
void set_mode(wwrblasHandle_t h, wwrblasPointerMode_t mode) {
  EXPECT_EQ(wwrblasSetPointerMode(h, mode), WWRBLAS_STATUS_SUCCESS);
}

TEST(PointerModeScopeTests, SetsTargetThenRestoresOnDestruction) {
  auto owner = std::make_shared<DeviceHandle>(0);
  auto handle = std::make_shared<BlasHandle>(owner);
  set_mode(handle->get(), WWRBLAS_POINTER_MODE_HOST);
  {
    Scope scope{handle, WWRBLAS_POINTER_MODE_DEVICE};
    EXPECT_EQ(read_mode(handle->get()), WWRBLAS_POINTER_MODE_DEVICE); // switched for the scope
  }
  EXPECT_EQ(read_mode(handle->get()), WWRBLAS_POINTER_MODE_HOST); // restored at }
}

TEST(PointerModeScopeTests, RestoresRecordedModeNotAHardcodedDefault) {
  // Start in DEVICE, the non-default mode: if the guard restored a hardcoded
  // default (HOST) rather than what it recorded, this is what would catch it.
  auto owner = std::make_shared<DeviceHandle>(0);
  auto handle = std::make_shared<BlasHandle>(owner);
  set_mode(handle->get(), WWRBLAS_POINTER_MODE_DEVICE);
  {
    Scope scope{handle, WWRBLAS_POINTER_MODE_HOST};
    EXPECT_EQ(read_mode(handle->get()), WWRBLAS_POINTER_MODE_HOST);
  }
  EXPECT_EQ(read_mode(handle->get()), WWRBLAS_POINTER_MODE_DEVICE); // back to the recorded mode
}

TEST(PointerModeScopeTests, NestedScopesRestoreInReverseOrder) {
  auto owner = std::make_shared<DeviceHandle>(0);
  auto handle = std::make_shared<BlasHandle>(owner);
  set_mode(handle->get(), WWRBLAS_POINTER_MODE_HOST);
  {
    Scope outer{handle, WWRBLAS_POINTER_MODE_DEVICE};
    EXPECT_EQ(read_mode(handle->get()), WWRBLAS_POINTER_MODE_DEVICE);
    {
      Scope inner{handle, WWRBLAS_POINTER_MODE_HOST};
      EXPECT_EQ(read_mode(handle->get()), WWRBLAS_POINTER_MODE_HOST);
    }
    // inner restored the mode it recorded -- outer's DEVICE, not the original HOST
    EXPECT_EQ(read_mode(handle->get()), WWRBLAS_POINTER_MODE_DEVICE);
  }
  EXPECT_EQ(read_mode(handle->get()), WWRBLAS_POINTER_MODE_HOST);
}

TEST(PointerModeScopeTests, RawHandleBorrowSetsAndRestores) {
  // The convenience ctor's raw-handle path: the guard BORROWS a bare
  // wwrblasHandle_t, so the real handle must outlive it -- here the BlasHandle
  // wrapper owns it for the whole test and the guard's inner scope ends first.
  auto owner = std::make_shared<DeviceHandle>(0);
  BlasHandle handle{owner};
  const wwrblasHandle_t raw = handle.get();
  set_mode(raw, WWRBLAS_POINTER_MODE_HOST);
  {
    RawScope scope{raw, WWRBLAS_POINTER_MODE_DEVICE}; // wraps `raw` in a shared_ptr internally
    EXPECT_EQ(read_mode(raw), WWRBLAS_POINTER_MODE_DEVICE);
  }
  EXPECT_EQ(read_mode(raw), WWRBLAS_POINTER_MODE_HOST);
}

TEST(PointerModeScopeTests, RawPointeeViaSharedPtrSetsAndRestores) {
  // The primitive the borrow ctor delegates to: a shared_ptr<wwrblasHandle_t>
  // pointee. It co-owns the pointer value, not the GPU handle, so the wrapper
  // still owns the real handle for the test's duration.
  auto owner = std::make_shared<DeviceHandle>(0);
  BlasHandle handle{owner};
  set_mode(handle.get(), WWRBLAS_POINTER_MODE_HOST);
  {
    RawScope scope{std::make_shared<wwrblasHandle_t>(handle.get()), WWRBLAS_POINTER_MODE_DEVICE};
    EXPECT_EQ(read_mode(handle.get()), WWRBLAS_POINTER_MODE_DEVICE);
  }
  EXPECT_EQ(read_mode(handle.get()), WWRBLAS_POINTER_MODE_HOST);
}

} // namespace wwr::extension::test
