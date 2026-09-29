// basic.cpp - DeviceBoundHandle runtime tests
//
// The device-bound CRTP layer's whole reason to exist is an *ordering*: the
// owning device must be made current (wwrSetDevice) BEFORE Derived::create runs,
// so the handle is created on the intended device. That ordering is arranged by
// smuggling the select into the argument slot of the BaseHandle mem-initializer
// (DeviceBoundHandle::on_device), which is sequenced before BaseHandle's body
// calls create. This suite pins that it actually holds -- and it does so on a
// single device, because the select is made observable by forcing it to FAIL
// (an out-of-range ordinal), not by owning a second GPU.
//
// A plain TU (see test/extension/runtime/basic.cpp for the shape): self-
// registering tests compiled straight into the executable, GTest::gtest_main
// supplies main().

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;
import wwr.extension.common;
import wwr.extension.handle;
import wwr.test.shared.abort_policy;

namespace wwr::extension::test {
namespace {

// Ordered log of construction events. It lives at namespace scope, NOT on the
// wrapper: create() runs from inside BaseHandle's constructor, before any
// DeviceBoundHandle-or-derived member is alive, so the log it writes to cannot
// be one of those members. Single-threaded tests, so a plain vector is fine.
std::vector<std::string> g_events;

using Abort = AbortPolicy<wwrError_t>;

// A non-aborting device-access policy that makes the select observable. gpu_check
// only calls handle_error on FAILURE, so on the success path (a valid device)
// this stays silent -- exactly why the ordering test uses an invalid ordinal.
// It records both to instance state (read back via device_policy(), proving the
// select's error is threaded by reference into the ctor and then moved into
// policy_device_ intact) and to g_events (proving order relative to create()).
struct SpyDevicePolicy {
  using error_type = wwrError_t;
  wwrError_t last_error = wwrSuccess;
  int calls = 0;
  void handle_error(const wwrError_t error, std::source_location) noexcept {
    last_error = error;
    ++calls;
    g_events.push_back("select");
  }
};

// A device-bound wrapper over a stand-in handle: no vendor resource is created
// (create() only logs and nulls the handle), so the only real runtime calls are
// the wwrSetDevice / wwrGetDevice the layer itself makes.
struct fake_handle_tag;
using FakeHandle = fake_handle_tag *;

class DeviceSpyWrapper
    : public DeviceBoundHandle<FakeHandle, DeviceSpyWrapper, Abort, Abort, SpyDevicePolicy> {
public:
  using DeviceBoundHandle::DeviceBoundHandle;
  void create(FakeHandle *h, std::source_location) {
    g_events.push_back("create");
    *h = nullptr;
  }
  void destroy(FakeHandle) {}
};

// An ordinal no test box has. wwrSetDevice rejects it up front (returns an
// error, does not abort or allocate), which is what makes the select visible.
constexpr int kInvalidDevice = 999;

} // namespace

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DeviceBoundHandle Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// The core regression guard: select is sequenced before create. If a refactor
// ever moved the wwrSetDevice out of the base-initializer slot (into the ctor
// body, or the derived), create would run first and this order would flip.
TEST(DeviceBoundHandleTests, SelectIsSequencedBeforeCreate) {
  g_events.clear();
  DeviceSpyWrapper handle(kInvalidDevice);
  ASSERT_EQ(g_events.size(), 2u);
  EXPECT_EQ(g_events[0], "select");
  EXPECT_EQ(g_events[1], "create");
  static_cast<void>(wwrGetLastError()); // clear the sticky failure for later tests
}

// The by-reference-then-move plumbing: on_device takes the ctor's policy_device
// *parameter* by reference (policy_device_ is not alive yet), records the select
// error into it, and the ctor then moves that same object into policy_device_.
// Reading it back proves nothing was dropped in the hand-off.
TEST(DeviceBoundHandleTests, SelectErrorIsThreadedIntoPolicyDevice) {
  g_events.clear();
  DeviceSpyWrapper handle(kInvalidDevice);
  EXPECT_EQ(handle.device_policy().calls, 1);
  EXPECT_NE(handle.device_policy().last_error, wwrSuccess);
  static_cast<void>(wwrGetLastError());
}

// Documents-as-test the limit of dev_idx(): record_device reads the *current*
// device (wwrGetDevice) after construction. The select to device 999 failed, so
// the current device is still 0 -- dev_idx() reports 0, NOT the requested 999.
// dev_idx() is a witness to thread state, not to where the handle truly lives;
// this pins that behaviour so no one "fixes" it to echo the request.
TEST(DeviceBoundHandleTests, DevIdxRecordsCurrentDeviceNotRequested) {
  g_events.clear();
  DeviceSpyWrapper handle(kInvalidDevice);
  EXPECT_EQ(handle.dev_idx(), 0);
  static_cast<void>(wwrGetLastError());
}

// The happy path: a valid device selects silently (no policy call), create runs,
// and record_device captures device 0.
TEST(DeviceBoundHandleTests, ValidDeviceSelectsSilentlyRecordsAndCreates) {
  g_events.clear();
  DeviceSpyWrapper handle(0);
  EXPECT_EQ(handle.device_policy().calls, 0); // success path never touches the policy
  EXPECT_EQ(handle.dev_idx(), 0);
  ASSERT_EQ(g_events.size(), 1u);
  EXPECT_EQ(g_events[0], "create");
}

} // namespace wwr::extension::test
