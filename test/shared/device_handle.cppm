/**
 * @file device_handle.cppm
 * @brief Test-side device handle: the shipped DeviceHandle, fixed to this suite's
 *        AbortPolicy
 *
 * A DeviceBuffer needs some type satisfying the device_handle ladder to back it.
 * The shipped library carries DeviceHandle again (wwr.extension.runtime) -- #93
 * had moved it here only to keep a forced AbortPolicy out of the library, and now
 * that it is policy-templated it bakes none, so it went home. This is just that
 * type bound to the extension suites' AbortPolicy -- the fullest tier (dev_idx() +
 * stream() + pool()), so a buffer built on it draws from that pool on that stream.
 * The abort-on-failure policy choice stays on the test side, where it belongs.
 *
 * Usage:
 *   import wwr.test.shared.device_handle;
 *   using namespace wwr::extension::test;
 */

export module wwr.test.shared.device_handle;

export import wwr.extension.runtime;        // DeviceHandle (and its stream/pool types)
export import wwr.test.shared.abort_policy; // AbortPolicy, baked in below
import wwr.extension.handle;                // device_handle_* concepts (static_asserts)
import wwr.runtime_api;                      // wwrError_t

export namespace wwr::extension::test {

/// The reference model the extension suites allocate against: the shipped
/// kit::DeviceHandle at the fullest tier, bound to the kit's AbortPolicy (aliased
/// into the test namespace by wwr.test.shared.abort_policy).
using DeviceHandle = ::wwr::extension::kit::DeviceHandle<AbortPolicy<wwrError_t>,
                                                         AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>>;

/// Pinning the fullest rung also pins the two it refines -- and guards that a
/// StreamWrapper still answers the stream tier (the handle buffer_suite/warp_reduce
/// use directly).
static_assert(device_handle_pool<DeviceHandle>);
static_assert(device_handle_stream<
              StreamWrapper<AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>, AbortPolicy<wwrError_t>>>);

} // namespace wwr::extension::test
