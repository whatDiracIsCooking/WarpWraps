// link_check.cpp - build-and-link proof for wwr.extension.tensor (NO_RUN)
//
// cuTENSOR / hipTENSOR abort in a library global constructor on a host with no
// supported GPU (libhiptensor's ctor calls the HIP runtime -- "no ROCm-capable
// device is detected" at hiptensor/.../hip_device.cpp). So NO binary linking
// wwr.extension.tensor can be RUN -- or even `--gtest_list_tests`'d, which is why
// a GoogleTest suite's SuiteListIsComplete guard cannot be used here -- on the
// GPU-less CI runner: the abort fires before main(). The module is therefore
// verified exactly like the vendor module (test/gpu/tensor_link_check.cpp) and
// test/hip's hiptensor NO_RUN executable: this TU is BUILT so every wwrtensor*
// symbol the wrappers call resolves at link time -- the successful link IS the
// assertion -- and the move/copy contract is pinned by the static_asserts below,
// which the compiler proves with no device. It carries NO ctest entry (NO_RUN),
// so it is never launched on either backend; cuTENSOR does not abort this way,
// but one uniform code path is simpler. The destroy-exactly-once RUNTIME
// contract is BaseHandle's, exercised on a device by the blas / fft suites;
// tensor adds only the create/destroy/descriptor bindings a link proves present.
//
// Backend-neutral -- built for either WWR_GPU_BACKEND.

import std;
import wwr.extension.common; // kit::AbortPolicy, the error accessors
import wwr.extension.handle; // BaseHandle (move/copy semantics under test)
import wwr.extension.tensor; // the wrappers; re-exports wwr.tensor's types/enums

namespace wwr::extension::test {
namespace {

using Policy = kit::AbortPolicy<wwrtensorStatus_t>;
using Handle = TensorHandleWrapper<Policy, Policy>;
using Desc = TensorDescriptorWrapper<Policy, Policy>;

// The RAII contract, proved at compile time (no device needed): copy is deleted
// (a copyable handle would double-free), move is noexcept, and a descriptor is
// built from a handle + extent, never default-constructed.
static_assert(!std::is_copy_constructible_v<Handle>);
static_assert(!std::is_copy_assignable_v<Handle>);
static_assert(std::is_nothrow_move_constructible_v<Handle>);
static_assert(std::is_nothrow_move_assignable_v<Handle>);
static_assert(!std::is_copy_constructible_v<Desc>);
static_assert(std::is_nothrow_move_constructible_v<Desc>);
static_assert(!std::is_default_constructible_v<Desc>);

// Never executed (see file header). Its body odr-uses the wrapper constructors
// and the error accessor so wwrtensorCreate / wwrtensorCreateTensorDescriptor,
// their Destroy pair, and wwrtensorGetErrorString must all resolve at link.
void force_link() {
  Handle handle;
  const std::array<std::int64_t, 2> extent{64, 64};
  Desc desc{handle, extent, WWRTENSOR_R_32F};
  (void)error_string<wwrtensorStatus_t>(WWRTENSOR_STATUS_NOT_SUPPORTED);
}

} // namespace
} // namespace wwr::extension::test

// Taking force_link's address through a volatile -- so the optimizer cannot elide
// it -- keeps the function, and therefore its wwrtensor* calls, in the link
// without ever calling it. A missing symbol fails the build, which is the whole
// assertion.
int main() {
  void (*volatile keep)() = &wwr::extension::test::force_link;
  (void)keep;
}
