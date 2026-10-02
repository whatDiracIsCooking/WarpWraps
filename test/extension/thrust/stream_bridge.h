// stream_bridge.h
//
// Host-callable entry points into the device TU stream_ops.cu, declared so the
// host gtest (stream_tests.cpp) can drive them. The .cu is device-compiled and
// cannot be imported; it #includes the wwr* layer's execution_policy.cuh and
// calls thrust::sort(wwr::par_on(stream), ...) -- the acceptance that the F2
// shim selects the right backend policy from a device TU. These declarations
// are the host side of that boundary, exactly as init_state_bridge.h is for
// src/extension/init_state.
//
// wwrStream_t is left to the includer, which already has it: the .cu through
// execution_policy.cuh's runtime.h, the host gtest through `import
// wwr.runtime_api`. Pulling runtime.h in here would force the src/ root onto
// the host TU and collide its `using` with the module's exported wwrStream_t.
#pragma once

#include <cstddef>

namespace wwr::test {

// Sorts d_data[0, n) ascending via thrust::sort(wwr::par_on(stream), ...), so
// the sort's work is enqueued on `stream`. d_data is device memory. The literal
// acceptance call: it proves the shim's policy is a usable execution policy that
// the sort runs correctly through.
void thrust_sort_on_stream(wwrStream_t stream, int *d_data, std::size_t n);

// Returns the stream that wwr::par_on(stream) binds its policy to -- Thrust's
// own get_stream() read back out of the policy the shim built. The direct,
// deterministic proof that par_on routes work to exactly the stream handed in:
// the caller asserts the returned handle equals `stream`.
wwrStream_t policy_bound_stream(wwrStream_t stream);

} // namespace wwr::test
