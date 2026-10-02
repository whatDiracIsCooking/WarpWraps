// stream_ops.cu
//
// The device half of the F2 stream test. It is the real consumer the acceptance
// names: a device TU that #includes the wwr* execution-policy shim and calls
// thrust::sort(wwr::par_on(stream), ...), letting the backend policy
// (thrust::cuda::par on CUDA, thrust::hip::par on HIP) be selected from the
// compiler's own device-pass macro. Shared unchanged between both backends --
// .cu means "device pass, whichever backend", as elsewhere in the tree.
//
// execution_policy.cuh first: it pulls runtime.h, which defines wwrStream_t for
// the bridge declarations that follow.
#include "wrappers/thrust/execution_policy.cuh"

#include "stream_bridge.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cstddef>

namespace wwr::test {

void thrust_sort_on_stream(const wwrStream_t stream, int *d_data, const std::size_t n) {
  // wwr::par_on(stream) is the shim under test; thrust::device_pointer_cast
  // wraps the raw device pointer so Thrust treats it as device memory.
  const auto first = ::thrust::device_pointer_cast(d_data);
  ::thrust::sort(wwr::par_on(stream), first, first + n);
}

wwrStream_t policy_bound_stream(const wwrStream_t stream) {
  // get_stream() is the stream accessor both backends declare as a hidden-friend
  // of the execution policy's base; ADL finds it from the policy type, so the
  // one spelling works for thrust::cuda::par and thrust::hip::par alike. What it
  // returns is the handle par_on bound the policy to -- it must be `stream`.
  return get_stream(wwr::par_on(stream));
}

} // namespace wwr::test
