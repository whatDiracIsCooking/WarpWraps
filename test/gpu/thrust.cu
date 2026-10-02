// Compile-time acceptance for the wwr::thrust re-export layer (step 2). Building
// this device TU under each backend IS the assertion; no ctest entry, no launch.
//
// Two things are proven by the mere act of compiling it:
//   1. Every re-exported name exists on the selected backend -- each
//      `using ::thrust::<name>` in the four family headers is a hard compile
//      error if CCCL (CUDA) or rocThrust (HIP) lacks that name, so pulling all
//      four headers checks the whole surface on whichever front end builds.
//   2. wwr::par_on threads through the policy-first overload -- one
//      representative call per family instantiates the real algorithm with the
//      backend's stream-bound execution policy.
#include <thrust/version.h>

#include "thrust/reduce.cuh"
#include "thrust/reorder.cuh"
#include "thrust/scan.cuh"
#include "thrust/transform.cuh"

static_assert(THRUST_VERSION > 0, "thrust/version.h did not define a version");

// Host-side launchers in a device TU: each thrust call dispatches device work on
// the stream wwr::par_on binds. Raw device pointers act as the iterators.
void wwr_thrust_accept(const wwr::wwrStream_t stream, float *in, float *out,
                       std::size_t n) {
  const auto policy = wwr::par_on(stream);
  wwr::thrust::sort(policy, in, in + n);                 // reorder
  out[0] = wwr::thrust::reduce(policy, in, in + n);      // reduce
  wwr::thrust::inclusive_scan(policy, in, in + n, out);  // scan
  wwr::thrust::fill(policy, out, out + n, 0.0f);         // transform
}
