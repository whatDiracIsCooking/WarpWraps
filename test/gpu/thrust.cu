// Compile-time acceptance for the wwr::thrust re-export layer. Building this
// device TU under each backend IS the assertion; no ctest entry, no launch.
//
// Two things are proven by the mere act of compiling it:
//   1. Every re-exported name exists on the selected backend -- each
//      `using ::thrust::<name>` in the leaf headers is a hard compile error if
//      CCCL (CUDA) or rocThrust (HIP) lacks that name, so pulling every leaf
//      checks the whole surface on whichever front end builds.
//   2. wwr::par_on threads through the policy-first overload -- one
//      representative call per Thrust family instantiates the real algorithm
//      with the backend's stream-bound execution policy.
#include <thrust/version.h>

// Every leaf, so every re-exported name is checked on the selected backend.
#include "thrust/copy.cuh"
#include "thrust/count.cuh"
#include "thrust/extrema.cuh"
#include "thrust/fill.cuh"
#include "thrust/for_each.cuh"
#include "thrust/generate.cuh"
#include "thrust/inner_product.cuh"
#include "thrust/partition.cuh"
#include "thrust/reduce.cuh"
#include "thrust/remove.cuh"
#include "thrust/replace.cuh"
#include "thrust/reverse.cuh"
#include "thrust/scan.cuh"
#include "thrust/sequence.cuh"
#include "thrust/sort.cuh"
#include "thrust/tabulate.cuh"
#include "thrust/transform.cuh"
#include "thrust/transform_reduce.cuh"
#include "thrust/transform_scan.cuh"
#include "thrust/unique.cuh"

static_assert(THRUST_VERSION > 0, "thrust/version.h did not define a version");

// Host-side launchers in a device TU: each thrust call dispatches device work on
// the stream wwr::par_on binds. Raw device pointers act as the iterators.
void wwr_thrust_accept(const wwr::wwrStream_t stream, float *in, float *out,
                       std::size_t n) {
  const auto policy = wwr::par_on(stream);
  wwr::thrust::sort(policy, in, in + n);                 // reorder (sort.cuh)
  out[0] = wwr::thrust::reduce(policy, in, in + n);      // reduce (reduce.cuh)
  wwr::thrust::inclusive_scan(policy, in, in + n, out);  // scan (scan.cuh)
  wwr::thrust::fill(policy, out, out + n, 0.0f);         // transform (fill.cuh)
}
