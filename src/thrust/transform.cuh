/**
 * @file transform.cuh
 * @brief Backend-neutral wwr::thrust re-export of the transform family
 *
 * The elementwise maps: transform, for_each, fill, generate, sequence,
 * tabulate and replace. Plain `using`-re-exports into namespace wwr::thrust;
 * see reorder.cuh for the pattern, the load-bearing leading `::`, and why the
 * execution policy stays behind wwr::par_on rather than being re-exported. Link
 * wwr::thrust.
 *
 * The functor-taking members (for_each, generate, tabulate, the free-functor
 * transform) need no special handling under this `using` layer: the caller's
 * own functor and types instantiate the template in the caller's device TU,
 * exactly as they would calling thrust:: directly.
 */

#pragma once

#include "execution_policy.cuh"

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/generate.h>
#include <thrust/replace.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

namespace wwr::thrust {

using ::thrust::transform;
using ::thrust::transform_if;

using ::thrust::for_each;
using ::thrust::for_each_n;

using ::thrust::fill;
using ::thrust::fill_n;

using ::thrust::generate;
using ::thrust::generate_n;

using ::thrust::sequence;

using ::thrust::tabulate;

using ::thrust::replace;
using ::thrust::replace_copy;
using ::thrust::replace_copy_if;
using ::thrust::replace_if;

} // namespace wwr::thrust
