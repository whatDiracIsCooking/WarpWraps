/**
 * @file reorder.cuh
 * @brief Backend-neutral wwr::thrust re-export of the reorder family
 *
 * sort, unique, partition, remove, copy and reverse -- the algorithms that
 * rearrange or compact a range. Plain `using`-re-exports into namespace
 * wwr::thrust: the names are identical across CUDA (CCCL) and HIP (rocThrust),
 * so every overload comes across and there is nothing to keep in sync. The
 * leading `::` on each is load-bearing -- inside namespace wwr::thrust a bare
 * `thrust` names THIS namespace, not the global one.
 *
 * The one divergent spelling, the stream-bound execution policy, is not a name
 * and so not re-exported here; callers pass wwr::par_on(stream)
 * (execution_policy.cuh) as the leading argument:
 *   wwr::thrust::sort(wwr::par_on(stream), first, last);
 *
 * Device-includable header, not a module: the Thrust headers assume a device
 * pass and a device TU cannot `import` (docs/architecture.md §8). Link
 * wwr::thrust.
 */

#pragma once

#include "execution_policy.cuh"

#include <thrust/copy.h>
#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/reverse.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace wwr::thrust {

// sort
using ::thrust::is_sorted;
using ::thrust::is_sorted_until;
using ::thrust::sort;
using ::thrust::sort_by_key;
using ::thrust::stable_sort;
using ::thrust::stable_sort_by_key;

// unique
using ::thrust::unique;
using ::thrust::unique_by_key;
using ::thrust::unique_by_key_copy;
using ::thrust::unique_copy;
using ::thrust::unique_count;

// partition
using ::thrust::is_partitioned;
using ::thrust::partition;
using ::thrust::partition_copy;
using ::thrust::partition_point;
using ::thrust::stable_partition;
using ::thrust::stable_partition_copy;

// remove
using ::thrust::remove;
using ::thrust::remove_copy;
using ::thrust::remove_copy_if;
using ::thrust::remove_if;

// copy
using ::thrust::copy;
using ::thrust::copy_if;
using ::thrust::copy_n;

// reverse
using ::thrust::reverse;
using ::thrust::reverse_copy;

} // namespace wwr::thrust
