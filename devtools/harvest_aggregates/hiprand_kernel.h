// Aggregation header for harvesting wwr.hip.hiprand_kernel's surface.
// hiprand_kernel.h calls bare printf without including <cstdio>; a plain
// (non -x hip) compile needs it pre-included, as src/hip/hiprand_kernel.cppm
// does. See this dir's README.
#include <cstdio>
#include <hiprand/hiprand_kernel.h>
