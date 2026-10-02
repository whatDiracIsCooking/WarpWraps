// Trivial link check for wwr::thrust (issue #235, F1): that the one target
// resolves Thrust's include dirs on both backends -- CCCL on CUDA, rocThrust
// on HIP -- and the header the acceptance names compiles under each front end.
// Building it is the assertion; no ctest entry, no launch.
#include <thrust/version.h>

static_assert(THRUST_VERSION > 0, "thrust/version.h did not define a version");
