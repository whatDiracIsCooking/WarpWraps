// Aggregation header for harvesting wwr.hip.hiptensor's surface.
// Mirrors src/hip/hiptensor.cppm's global module fragment one-for-one so the
// manifest is the surface that module wraps at BOTH ends of the version range:
//
//   * <array> is pre-included as a load-bearing workaround (host_defines.h
//     poisons __has_attribute(__noinline__); pulling <array> first sidesteps it
//     via the include guard). docs/architecture.md, section 9.
//   * __has_include picks the header: 2.2.0 (ROCm 7.2, the pin) added the
//     C-linkage hiptensor.h, which pulls its own version header; 2.1.0 (ROCm
//     7.1, the floor) ships only hiptensor.hpp, which does NOT -- so
//     hiptensor-version.hpp is added beside it, without which hiptensorGetVersion
//     is the single name left undeclared at the floor.
//
// Keep in lockstep with hiptensor.cppm's GMF. See this dir's README.
#include <array>
#if __has_include(<hiptensor/hiptensor.h>)
#include <hiptensor/hiptensor.h>
#else
#include <hiptensor/hiptensor-version.hpp>
#include <hiptensor/hiptensor.hpp>
#endif
