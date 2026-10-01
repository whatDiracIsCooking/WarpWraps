// conversions.cpp - host runtime tests for the fp16 / bf16 host conversion
// wrappers (wwr.fp16, wwr.bf16).
//
// These wrappers run on the CPU -- unlike the __device__ conversions in
// fp16.h / bf16.h, which fp16.cu / bf16.cu prove by compiling -- so they can
// be exercised for real with no GPU. The compile-time fp16.cppm / bf16.cppm
// module tests prove the wrappers are reachable and link (WWR_LINK_CHECK); this
// proves they compute.
//
// Every value here is chosen to make == exact, with no vendor-library
// reproducibility assumed: 1.5 is representable in both formats, and 1 + 2^-9 is
// representable in half (mantissa step 2^-10 at 1.0) but not in bfloat16 (step
// 2^-7). That last pair is the discriminating case -- a half<->bfloat16
// intrinsic swap would make half lose the value or bfloat16 keep it, so both
// KeepsHalfPrecision and NarrowsBelowBfloat16Precision would flip.
#include <gtest/gtest.h>

import wwr.fp16;
import wwr.bf16;

namespace {

using wwr::wwrBfloat162Float;
using wwr::wwrFloat2Bfloat16;
using wwr::wwrFloat2Half;
using wwr::wwrHalf2Float;

// Representable in both formats -> the round trip is lossless either way.
constexpr float kExact = 1.5f;

// Two half-steps above 1.0: representable in half, below bfloat16's precision.
constexpr float kHalfOnly = 1.0f + 0x1p-9f; // 1.001953125

TEST(HalfConversion, RoundTripExact) {
  EXPECT_EQ(wwrHalf2Float(wwrFloat2Half(kExact)), kExact);
}

TEST(HalfConversion, KeepsHalfPrecision) {
  EXPECT_EQ(wwrHalf2Float(wwrFloat2Half(kHalfOnly)), kHalfOnly);
}

TEST(Bfloat16Conversion, RoundTripExact) {
  EXPECT_EQ(wwrBfloat162Float(wwrFloat2Bfloat16(kExact)), kExact);
}

TEST(Bfloat16Conversion, NarrowsBelowBfloat16Precision) {
  EXPECT_EQ(wwrBfloat162Float(wwrFloat2Bfloat16(kHalfOnly)), 1.0f);
}

} // namespace
