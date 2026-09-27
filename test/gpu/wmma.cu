// Compile-time test for src/wmma.cuh, the include switch plus one
// namespace alias. The header defines a single name, wwr::gpuwmma, and has no
// in-tree caller yet, so without this TU nothing compiles it and a break on one
// backend would ship unseen -- <mma.h> and <rocwmma/rocwmma.hpp> are separate
// implementations of a shared spelling, not one header behind two paths, which
// is a weaker guarantee than cooperative_groups.cuh's.
//
// Building this .cu under the selected backend's device pass IS the test: every
// gpuwmma entity named below has to resolve on both backends. The kernels are
// never launched -- WMMA is arch-gated (sm_70+, and a gfx11/gfx12/CDNA part on
// the AMD side) and this tier configures with no device. Reached through
// wwr.device, exactly as a real device consumer reaches the header.
//
// The static_asserts turn the silent divergences the header documents into
// compile-time tripwires. Backend is selected on WWR_SELECTED_CUDA -- the
// device-pass macro the header itself switches on, available here for free and
// consistent across both of HIP's compile passes (unlike the
// WWR_GPU_BACKEND_* define the host-compiled .cppm tests use).
#include "bf16.cuh"
#include "fp16.cuh"
#include "wmma.cuh"

namespace {

namespace w = wwr::gpuwmma;

// Local rather than <type_traits>: a device TU here imports no modules and
// pulls no standard library it does not need, and this is two lines.
template<typename A, typename B>
struct same_type {
  static constexpr bool value = false;
};
template<typename A>
struct same_type<A, A> {
  static constexpr bool value = true;
};

// The one portable tile. 16x16x16 is the whole intersection: CUDA's 32x8x16 and
// 8x32x16 have no rocWMMA counterpart on RDNA, where the shape fails to
// instantiate rather than falling back.
using FragA = w::fragment<w::matrix_a, 16, 16, 16, wwr::gpuHalf, w::row_major>;
using FragB = w::fragment<w::matrix_b, 16, 16, 16, wwr::gpuHalf, w::col_major>;
using FragC = w::fragment<w::accumulator, 16, 16, 16, float>;

// gpuHalf is the portable element type: __half on both backends, and rocWMMA
// spells its own hfloat16_t the same way. Half of the element-type
// asymmetry -- the bf16 half is pinned in the second kernel below.
static_assert(same_type<FragC::element_type, float>::value,
              "accumulator element_type is expected to be float on both backends");

// Silent divergence: a fragment's num_elements is NOT portable. It is
// the per-lane share of the tile, so it falls out of the warp width -- CUDA's 32
// lanes give matrix_a 16 and the accumulator 8, and an AMD wavefront gives
// something else again. Worse, under HIP it differs between the two compile
// passes of this one TU (wave64 in the host pass, wave32 for gfx1200), so no
// single number can be asserted there at all and host code sizing a buffer from
// it is wrong in a way nothing diagnoses.
//
// So the CUDA branch pins the literal counts -- which are exactly what a caller
// is tempted to hard-code -- and the HIP branch pins only what is true in both
// of its passes. WWR_WARP_SIZE is no help there: it is the configure-time 32
// in both passes, while the host pass lays the fragment out for 64, so
// multiplying by it fails the host pass outright. __AMDGCN_WAVEFRONT_SIZE__ does
// give the width this pass used, but clang 20 deprecates it ("compile-time-
// constant access to the wavefront size will be removed in a future release"),
// which is the whole argument for a configure-time warp size arriving from the
// vendor -- so neither spelling of the warp width belongs in an assertion here.
#if defined(WWR_SELECTED_CUDA)
static_assert(FragA::num_elements == 16,
              "CUDA matrix_a 16x16x16 is expected to be 16 elements per lane");
static_assert(FragC::num_elements == 8,
              "CUDA accumulator 16x16x16 is expected to be 8 elements per lane");
#else
// Both hold in the wave64 host pass (4 and 4) and the gfx1200 device pass (8 and
// 8), and both are false on CUDA -- which is the divergence, stated without
// naming a width.
static_assert(FragA::num_elements == FragC::num_elements,
              "rocWMMA matrix_a and accumulator are expected to hold the same "
              "per-lane share of a 16x16 tile, unlike CUDA's 16 and 8");
static_assert(FragA::num_elements != 16,
              "rocWMMA matrix_a is not expected to match CUDA's 16 elements per "
              "lane; num_elements is not portable");
#endif

} // namespace

// The portable half-in, float-out tile: fill, load both operands, accumulate,
// store. Every entry point of the shared surface, in the order a real kernel
// calls them, plus the layout_t enumerator store_matrix_sync takes.
__global__ void wwr_wmma_tile(const wwr::gpuHalf *a, const wwr::gpuHalf *b, float *c) {
  FragA fa;
  FragB fb;
  FragC acc;

  w::fill_fragment(acc, 0.0f);
  w::load_matrix_sync(fa, a, 16);
  w::load_matrix_sync(fb, b, 16);
  w::mma_sync(acc, fa, fb, acc);

  // frag.x[i] is the portable element access, and the reason it is worth
  // naming: CUDA's x is a plain array member, rocWMMA's is a vector member of
  // an anonymous union it carries for exactly this compatibility. rocWMMA also
  // offers operator[], which CUDA has not -- so subscripting the fragment
  // itself compiles on one backend only.
  acc.x[0] = acc.x[0] + 1.0f;

  w::store_matrix_sync(c, acc, 16, w::mem_row_major);
}

// bfloat16 is where the element type stops being portable, and it is a silent
// trap rather than a missing feature: both backends do 16x16x16 bf16, but
// wwr::gpuBfloat16 is __nv_bfloat16 on CUDA and __hip_bfloat16 on HIP, and
// rocWMMA knows only the older hip_bfloat16 that its own bfloat16_t names. So
// the type bf16.cuh hands a kernel works on one backend and fails to instantiate
// PackTraits on the other, which is why this kernel needs the #if that the tile
// above does not.
#if defined(WWR_SELECTED_CUDA)
using WmmaBf16 = wwr::gpuBfloat16;
static_assert(same_type<WmmaBf16, wwr::gpuBfloat16>::value,
              "on CUDA the wmma bf16 element type IS gpuBfloat16");
#else
using WmmaBf16 = w::bfloat16_t;
static_assert(!same_type<WmmaBf16, wwr::gpuBfloat16>::value,
              "rocWMMA's bfloat16_t is still expected to differ from gpuBfloat16; "
              "if ROCm has unified them, wmma.cuh needs updating");
#endif

__global__ void wwr_wmma_bf16_fragments(const WmmaBf16 *a) {
  // Declaring them is the claim: instantiating the fragment is what fails on
  // an element type the backend's WMMA does not know.
  w::fragment<w::matrix_a, 16, 16, 16, WmmaBf16, w::row_major> fa;
  w::fragment<w::matrix_b, 16, 16, 16, WmmaBf16, w::col_major> fb;
  w::fragment<w::accumulator, 16, 16, 16, float> acc;

  w::fill_fragment(acc, 0.0f);
  w::load_matrix_sync(fa, a, 16);
  (void)fb;
  (void)acc;
}
