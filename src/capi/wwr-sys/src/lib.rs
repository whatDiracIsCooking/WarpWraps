//! Raw `-sys` bindings over the `wwr` C ABI (the `libwwr_capi_*.so` family).
//!
//! The crate is SPLIT PER SURFACE, mirroring the C-side per-unit split: one
//! always-compiled `common` module (the PODs more than one surface uses --
//! `wwr_complex64` / `wwr_complex128`) plus one module per surface behind a
//! Cargo feature:
//!
//! | feature       | module                | links (via `build.rs`)         |
//! |---------------|-----------------------|--------------------------------|
//! | `complex`     | [`complex`]           | `libwwr_capi_complex.so` (no vendor math lib) |
//! | `runtime_api` | [`runtime_api`]       | `libwwr_capi_runtime_api.so`   |
//! | `blas`        | [`blas`]              | `libwwr_capi_blas.so` -> cuBLAS / hipBLAS |
//! | `blaslt`      | [`blaslt`]            | `libwwr_capi_blaslt.so` -> cuBLASLt / hipBLASLt |
//! | `solver`      | [`solver`]            | `libwwr_capi_solver.so` -> cuSOLVER / hipSOLVER |
//! | `rand`        | [`rand`]              | `libwwr_capi_rand.so` -> cuRAND / hipRAND |
//! | `rtc`         | [`rtc`]               | `libwwr_capi_rtc.so` -> nvRTC / hipRTC |
//! | `sparse`      | [`sparse`]            | `libwwr_capi_sparse.so` -> cuSPARSE / hipSPARSE |
//! | `tensor`      | [`tensor`]            | `libwwr_capi_tensor.so` -> cuTENSOR / hipTENSOR |
//! | `fft`         | [`fft`]               | `libwwr_capi_fft.so` -> cuFFT / hipFFT |
//! | `comp`        | [`comp`]              | `libwwr_capi_comp.so` -> nvCOMP / hipCOMP |
//! | `fp16`        | [`fp16`]              | `libwwr_capi_fp16.so` (no vendor math lib; half type) |
//! | `bf16`        | [`bf16`]              | `libwwr_capi_bf16.so` (no vendor math lib; bfloat16 type) |
//! | `vector_types`| [`vector_types`]      | `libwwr_capi_vector_types.so` (no vendor math lib; vector types) |
//! | `fp8`         | [`fp8`]               | `libwwr_capi_fp8.so` (no vendor math lib; fp8 type) |
//! | `tx`          | [`tx`]                | `libwwr_capi_tx.so` -> NVTX (header-only) / rocTX |
//! | `ccl`         | [`ccl`]               | `libwwr_capi_ccl.so` -> NCCL / RCCL |
//! | `extension_random_normal` | [`extension_random_normal`] | `libwwr_capi_extension_random_normal.so` -> cuRAND / hipRAND + runtime (DEVICE kernels) |
//!
//! ALL are default features, so an existing user who wants the whole ABI
//! is unchanged -- they just transparently link the per-unit `.so`s instead of
//! one umbrella. A user who needs one surface opts DOWN:
//! `wwr-sys = { ..., default-features = false, features = ["complex"] }`
//! -- and then NOTHING of another surface's vendor dependency is linked (a
//! `complex`-only build NEEDs no cuBLAS -- the Rust-side isolation this split
//! buys, mirroring the per-unit `.so`s on the C side).
//!
//! Each module is the GENERATED `src/<unit>.rs` (and `common.rs`): `generate.py`
//! writes them from the same spec it writes the C headers and thunks from, so the
//! crate cannot drift from the ABI. Do not edit them by hand; regenerate with
//! `python3 src/capi/generate.py`.
//!
//! The backend (CUDA or HIP) is chosen only by which `wwr` library the linked
//! `libwwr_capi_*.so` was built against -- this Rust is identical for either, the
//! whole point of the C ABI. `build.rs` finds those `.so`s and links only the
//! ENABLED features' libraries; see `src/capi/README.md`.
//!
//! This is the raw surface: every function is `unsafe extern "C"`. A safe
//! `Result`-returning wrapper over `wwr_status` is deliberately out of scope
//! here -- the `-sys` crate is the binding, nothing more.

#![allow(non_camel_case_types, non_snake_case)]

// The shared PODs any surface may name -- always compiled, not feature-gated.
pub mod common;
pub use common::*;

#[cfg(feature = "complex")]
pub mod complex;
#[cfg(feature = "complex")]
pub use complex::*;

#[cfg(feature = "runtime_api")]
pub mod runtime_api;
#[cfg(feature = "runtime_api")]
pub use runtime_api::*;

#[cfg(feature = "blas")]
pub mod blas;
#[cfg(feature = "blas")]
pub use blas::*;

#[cfg(feature = "blaslt")]
pub mod blaslt;
#[cfg(feature = "blaslt")]
pub use blaslt::*;

#[cfg(feature = "solver")]
pub mod solver;
#[cfg(feature = "solver")]
pub use solver::*;

#[cfg(feature = "rand")]
pub mod rand;
#[cfg(feature = "rand")]
pub use rand::*;

#[cfg(feature = "rtc")]
pub mod rtc;
#[cfg(feature = "rtc")]
pub use rtc::*;

#[cfg(feature = "sparse")]
pub mod sparse;
#[cfg(feature = "sparse")]
pub use sparse::*;

#[cfg(feature = "tensor")]
pub mod tensor;
#[cfg(feature = "tensor")]
pub use tensor::*;

#[cfg(feature = "fft")]
pub mod fft;
#[cfg(feature = "fft")]
pub use fft::*;

#[cfg(feature = "comp")]
pub mod comp;
#[cfg(feature = "comp")]
pub use comp::*;

#[cfg(feature = "fp16")]
pub mod fp16;
#[cfg(feature = "fp16")]
pub use fp16::*;

#[cfg(feature = "bf16")]
pub mod bf16;
#[cfg(feature = "bf16")]
pub use bf16::*;

#[cfg(feature = "vector_types")]
pub mod vector_types;
#[cfg(feature = "vector_types")]
pub use vector_types::*;

#[cfg(feature = "fp8")]
pub mod fp8;
#[cfg(feature = "fp8")]
pub use fp8::*;

#[cfg(feature = "tx")]
pub mod tx;
#[cfg(feature = "tx")]
pub use tx::*;

#[cfg(feature = "ccl")]
pub mod ccl;
#[cfg(feature = "ccl")]
pub use ccl::*;

#[cfg(feature = "extension_random_normal")]
pub mod extension_random_normal;
#[cfg(feature = "extension_random_normal")]
pub use extension_random_normal::*;

#[cfg(test)]
mod tests {
    // A smoke test that LINKS the per-unit .so(s) and CALLS through the FFI:
    // proves build.rs found the libs, the #[repr(C)] by-value layout matches the
    // C ABI, and the values come back right. Host-only, both backends -- mirrors
    // driver.rs. Run with WWR_CAPI_LIB_DIR pointing at the built/installed .so's.

    // The complex surface needs no device and no vendor math lib.
    #[cfg(feature = "complex")]
    #[test]
    fn complex64_arithmetic() {
        use crate::{wwr_cabs64, wwr_cadd64, wwr_complex64};
        // SAFETY: a by-value call into the generated extern "C" thunk over the
        // host-callable wwr functions; the PODs are #[repr(C)]. Sound.
        unsafe {
            let a = wwr_complex64 { re: 3.0, im: 4.0 };
            let b = wwr_complex64 { re: 1.0, im: 2.0 };
            let s = wwr_cadd64(a, b);
            assert!((s.re - 4.0).abs() < 1e-4 && (s.im - 6.0).abs() < 1e-4);
            let m = wwr_cabs64(a);
            assert!((m - 5.0).abs() < 1e-4);
        }
    }

    // The status surface: wwr_runtime_get_version needs no device, so it is
    // deterministic on a no-GPU runner and returns the stable WWR_STATUS_SUCCESS
    // (== 0) with a positive version whichever backend the .so links.
    #[cfg(feature = "runtime_api")]
    #[test]
    fn runtime_version_status() {
        use crate::{wwr_runtime_get_version, wwr_status};
        let mut ver: i32 = 0;
        // SAFETY: &mut ver is a valid out-param address for the extern "C" call.
        let st = unsafe { wwr_runtime_get_version(&mut ver) };
        assert_eq!(st, wwr_status::WWR_STATUS_SUCCESS);
        assert!(ver > 0);
    }

    // The fp16 surface: a host float->half->float round trip. Needs no device
    // and no vendor math lib -- the conversions are host-callable. Proves the
    // per-unit .so links, the shared wwr_half POD layout matches, and 1.5 (exact
    // in half) survives the trip.
    #[cfg(feature = "fp16")]
    #[test]
    fn fp16_round_trip() {
        use crate::{wwr_float2half, wwr_half2float};
        // SAFETY: by-value calls into the generated extern "C" thunks; wwr_half
        // is #[repr(C)]. Host-only.
        unsafe {
            let h = wwr_float2half(1.5);
            let back = wwr_half2float(h);
            assert!((back - 1.5).abs() < 1e-6);
        }
    }

    // The vector_types surface: a host make_float4 construction. Proves the
    // per-unit .so links and the #[repr(C)] POD layout matches the vendor
    // aggregate the thunk builds. Host-only.
    #[cfg(feature = "vector_types")]
    #[test]
    fn vector_types_make_float4() {
        use crate::wwr_make_float4;
        // SAFETY: a by-value call into the generated extern "C" thunk; the POD is
        // #[repr(C)]. Host-only.
        let v = unsafe { wwr_make_float4(1.0, 2.0, 3.0, 4.0) };
        assert_eq!((v.x, v.y, v.z, v.w), (1.0, 2.0, 3.0, 4.0));
    }

    // The fp8 surface: a host narrowing conversion returning the storage byte.
    // Proves the per-unit .so links, the value-enum map crosses the boundary, and
    // the storage POD comes back. 1.0 in E4M3 is a known bit pattern (0x38), but
    // we only assert the call links and returns (the exact byte is a vendor
    // detail); host-only, no device.
    #[cfg(feature = "fp8")]
    #[test]
    fn fp8_narrow_links() {
        use crate::{
            wwr_float2fp8, wwr_fp8_interpretation::WWR_FP8_E4M3,
            wwr_fp8_saturation::WWR_FP8_SATFINITE,
        };
        // SAFETY: a by-value call into the generated extern "C" thunk; the enums
        // are #[repr(i32)] and the POD #[repr(C)]. Host-only.
        let s = unsafe { wwr_float2fp8(1.0, WWR_FP8_SATFINITE, WWR_FP8_E4M3) };
        assert_eq!(s.bits, 0x38);
    }
}
