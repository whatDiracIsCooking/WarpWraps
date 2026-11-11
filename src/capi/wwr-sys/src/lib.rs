//! Raw `-sys` bindings over the `wwr` C ABI (`libwwr_capi.so`).
//!
//! The whole surface is the GENERATED `ffi.rs` (one `extern "C"` block,
//! `#[repr(C)]` PODs, the `#[repr(i32)]` `wwr_status` enum, ZERO `#[cfg]`) --
//! `generate.py` writes `src/ffi.rs` here from the same spec it writes the C
//! header and thunk from, so this crate cannot drift from the ABI. Do not edit
//! `ffi.rs` by hand; regenerate with `python3 src/capi/generate.py`.
//!
//! The backend (CUDA or HIP) is chosen only by which `wwr` library the linked
//! `libwwr_capi.so` was built against -- this Rust is identical for either, the
//! whole point of the C ABI. `build.rs` finds that `.so`; see `src/capi/README.md`.
//!
//! This is the raw surface: every function is `unsafe extern "C"`. A safe
//! `Result`-returning wrapper over `wwr_status` is deliberately out of scope
//! here -- the `-sys` crate is the binding, nothing more.

#![allow(non_camel_case_types, non_snake_case)]

// The single binding source, generated alongside gen/ffi.rs from spec/complex.py.
include!("ffi.rs");

#[cfg(test)]
mod tests {
    use super::*;

    // A smoke test that LINKS libwwr_capi.so and CALLS through the FFI: proves
    // build.rs found the lib, the #[repr(C)] by-value layout matches the C ABI,
    // and the arithmetic comes back right. Host-only, both backends -- mirrors
    // driver.rs. Run with WWR_CAPI_LIB_DIR pointing at the built/installed .so.
    #[test]
    fn complex64_arithmetic() {
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
    #[test]
    fn runtime_version_status() {
        let mut ver: i32 = 0;
        // SAFETY: &mut ver is a valid out-param address for the extern "C" call.
        let st = unsafe { wwr_runtime_get_version(&mut ver) };
        assert_eq!(st, wwr_status::WWR_STATUS_SUCCESS);
        assert!(ver > 0);
    }
}
