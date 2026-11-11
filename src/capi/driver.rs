// A PURE RUST consumer — the real thing driver.c has stood in for. It pulls in
// the GENERATED gen/ffi.rs (one extern "C" block, #[repr(C)] PODs, the
// #[repr(i32)] wwr_status enum, ZERO #[cfg]) and links libwwr_capi.so, so this
// proves what driver.c could only approximate: that rustc's own -l lookup finds
// the library, the #[repr(C)] layout matches the C ABI by-value calling
// convention, and the arithmetic + status come back right. It mirrors driver.c's
// assertions so the two stand in for each other. A wrong answer is a panic =>
// non-zero exit, which ctest reads as a failure.
//
// Built rustc-direct (no cargo), matching the spike's minimalism: capi's
// CMakeLists compiles this one file with -L/-l onto the wwr.capi target. The
// backend is chosen only by which wwr library that target linked -- this Rust is
// identical for CUDA or HIP, exactly as the generated ffi.rs promises.

// The generated bindings are not a module path, just a textual include -- the
// demo stays self-contained and carries no build.rs / bindgen step.
#[allow(non_camel_case_types, non_snake_case, dead_code)]
mod ffi {
    include!(concat!(env!("WWR_CAPI_FFI"), "/ffi.rs"));
}

use ffi::*;

fn near_f(got: f32, want: f32) -> bool {
    (got - want).abs() < 1e-4
}

fn near_d(got: f64, want: f64) -> bool {
    (got - want).abs() < 1e-9
}

fn main() {
    let mut ok = true;

    // SAFETY: every call below is into the generated extern "C" thunk over the
    // host-callable wwr functions; the PODs are #[repr(C)] and the out-params are
    // valid local addresses, so each FFI call is sound.
    unsafe {
        let a = wwr_complex64 { re: 3.0, im: 4.0 };
        let b = wwr_complex64 { re: 1.0, im: 2.0 };

        let s = wwr_cadd64(a, b);
        let p = wwr_cmul64(a, b);
        let m = wwr_cabs64(a);
        let q = wwr_cdiv64(a, b);

        println!("add  = {:.4} + {:.4}i   (expect 4.0000 + 6.0000i)", s.re, s.im);
        println!("mul  = {:.4} + {:.4}i   (expect -5.0000 + 10.0000i)", p.re, p.im);
        println!("abs  = {m:.4}             (expect 5.0000)");
        println!("div  = {:.4} + {:.4}i   (expect 2.2000 - 0.4000i)", q.re, q.im);

        ok &= near_f(s.re, 4.0) && near_f(s.im, 6.0);
        ok &= near_f(p.re, -5.0) && near_f(p.im, 10.0);
        ok &= near_f(m, 5.0);
        ok &= near_f(q.re, 2.2) && near_f(q.im, -0.4);

        let a2 = wwr_complex128 { re: 3.0, im: 4.0 };
        let b2 = wwr_complex128 { re: 1.0, im: 2.0 };

        let s2 = wwr_cadd128(a2, b2);
        let p2 = wwr_cmul128(a2, b2);
        let m2 = wwr_cabs128(a2);
        let q2 = wwr_cdiv128(a2, b2);

        println!("add128 = {:.4} + {:.4}i (expect 4.0000 + 6.0000i)", s2.re, s2.im);
        println!("mul128 = {:.4} + {:.4}i (expect -5.0000 + 10.0000i)", p2.re, p2.im);
        println!("abs128 = {m2:.4}           (expect 5.0000)");
        println!("div128 = {:.4} + {:.4}i (expect 2.2000 - 0.4000i)", q2.re, q2.im);

        ok &= near_d(s2.re, 4.0) && near_d(s2.im, 6.0);
        ok &= near_d(p2.re, -5.0) && near_d(p2.im, 10.0);
        ok &= near_d(m2, 5.0);
        ok &= near_d(q2.re, 2.2) && near_d(q2.im, -0.4);

        // The status surface (#329 increment 2), mirroring driver.c. These prove
        // Rust sees the stable wwr_status enum the thunk maps the backend status
        // onto (architecture.md section 6), one mapping whichever backend
        // libwwr_capi links. Every assertion is safe on a no-GPU runner and on
        // both backends -- see each note in driver.c.

        // (1) Structural stability: values come from the GENERATED enum and are
        // identical on either backend -- the fixed layout IS the guarantee.
        ok &= wwr_status::WWR_STATUS_SUCCESS as i32 == 0;
        ok &= wwr_status::WWR_STATUS_INVALID_VALUE != wwr_status::WWR_STATUS_INVALID_DEVICE;
        ok &= wwr_status::WWR_STATUS_SUCCESS != wwr_status::WWR_STATUS_UNKNOWN;

        // (2) Success path -- deterministic with or without a GPU, both backends.
        // The version query needs no device; a healthy toolkit returns a positive
        // version and wwrSuccess, mapped to WWR_STATUS_SUCCESS (== 0).
        let mut ver: i32 = 0;
        let vs = wwr_runtime_get_version(&mut ver);
        println!(
            "runtime version = {}  status = {}  (expect status 0, version > 0)",
            ver, vs as i32
        );
        ok &= vs == wwr_status::WWR_STATUS_SUCCESS;
        ok &= ver > 0;

        // (3) Error path -- a NON-success status, asserted structurally (not by a
        // specific code, same reasoning as driver.c). set_device(-1) is argument
        // validation, rejected independently of GPU presence.
        let bad = wwr_set_device(-1);
        println!("set_device(-1) status = {}  (expect non-zero)", bad as i32);
        ok &= bad != wwr_status::WWR_STATUS_SUCCESS;
    }

    if !ok {
        eprintln!("FAIL: a computed value was wrong");
        std::process::exit(1);
    }
}
