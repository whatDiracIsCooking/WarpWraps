// A Rust consumer of an INSTALLED wwr -- the end-to-end half the C consumer
// (example/capi-consumer/main.c) stands in for. Where that C project links
// libwwr_capi.so through find_package(wwr), this one reaches the SAME installed
// .so the way a real Rust user does: it depends on the installed `wwr-sys`
// crate (a path dependency on <prefix>/share/wwr/rust/wwr-sys, filled in by
// devtools/install-check.sh), whose build.rs points rustc at the installed lib
// via WWR_CAPI_LIB_DIR=<prefix>/lib. So this binary proves "add wwr-sys as a
// dependency and link the installed lib", not just the crate's own `cargo test`.
//
// Mirrors driver.c / capi-consumer/main.c: calls through the FFI and asserts the
// numbers, returns non-zero on any wrong answer, and 77 when no device runtime
// is reachable -- the ctest/consumer "skip" convention install-check.sh treats
// as a pass rather than a failure (see example/README.md).

use std::process::ExitCode;

// The crate is split per surface (PR #366): the shared PODs live in `common`,
// each surface in its own feature-gated module. Reaching them through the module
// paths (complex::, runtime_api::, common::) documents which surface each call
// belongs to -- this consumer enables both the `complex` and `runtime_api`
// features (the whole default set works too; see Cargo.toml).
use wwr_sys::common::wwr_complex64;
use wwr_sys::complex::{wwr_cabs64, wwr_cadd64};
use wwr_sys::runtime_api::{wwr_runtime_get_version, wwr_status};

fn near_f(got: f32, want: f32) -> bool {
    (got - want).abs() < 1e-4
}

fn main() -> ExitCode {
    // The host arithmetic surface -- deterministic, no device needed, identical
    // on both backends. wwr_cadd64 is the one the issue names; wwr_cabs64 is
    // exercised too so a mis-linked .so that resolves one symbol but not the
    // rest also fails here.
    //
    // SAFETY: by-value calls into the generated extern "C" thunk over the
    // host-callable wwr functions; the PODs are #[repr(C)]. Sound.
    let (s, m) = unsafe {
        let a = wwr_complex64 { re: 3.0, im: 4.0 };
        let b = wwr_complex64 { re: 1.0, im: 2.0 };
        (wwr_cadd64(a, b), wwr_cabs64(a))
    };

    println!("add = {:.4} + {:.4}i  (expect 4.0000 + 6.0000i)", s.re, s.im);
    println!("abs = {m:.4}            (expect 5.0000)");

    let mut ok = near_f(s.re, 4.0) && near_f(s.im, 6.0);
    ok &= near_f(m, 5.0);

    // The status surface's frozen enumerator contract (architecture.md §6/§21):
    // these values come from the installed crate's ffi.rs and are stable across
    // backends, so this asserts the binding travelled correctly with no GPU.
    ok &= (wwr_status::WWR_STATUS_SUCCESS as i32) == 0;
    ok &= wwr_status::WWR_STATUS_SUCCESS != wwr_status::WWR_STATUS_UNKNOWN;

    if !ok {
        eprintln!("FAIL: a computed value was wrong");
        return ExitCode::FAILURE;
    }

    // The runtime version query: positive version + WWR_STATUS_SUCCESS with no
    // device present on a healthy toolkit; non-success on a box with no reachable
    // runtime, which we report as 77 (ctest's "skipped") rather than a failure --
    // install-check.sh builds this everywhere but only a GPU box runs this leg.
    let mut ver: i32 = 0;
    // SAFETY: &mut ver is a valid out-param address for the extern "C" call.
    let vs = unsafe { wwr_runtime_get_version(&mut ver) };
    println!(
        "runtime version = {ver}  status = {}  (expect status 0, version > 0)",
        vs as i32
    );
    if vs != wwr_status::WWR_STATUS_SUCCESS || ver <= 0 {
        eprintln!("no reachable runtime -- skipping the version assertion");
        return ExitCode::from(77);
    }

    ExitCode::SUCCESS
}
