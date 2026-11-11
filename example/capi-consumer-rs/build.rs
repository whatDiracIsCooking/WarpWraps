// Put an rpath to the installed lib dir on THIS binary.
//
// wwr-sys's own build.rs emits the link search + -lwwr_capi, but a build
// script's `cargo:rustc-link-arg` applies only to the crate that emits it --
// it does NOT propagate to a downstream binary's final link. So wwr-sys's rpath
// reaches its own `cargo test` binary but not a consumer like this one, which
// would then find `libwwr_capi.so.1` only via LD_LIBRARY_PATH. A real Rust user
// linking a .so outside the system lib dirs adds the rpath on their binary's
// link; this does exactly that, reading the same WWR_CAPI_LIB_DIR the crate uses.
fn main() {
    println!("cargo:rerun-if-env-changed=WWR_CAPI_LIB_DIR");
    if let Ok(dir) = std::env::var("WWR_CAPI_LIB_DIR") {
        if !dir.is_empty() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }
}
