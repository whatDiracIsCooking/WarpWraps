// Locate the prebuilt libwwr_capi.so and point the linker at it.
//
// The .so is built/installed by the wwr CMake project, not by cargo (cargo does
// not know how to build a C++23-module target). So this crate does NOT compile
// anything -- it only tells rustc WHERE the lib is and WHAT to link:
//
//   WWR_CAPI_LIB_DIR  the directory holding libwwr_capi.so -- the CMake build's
//                     lib dir (where `cmake --build ... --target wwr.capi` put
//                     it) or the install prefix's lib dir. The primary knob.
//   pkg-config        a fallback, if a future install ships a wwr_capi.pc; not
//                     relied on today, so a missing pkg-config is not an error.
//
// The link name is the lib's OUTPUT_NAME `wwr_capi` (-> libwwr_capi.so), NOT the
// dotted CMake target wwr.capi: rustc's -l does a literal libwwr_capi.so lookup.
// It must stay in step with `OUTPUT_NAME wwr_capi` in src/capi/CMakeLists.txt,
// `links = "wwr_capi"` in Cargo.toml, and ffi.rs's #[link(name = "wwr_capi")]
// -- change one, change all (the increment-3 naming lesson). This const is the
// one spelling build.rs owns.
const LIB_NAME: &str = "wwr_capi";

fn main() {
    println!("cargo:rerun-if-env-changed=WWR_CAPI_LIB_DIR");

    if let Ok(dir) = std::env::var("WWR_CAPI_LIB_DIR") {
        if !dir.is_empty() {
            println!("cargo:rustc-link-search=native={dir}");
            // An rpath so the test binary (and any consumer that opts in) finds
            // the .so at run time without LD_LIBRARY_PATH. Harmless for a real
            // install, which resolves via the system lib dir.
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }

    // dylib: link the shared libwwr_capi.so (the only artifact the target
    // produces). Emitted unconditionally so the search path above -- or a system
    // lib dir / pkg-config path -- can satisfy it.
    println!("cargo:rustc-link-lib=dylib={LIB_NAME}");
}
