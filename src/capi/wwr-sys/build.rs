// Locate the prebuilt per-unit libwwr_capi_<unit>.so's and point the linker at
// ONLY the ones the enabled features need.
//
// The .so's are built/installed by the wwr CMake project, not by cargo (cargo
// does not know how to build a C++23-module target). So this crate does NOT
// compile anything -- it only tells rustc WHERE the libs are and WHICH to link:
//
//   WWR_CAPI_LIB_DIR  the directory holding the libwwr_capi_<unit>.so's -- the
//                     CMake build's lib dir (where `cmake --build ...` put them)
//                     or the install prefix's lib dir. The primary knob.
//
// The PER-UNIT isolation lives here: one rustc-link-lib per ENABLED feature
// (CARGO_FEATURE_<FEATURE> is set by cargo when the feature is on), so a build
// with `--no-default-features --features complex` links libwwr_capi_complex.so
// and NOTHING of blas -> the consumer's binary NEEDs no cuBLAS. The default (all
// features) links all three, the drop-in replacement for the old umbrella link.
//
// A link name is the lib's OUTPUT_NAME `wwr_capi_<unit>` (-> libwwr_capi_<unit>.so),
// NOT the dotted CMake target wwr.capi.<unit>: rustc's -l does a literal file
// lookup. Each must stay in step with the `OUTPUT_NAME wwr_capi_<unit>` set in
// src/capi/CMakeLists.txt -- change one, change both.

// (cargo feature name, the unit's link-lib stem). The feature env var cargo sets
// is CARGO_FEATURE_<NAME uppercased, '-' -> '_'>; our feature names are already
// lower snake_case, so the mapping is a straight uppercase.
const UNITS: &[(&str, &str)] = &[
    ("complex", "wwr_capi_complex"),
    ("runtime_api", "wwr_capi_runtime_api"),
    ("blas", "wwr_capi_blas"),
    ("blaslt", "wwr_capi_blaslt"),
    ("solver", "wwr_capi_solver"),
    ("rand", "wwr_capi_rand"),
    ("rtc", "wwr_capi_rtc"),
    ("sparse", "wwr_capi_sparse"),
    ("tensor", "wwr_capi_tensor"),
    ("fft", "wwr_capi_fft"),
    ("comp", "wwr_capi_comp"),
    ("fp16", "wwr_capi_fp16"),
    ("bf16", "wwr_capi_bf16"),
    ("vector_types", "wwr_capi_vector_types"),
    ("fp8", "wwr_capi_fp8"),
    ("tx", "wwr_capi_tx"),
    ("ccl", "wwr_capi_ccl"),
];

fn main() {
    println!("cargo:rerun-if-env-changed=WWR_CAPI_LIB_DIR");

    if let Ok(dir) = std::env::var("WWR_CAPI_LIB_DIR") {
        if !dir.is_empty() {
            println!("cargo:rustc-link-search=native={dir}");
            // An rpath so the test binary (and any consumer that opts in) finds
            // the .so's at run time without LD_LIBRARY_PATH. Harmless for a real
            // install, which resolves via the system lib dir.
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }

    // One dylib link per ENABLED feature -- this is the per-unit isolation. A
    // disabled surface contributes no link, so its vendor dependency is absent
    // from the final binary's NEEDED set.
    for (feature, lib) in UNITS {
        let var = format!("CARGO_FEATURE_{}", feature.to_uppercase());
        if std::env::var_os(&var).is_some() {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
}
