#pragma once

// WWR_LINK_CHECK(sym)
//
// Forces the linker to resolve `sym` at build time without a runtime call.
// Place in the global module fragment of a compile-time test module.
//
// Usage:
//   module;
//   #include "test/shared/link_check.h"
//   WWR_LINK_CHECK(cudaMalloc)
//
// Two things make the check real, and both are needed:
//
//  * [[gnu::used]] -- without it the compiler discards this unused internal
//    variable at -O1 and above, and with it the only reference to `sym`, so the
//    object file never asks the linker for the symbol at all (verified with nm
//    on a Release build: no undefined reference survived).
//  * The test module's object must actually be linked into an executable. The
//    test libraries are static archives, and the linker only pulls an archive
//    member that something references -- so test/cuda/CMakeLists.txt links every
//    test library WHOLE_ARCHIVE. A new test module needs no extra step there;
//    it is picked up from CUDA_TEST_LIBRARIES.
//
// A symbol that is declared by the wrapper but has no definition in any linked
// library now fails the link of cuda_compile_tests with "undefined reference".
#define WWR_LINK_CHECK(sym)                                                                     \
  [[maybe_unused, gnu::used]] static constinit auto *link_check_##sym = &sym;

// WWR_DECLARED_CHECK(sym)
//
// For a symbol the vendor header declares but the CUDA library does not export,
// so WWR_LINK_CHECK cannot pass. Only checks that the name resolves through the
// wrapper module; it takes the address without forcing a reference, so nothing
// reaches the linker. Use it in place of WWR_LINK_CHECK, with a comment naming
// the library version the symbol was found missing from.
//
// You do NOT have to remember to switch back to WWR_LINK_CHECK when a newer
// library starts exporting the symbol: manifest_conformance.py's macro-
// correctness assertion (#121) reads the harvested linkable surface and fails
// the moment a WWR_DECLARED_CHECK names a symbol the pinned .so now exports --
// and, conversely, the moment a WWR_LINK_CHECK names one it does not.
#define WWR_DECLARED_CHECK(sym)                                                                 \
  [[maybe_unused]] static constexpr auto *declared_check_##sym = &sym;
