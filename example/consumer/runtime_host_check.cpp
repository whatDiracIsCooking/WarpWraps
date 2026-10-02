// runtime_host_check.cpp -- consuming wwr the way a non-module project does.
//
// main.cpp reaches the runtime API by `import wwr.runtime_api;`. This TU reaches
// the SAME surface the other way a consumer can: #include "wwr/runtime_api.h",
// no import at all. It is the install-check for that path -- that
// wwr/runtime_api.h, the detail/runtime_api_names.h it pulls in, and the
// wwr::runtime_api::host target all travel in the package and re-attach in a
// find_package consumer, so a project not using C++ modules can still use wwr.
//
// A SEPARATE translation unit from main.cpp on purpose: main.cpp imports
// wwr.runtime_api, and importing the module AND #including this header in one TU
// would declare the same wwr* names twice (module-attached vs global-module) --
// ill-formed. Across two TUs linked together it is fine; the names resolve to the
// same vendor entry points in both.
#include "wwr/runtime_api.h"

// Proof is COMPILE + LINK, no device needed -- same contract as main.cpp's
// wrappers_link / extension_link. Compile: the wwr* runtime names exist from a
// pure #include. Link: taking the address of a real backend entry point bound
// through the header forces its symbol to resolve.
bool runtime_host_check() {
  wwr::wwrError_t status = wwr::wwrSuccess;
  wwr::wwrError_t (*get_device)(int *) = &wwr::wwrGetDevice;
  return status == wwr::wwrSuccess && get_device != nullptr;
}
