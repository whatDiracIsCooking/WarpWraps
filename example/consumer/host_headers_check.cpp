// host_headers_check.cpp -- consuming the fp16 / bf16 / rand surfaces the
// non-module way, by #include rather than import.
//
// The runtime_host_check.cpp twin for the three modules rolled onto the
// header-consumption path after runtime_api. main.cpp reaches these surfaces by
// `import wwr.fp16;` etc. (indirectly, through the layers it uses); this TU
// reaches the SAME surfaces the other way a consumer can -- #include "wwr/fp16.h",
// "wwr/bf16.h", "wwr/rand.h", no import at all. It is the install-check for those
// paths: that the wwr/*.h headers, the detail/*_names.h fragments they pull in,
// and the wwr::fp16::host / wwr::bf16::host / wwr::rand::host targets all travel
// in the package and re-attach in a find_package consumer.
//
// A SEPARATE translation unit on purpose, for the reason runtime_host_check.cpp
// documents: a TU that both imports a module and #includes its header twin would
// declare the same wwr* names twice. Across TUs linked together it is fine.
#include "wwr/bf16.h"
#include "wwr/fp16.h"
#include "wwr/rand.h"

// Proof is COMPILE + LINK, no device needed. Compile: the wwr* names exist from a
// pure #include. Link: calling the conversion forwarders and taking the address
// of a real RNG entry point bound through the header forces the symbols to
// resolve.
bool host_headers_check() {
  const wwr::wwrHalf half = wwr::wwrFloat2Half(1.5f);
  const wwr::wwrBfloat16 bf = wwr::wwrFloat2Bfloat16(1.5f);

  wwr::wwrrandStatus_t (*create)(wwr::wwrrandGenerator_t *, wwr::wwrrandRngType_t) =
      &wwr::wwrrandCreateGenerator;

  return wwr::wwrHalf2Float(half) == 1.5f && wwr::wwrBfloat162Float(bf) == 1.5f &&
         create != nullptr;
}
