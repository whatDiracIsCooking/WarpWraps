"""Read a committed `vendor/<pin>/<lib>.json` manifest and resolve a vendor
symbol to its declaration or enum value.

The fragment (fragments.py) gives the SET of names and each backend's spelling;
the manifest gives the SIGNATURE of each. `vendor_harvest.py` builds these from
the full clang AST, SDK-free and already in CI, so sourcing signatures here means
the C ABI never hand-writes one: it looks up the selected backend's vendor name
and reads the `decl` clang recorded (e.g.
`cudaError_t cudaRuntimeGetVersion(int * runtimeVersion)`), or, for an enum
constant, the `value` clang folded.

Each manifest is ONE (backend, sdk, arch) configuration -- its `harvest.prefix`
is `cuda` or `hip`. The generator loads the pair that matches the pins the rest
of the repo builds against, and cross-references the fragment's per-backend name
against the matching manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str  # clang kind: FunctionDecl, EnumConstantDecl, CXXRecordDecl, ...
    decl: str
    value: int | None  # set only for EnumConstantDecl


class Manifest:
    """One vendor manifest, indexed by symbol name."""

    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self.path = path
        self.prefix: str = data["harvest"]["prefix"]
        self._by_name: dict[str, Symbol] = {
            s["name"]: Symbol(s["name"], s["kind"], s.get("decl", ""), s.get("value"))
            for s in data["symbols"]
        }

    def get(self, vendor_name: str) -> Symbol:
        try:
            return self._by_name[vendor_name]
        except KeyError:
            raise KeyError(
                f"{vendor_name!r} is not in {self.path.name}: the fragment names "
                f"it but the vendor manifest has no such symbol. Check the "
                f"fragment's spelling (or the _RAW override) against the vendor "
                f"header the manifest was harvested from."
            ) from None
