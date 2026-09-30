# RAII wrapper for a GPU surface object -- the sibling of wwr.extension.texture,
# same BaseHandle shape, created from a resource descriptor alone (a surface has
# no texture descriptor). One interface unit, no partitions. See interface.cppm.
wwr_add_cxx_module_library(
  NAME
  wwr.extension.surface
  PRIMARY_INTERFACE
  interface.cppm
  LINK_PUBLIC
  wwr::runtime_api
  wwr::extension::common
  wwr::extension::handle
  IMPORT_STD
)
