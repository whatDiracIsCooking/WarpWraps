# The in-tree examples. See README.md for what belongs here.
#
# consumer/ is deliberately absent: it is a standalone project with its own
# project() call, configured against an INSTALLED wwr by
# devtools/install-check.sh. Adding it here would give it wwr's source tree,
# which is the one thing it must not have.
#
# Added unconditionally, WWR_COMPILE_TIME_ONLY included, so
# devtools/cross-backend-check.sh compiles what is here for the other backend.
# An example that only ever builds on one backend is not an example of portable
# code.
add_subdirectory(warp_reduce)
