/**
 * @file interface.cppm
 * @brief Primary interface for wwr.extension.comp
 *
 * The error-handling and RAII layer for backend-neutral batched compression
 * (nvCOMP / hipCOMP, per WWR_GPU_BACKEND). Compression has no element-letter
 * surface, so typed errors + a scratch-buffer RAII are the convenience this layer
 * adds over the stateless batched LLIF in wwr.comp. It aggregates:
 * - :comp_error   - Error code specializations for wwrcompStatus_t (hand-written,
 *                   since wwr.comp exposes no vendor status->string function)
 * - :comp_manager - CompScratch, RAII over the device temp/scratch buffer the
 *                   batched calls borrow (see comp_manager.cppm for the scope note
 *                   on why there is no neutral *manager object* to wrap)
 *
 * Usage:
 *   import wwr.extension.comp;
 *   using namespace wwr::extension;
 */

export module wwr.extension.comp;

import std;

// Re-export the vendor compression module: wwrcompStatus_t is the error type the
// comp_error specializations serve, so a consumer can name it (and the batched
// LLIF functions) without importing wwr.comp separately.
export import wwr.comp;
export import :comp_error;
export import :comp_manager;
