/**
 * @file blaslt_matmul.cppm
 * @brief RAII wrappers for the cuBLASLt / hipBLASLt matmul descriptor zoo
 *
 * The modern "Lt" GEMM surface configures a matmul through opaque descriptor
 * objects, each hand-paired create/destroy -- the exact leak-prone shape RAII
 * removes. This partition wraps the three a matmul needs:
 *
 * - MatmulDesc       (wwrblasLtMatmulDesc_t)       - the operation descriptor:
 *                     compute type, scale type, transpose/epilogue attributes.
 * - MatrixLayout     (wwrblasLtMatrixLayout_t)     - one per operand (A/B/C/D):
 *                     element data type and the rows/cols/ld shape.
 * - MatmulPreference (wwrblasLtMatmulPreference_t) - heuristic-search preferences
 *                     (e.g. workspace limit) for wwrblasLtMatmulAlgoGetHeuristic.
 *
 * Each is a thin BaseHandle subclass. All three handle types are pointers on both
 * backends, so they ride the null-sentinel liveness path. Their create calls take
 * arguments (unlike a bare handle), so each takes BaseHandle's
 * skip-default-create path: the constructor builds the object itself from its
 * create arguments and records ownership with adopt(), exactly as TextureObject /
 * SurfaceObject do.
 *
 * The error type is wwrblasLtStatus_t (== wwrblasStatus_t); its policy comes from
 * wwr.extension.blas via :blaslt_error.
 *
 * The scale type (MatmulDesc) and element data type (MatrixLayout) are a vendor
 * cudaDataType / hipDataType enum. src/blaslt.cppm deliberately does not surface
 * that type under a wwrblasLt* name (it is cudaDataType/hipDataType, which
 * wwr.solver already neutralises as wwrsolverDataType_t), so rather than reach
 * cross-library or name a backend-specific type in this neutral module, the data
 * type is a deduced template parameter: the caller passes whatever enumerator the
 * vendor expects (e.g. a WWRSOLVER_R_32F from wwr.solver, or a raw CUDA_R_32F /
 * HIP_R_32F from the vendor module). The compute type IS neutral
 * (wwrblasLtComputeType_t) and is named directly.
 *
 * Deliberately omitted: the matrix-transform descriptor
 * (wwrblasLtMatrixTransformDesc_t) -- it belongs to wwrblasLtMatrixTransform, a
 * standalone transpose/convert op, not the matmul path this layer targets; and
 * the matmul-algo / heuristic-result objects, which are plain value structs the
 * vendor fills in, not create/destroy-paired resources needing RAII. Attributes
 * are set afterwards through the raw wwrblasLt*SetAttribute calls on the handle
 * each wrapper converts to.
 */

export module wwr.extension.blaslt:blaslt_matmul;

import :blaslt_error; // re-exports wwr.extension.blas: the wwrblasLtStatus_t policy
import wwr.blaslt;
import wwr.extension.common;
import wwr.extension.handle;
import std;

export namespace wwr::extension {

/**
 * @brief RAII wrapper for a cuBLASLt / hipBLASLt matmul operation descriptor
 *
 * @tparam P_create Error policy for creation, typed to wwrblasLtStatus_t
 * @tparam P_destroy Error policy for destruction, typed to wwrblasLtStatus_t
 *
 * @note P_destroy MUST NOT THROW -- it is called from the destructor.
 */
template<error_policy<wwrblasLtStatus_t> P_create,
         nothrow_error_policy<wwrblasLtStatus_t> P_destroy>
class MatmulDesc
    : public BaseHandle<wwrblasLtMatmulDesc_t, MatmulDesc<P_create, P_destroy>, P_create, P_destroy> {
  using Base =
      BaseHandle<wwrblasLtMatmulDesc_t, MatmulDesc<P_create, P_destroy>, P_create, P_destroy>;

public:
  /// @brief Create a matmul descriptor
  /// @tparam DataType The vendor scale-type enum (cudaDataType / hipDataType)
  /// @param compute_type The matmul's compute type
  /// @param scale_type The scale (alpha/beta) data type
  /// @param policy_create Error policy for the create call
  /// @param policy_destroy Error policy for the destroy call (must not throw)
  /// @param location Source location where creation was requested
  template<typename DataType>
  MatmulDesc(wwrblasLtComputeType_t compute_type, DataType scale_type, P_create policy_create = {},
             P_destroy policy_destroy = {},
             std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}, std::move(policy_create),
             std::move(policy_destroy)) {
    this->adopt(gpu_check(wwrblasLtMatmulDescCreate(&this->handle_, compute_type, scale_type),
                          this->create_policy(), location));
  }

  /// @brief Destroy the matmul descriptor (BaseHandle destructor hook)
  /// @param handle The descriptor to destroy
  void destroy(wwrblasLtMatmulDesc_t handle) {
    if (handle != nullptr) {
      gpu_check(wwrblasLtMatmulDescDestroy(handle), this->destroy_policy());
    }
  }
};

/**
 * @brief RAII wrapper for a cuBLASLt / hipBLASLt matrix layout descriptor
 *
 * @tparam P_create Error policy for creation, typed to wwrblasLtStatus_t
 * @tparam P_destroy Error policy for destruction, typed to wwrblasLtStatus_t
 *
 * @note P_destroy MUST NOT THROW -- it is called from the destructor.
 */
template<error_policy<wwrblasLtStatus_t> P_create,
         nothrow_error_policy<wwrblasLtStatus_t> P_destroy>
class MatrixLayout
    : public BaseHandle<wwrblasLtMatrixLayout_t, MatrixLayout<P_create, P_destroy>, P_create,
                        P_destroy> {
  using Base =
      BaseHandle<wwrblasLtMatrixLayout_t, MatrixLayout<P_create, P_destroy>, P_create, P_destroy>;

public:
  /// @brief Create a matrix layout descriptor
  /// @tparam DataType The vendor element-type enum (cudaDataType / hipDataType)
  /// @param data_type The matrix element data type
  /// @param rows Number of rows
  /// @param cols Number of columns
  /// @param ld Leading dimension
  /// @param policy_create Error policy for the create call
  /// @param policy_destroy Error policy for the destroy call (must not throw)
  /// @param location Source location where creation was requested
  template<typename DataType>
  MatrixLayout(DataType data_type, std::uint64_t rows, std::uint64_t cols, std::int64_t ld,
               P_create policy_create = {}, P_destroy policy_destroy = {},
               std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}, std::move(policy_create),
             std::move(policy_destroy)) {
    this->adopt(gpu_check(wwrblasLtMatrixLayoutCreate(&this->handle_, data_type, rows, cols, ld),
                          this->create_policy(), location));
  }

  /// @brief Destroy the matrix layout descriptor (BaseHandle destructor hook)
  /// @param handle The layout to destroy
  void destroy(wwrblasLtMatrixLayout_t handle) {
    if (handle != nullptr) {
      gpu_check(wwrblasLtMatrixLayoutDestroy(handle), this->destroy_policy());
    }
  }
};

/**
 * @brief RAII wrapper for a cuBLASLt / hipBLASLt matmul preference descriptor
 *
 * @tparam P_create Error policy for creation, typed to wwrblasLtStatus_t
 * @tparam P_destroy Error policy for destruction, typed to wwrblasLtStatus_t
 *
 * @note P_destroy MUST NOT THROW -- it is called from the destructor.
 */
template<error_policy<wwrblasLtStatus_t> P_create,
         nothrow_error_policy<wwrblasLtStatus_t> P_destroy>
class MatmulPreference
    : public BaseHandle<wwrblasLtMatmulPreference_t, MatmulPreference<P_create, P_destroy>, P_create,
                        P_destroy> {
  using Base = BaseHandle<wwrblasLtMatmulPreference_t, MatmulPreference<P_create, P_destroy>,
                          P_create, P_destroy>;

public:
  /// @brief Create a matmul preference descriptor
  /// @param policy_create Error policy for the create call
  /// @param policy_destroy Error policy for the destroy call (must not throw)
  /// @param location Source location where creation was requested
  MatmulPreference(P_create policy_create = {}, P_destroy policy_destroy = {},
                   std::source_location location = std::source_location::current())
      : Base(typename Base::skip_default_create_t{}, std::move(policy_create),
             std::move(policy_destroy)) {
    this->adopt(gpu_check(wwrblasLtMatmulPreferenceCreate(&this->handle_), this->create_policy(),
                          location));
  }

  /// @brief Destroy the matmul preference descriptor (BaseHandle destructor hook)
  /// @param handle The preference to destroy
  void destroy(wwrblasLtMatmulPreference_t handle) {
    if (handle != nullptr) {
      gpu_check(wwrblasLtMatmulPreferenceDestroy(handle), this->destroy_policy());
    }
  }
};

} // namespace wwr::extension
