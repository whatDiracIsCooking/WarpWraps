/**
 * @file reinterpret_tag.cppm
 * @brief Disambiguation tag for the reinterpreting (cross-type) view constructor
 *
 * The tag is exported from this partition so the other partitions can name it,
 * and the primary interface unit re-exports the partition because [module.unit]
 * requires every interface partition to be exported by it. So consumers CAN
 * spell reinterpret_view. They should not: reinterpret_buffer_view() is the
 * intended entry point, and direct BufferViewWrapper(reinterpret_view, src)
 * construction is an implementation detail by convention. See interface.cppm for
 * why hiding it is not achievable - the factory is a template, so the tag's
 * definition has to be reachable where it instantiates.
 */

export module wwr.extension.memory_buffer:reinterpret_tag;

export namespace wwr::extension {

/**
 * @brief Tag selecting the reinterpreting (cross-type) view constructor
 *
 * Reinterpreting a buffer's bytes as a different element type is a sharp
 * operation, so it is never an implicit conversion: a view whose element type
 * differs from its source's is only ever formed by naming this tag, which the
 * reinterpret_buffer_view() factory does on the caller's behalf.
 */
struct reinterpret_view_tag_t {
  explicit reinterpret_view_tag_t() = default;
};
inline constexpr reinterpret_view_tag_t reinterpret_view{};

} // namespace wwr::extension
