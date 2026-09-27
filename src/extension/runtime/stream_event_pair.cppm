/**
 * @file stream_event_pair.cppm
 * @brief Pair of GPU stream and event handles
 *
 * Provides StreamEventPair class bundling a GpuStream and GpuEvent together.
 */

export module wwr.extension.runtime:stream_event_pair;

import :gpu_stream;
import :gpu_event;
import :convenience_runtime;
import wwr.extension.common;
import wwr.runtime_api;
import std;

export namespace wwr::extension {

/**
 * @brief Configuration for StreamEventPair construction.
 *
 * The stream and event are both created on `device` (default 0). The flag/
 * priority fields are optional — unset fields use the default GPU creation API.
 * If stream_priority is set without stream_flags, flags default to 0.
 */
struct StreamEventConfig {
  int device = 0;
  std::optional<unsigned int> stream_flags;
  std::optional<int> stream_priority;
  std::optional<unsigned int> event_flags;
};

/**
 * @brief Bundles a GpuStream and GpuEvent together.
 *
 * Owns a stream and an associated event. Construction is controlled via
 * StreamEventConfig to allow optional flags and priority.
 */
class StreamEventPair : private NonCopyable {
private:
  GpuStream stream_;
  GpuEvent event_;

  static GpuStream make_stream(const StreamEventConfig &cfg) {
    if (cfg.stream_priority)
      return GpuStream(cfg.device, cfg.stream_flags.value_or(gpuStreamDefault),
                       *cfg.stream_priority);
    if (cfg.stream_flags)
      return GpuStream(cfg.device, *cfg.stream_flags);
    return GpuStream(cfg.device);
  }

public:
  StreamEventPair() = default;

  explicit StreamEventPair(StreamEventConfig cfg)
      : stream_(make_stream(cfg)),
        event_(cfg.event_flags ? GpuEvent(cfg.device, *cfg.event_flags) : GpuEvent(cfg.device)) {}

  // Copy operations are implicitly deleted via the NonCopyable base.

  StreamEventPair(StreamEventPair &&) = default;
  StreamEventPair &operator=(StreamEventPair &&) = default;

  GpuStream &gpu_stream() { return stream_; }
  const GpuStream &gpu_stream() const { return stream_; }

  GpuEvent &gpu_event() { return event_; }
  const GpuEvent &gpu_event() const { return event_; }

  gpuStream_t stream_raw() const { return stream_.get(); }
  gpuEvent_t event_raw() const { return event_.get(); }

  gpuError_t record() { return event_.record(stream_.get()); }
  gpuError_t record(const unsigned int flags) { return event_.record(stream_.get(), flags); }
  gpuError_t stream_sync() { return stream_.sync(); }
  gpuError_t event_sync() { return event_.sync(); }
};

} // namespace wwr::extension
