#ifndef XLA_CLIENT_TENSOR_SOURCE_H_
#define XLA_CLIENT_TENSOR_SOURCE_H_

#include <ATen/Tensor.h>
#include <torch/csrc/lazy/core/metrics.h>

#include <atomic>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "torch_xla/csrc/dtype.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/status.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

// AtenSource lifecycle tracking - enable with TF_VLOG or ATENSOURCE_TRACE env var
#ifndef ATENSOURCE_TRACE_ENABLED
#define ATENSOURCE_TRACE_ENABLED 1
#endif

namespace torch_xla {
namespace runtime {

// Owns a contiguous block of data with the shape and layout matching `shape()`.
class TensorSource {
 public:
  TensorSource(std::string device) : device_(std::move(device)) {}

  virtual const void* data() const = 0;

  virtual const xla::Shape& shape() const = 0;

  const std::string& device() const { return device_; }

  virtual std::vector<int64_t> byte_strides() const {
    std::vector<int64_t> byte_strides(shape().dimensions_size());
    MaybeThrow(
        xla::ShapeUtil::ByteStrides(shape(), absl::MakeSpan(byte_strides)));
    return byte_strides;
  }

  virtual std::vector<int64_t> dimensions() const {
    auto dimensions = shape().dimensions();
    return {dimensions.begin(), dimensions.end()};
  }

  virtual xla::PrimitiveType primitive_type() const {
    return shape().element_type();
  }

 private:
  std::string device_;
};

class AtenSource : public TensorSource {
 public:
  // Static tracking for AtenSource instances
  static std::atomic<uint64_t>& LiveCount() {
    static std::atomic<uint64_t> count{0};
    return count;
  }
  static std::atomic<uint64_t>& TotalCreated() {
    static std::atomic<uint64_t> count{0};
    return count;
  }
  static std::atomic<uint64_t>& TotalDestroyed() {
    static std::atomic<uint64_t> count{0};
    return count;
  }
  static std::atomic<int64_t>& LiveBytes() {
    static std::atomic<int64_t> bytes{0};
    return bytes;
  }

  AtenSource(const at::Tensor& tensor, xla::Shape shape, std::string device)
      : TensorSource(std::move(device)), shape_(std::move(shape)) {
    at::ScalarType target_torch_type = TorchTypeFromXlaType(primitive_type());
    if (target_torch_type != tensor.type().scalarType()) {
      TORCH_LAZY_COUNTER("AtenSourceDowncasts", 1);
    }
    // TODO(ysiraichi): check, first, if tensor lives in a device that the
    // current PjRt client has access. If so, we don't need to go through the
    // CPU.
    tensor_ = std::move(
        tensor
            .to(at::TensorOptions().device(at::kCPU).dtype(target_torch_type),
                /*non_blocking=*/false,
                /*copy=*/false, at::MemoryFormat::Contiguous)
            .contiguous());

    // Track this instance
    id_ = TotalCreated().fetch_add(1, std::memory_order_relaxed);
    bytes_ = tensor_.nbytes();
    LiveCount().fetch_add(1, std::memory_order_relaxed);
    LiveBytes().fetch_add(bytes_, std::memory_order_relaxed);

#if ATENSOURCE_TRACE_ENABLED
    // Get storage use_count for debugging reference counting
    int64_t storage_use_count = tensor_.storage().use_count();
    std::cerr << "[AtenSource] CREATED id=" << id_
              << " device=" << this->device()
              << " bytes=" << bytes_
              << " shape=[";
    for (size_t i = 0; i < tensor_.dim(); ++i) {
      if (i > 0) std::cerr << ",";
      std::cerr << tensor_.size(i);
    }
    std::cerr << "]"
              << " storage_use_count=" << storage_use_count
              << " live_count=" << LiveCount().load()
              << " live_bytes=" << LiveBytes().load()
              << std::endl;
#endif
  }

  ~AtenSource() {
    TotalDestroyed().fetch_add(1, std::memory_order_relaxed);
    LiveCount().fetch_sub(1, std::memory_order_relaxed);
    LiveBytes().fetch_sub(bytes_, std::memory_order_relaxed);

#if ATENSOURCE_TRACE_ENABLED
    // Get storage use_count before destruction
    int64_t storage_use_count = tensor_.storage().use_count();
    std::cerr << "[AtenSource] DESTROYED id=" << id_
              << " bytes=" << bytes_
              << " storage_use_count=" << storage_use_count
              << " live_count=" << LiveCount().load()
              << " live_bytes=" << LiveBytes().load()
              << std::endl;
#endif
  }

  const void* data() const override { return tensor_.const_data_ptr(); }

  const xla::Shape& shape() const override { return shape_; }

  std::vector<int64_t> byte_strides() const override {
    std::vector<int64_t> strides;
    for (auto& stride : tensor_.strides()) {
      strides.push_back(stride * tensor_.itemsize());
    }
    return strides;
  }

  std::vector<int64_t> dimensions() const override {
    auto sizes = tensor_.sizes();
    return {sizes.begin(), sizes.end()};
  }

  uint64_t id() const { return id_; }
  int64_t bytes() const { return bytes_; }

 private:
  at::Tensor tensor_;
  xla::Shape shape_;
  uint64_t id_{0};
  int64_t bytes_{0};
};

class LiteralSource : public TensorSource {
 public:
  LiteralSource(xla::Literal literal, std::string device)
      : TensorSource(std::move(device)), literal_(std::move(literal)) {}

  const void* data() const override { return literal_.untyped_data(); }

  const xla::Shape& shape() const override { return literal_.shape(); }

 private:
  xla::Literal literal_;
};

}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_COMPUTATION_CLIENT_H_
