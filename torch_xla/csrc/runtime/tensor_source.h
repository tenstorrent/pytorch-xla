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
    bool dtype_mismatch = (target_torch_type != tensor.type().scalarType());
    if (dtype_mismatch) {
      TORCH_LAZY_COUNTER("AtenSourceDowncasts", 1);
    }

#if ATENSOURCE_TRACE_ENABLED
    // Capture input tensor info BEFORE .to().contiguous()
    uintptr_t input_data_ptr = reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
    uintptr_t input_storage_ptr = reinterpret_cast<uintptr_t>(tensor.storage().data());
    int64_t input_storage_use_count = tensor.storage().use_count();
    bool input_is_contiguous = tensor.is_contiguous();
#endif

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
    // Get output tensor info AFTER .to().contiguous()
    uintptr_t output_data_ptr = reinterpret_cast<uintptr_t>(tensor_.const_data_ptr());
    uintptr_t output_storage_ptr = reinterpret_cast<uintptr_t>(tensor_.storage().data());
    int64_t output_storage_use_count = tensor_.storage().use_count();
    bool same_storage = (input_storage_ptr == output_storage_ptr);
    bool same_data = (input_data_ptr == output_data_ptr);

    std::cerr << "[AtenSource] CREATED id=" << id_
              << " device=" << this->device()
              << " bytes=" << bytes_
              << std::hex
              << " input_data_ptr=0x" << input_data_ptr
              << " input_storage_ptr=0x" << input_storage_ptr
              << " output_data_ptr=0x" << output_data_ptr
              << " output_storage_ptr=0x" << output_storage_ptr
              << std::dec
              << " input_storage_usecount=" << input_storage_use_count
              << " output_storage_usecount=" << output_storage_use_count
              << " input_is_contiguous=" << input_is_contiguous
              << " dtype_mismatch=" << dtype_mismatch
              << " same_storage=" << same_storage
              << " same_data=" << same_data
              << " shape=[";
    for (size_t i = 0; i < tensor_.dim(); ++i) {
      if (i > 0) std::cerr << ",";
      std::cerr << tensor_.size(i);
    }
    std::cerr << "]"
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
    // Get info before destruction
    int64_t storage_use_count = tensor_.storage().use_count();
    uintptr_t data_ptr = reinterpret_cast<uintptr_t>(tensor_.const_data_ptr());
    uintptr_t storage_ptr = reinterpret_cast<uintptr_t>(tensor_.storage().data());
    int64_t storage_nbytes = tensor_.storage().nbytes();

    std::cerr << "[AtenSource] DESTROYING id=" << id_
              << " bytes=" << bytes_
              << std::hex
              << " data_ptr=0x" << data_ptr
              << " storage_ptr=0x" << storage_ptr
              << std::dec
              << " storage_nbytes=" << storage_nbytes
              << " storage_use_count=" << storage_use_count
              << " live_count=" << LiveCount().load()
              << " live_bytes=" << LiveBytes().load();

    // Check if this is the last reference - memory should be freed after this
    if (storage_use_count == 1) {
      std::cerr << " [LAST_REF - memory should free]";
    }
    std::cerr << std::endl;
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
