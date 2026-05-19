#include <torch/csrc/jit/python/pybind_utils.h>

#ifdef USE_DISTRIBUTED
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#endif

namespace torch::jit {

std::optional<InferredType> detail::_tryToInferTypeImpl(py::handle input) {
#ifdef USE_DISTRIBUTED
  if (py::isinstance<c10d::ProcessGroup>(input)) {
    return InferredType(CapsuleType::get());
  }
#endif

  return std::nullopt;
}

} // namespace torch::jit
