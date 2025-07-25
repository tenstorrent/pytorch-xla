#include "torch_xla/csrc/shape_helper.h"

#include "torch_xla/csrc/status.h"
#include "xla/hlo/builder/xla_builder.h"

namespace torch_xla {

const xla::Shape& ShapeHelper::ShapeOfXlaOp(xla::XlaOp op) {
  return *GetValueOrThrow(GetShape(op));
}

absl::StatusOr<const xla::Shape * absl_nonnull> GetShape(xla::XlaOp op) {
  return op.builder()->GetShapePtr(op);
}

}  // namespace torch_xla
