#ifndef XLA_CLIENT_TF_LOGGING_H_
#define XLA_CLIENT_TF_LOGGING_H_

#include <sstream>

#include "absl/base/optimization.h"
#include "absl/log/absl_log.h"

#include "torch_xla/csrc/runtime/tsl_platform_logging.h"

namespace torch_xla {
namespace runtime {
namespace internal {

// TODO: replace all TF_*LOG macro calls with ABSL_*LOG()
//
// Why are we using Abseil logging, now?
// =====================================
// Ref: https://github.com/openxla/xla/pull/29477
//
// OpenXLA removed their internal logging in favor of Abseil.
//
// Why do we have the `TF_` prefix?
// ================================
// Ref: https://github.com/pytorch/xla/pull/34
//
// So as not to clash with C10 definition.
#define TF_LOG(severity) ABSL_LOG(severity)
#define TF_VLOG(level) ABSL_VLOG(level)

struct ErrorSink : public std::basic_ostringstream<char> {};

class ErrorGenerator {
 public:
  ErrorGenerator(const char* file, int line) : file_(file), line_(line) {}

  // Use a dummy & operator as it has lower precedence WRT the streaming
  // operator, and hence allows collecting user error messages before we finally
  // throw.
  ABSL_ATTRIBUTE_NORETURN void operator&(
      const std::basic_ostream<char>& oss) const;

 private:
  const char* file_ = nullptr;
  int line_ = 0;
};

#define TF_ERROR_STREAM()                                              \
  ::torch_xla::runtime::internal::ErrorGenerator(__FILE__, __LINE__) & \
      ::torch_xla::runtime::internal::ErrorSink()

#define TF_CHECK(condition)              \
  while (TF_PREDICT_FALSE(!(condition))) \
  TF_ERROR_STREAM() << "Check failed: " #condition ": "

#define TF_CHECK_OP_LOG(name, op, val1, val2)                      \
  while (::tsl::torch_xla::internal::CheckOpString _result{        \
      ::tsl::torch_xla::internal::name##Impl(                      \
          ::tsl::torch_xla::internal::GetReferenceableValue(val1), \
          ::tsl::torch_xla::internal::GetReferenceableValue(val2), \
          #val1 " " #op " " #val2)})                               \
  TF_ERROR_STREAM() << *(_result.str_)

#define TF_CHECK_OP(name, op, val1, val2) TF_CHECK_OP_LOG(name, op, val1, val2)

// TF_CHECK_EQ/NE/...
#define TF_CHECK_EQ(val1, val2) TF_CHECK_OP(Check_EQ, ==, val1, val2)
#define TF_CHECK_NE(val1, val2) TF_CHECK_OP(Check_NE, !=, val1, val2)
#define TF_CHECK_LE(val1, val2) TF_CHECK_OP(Check_LE, <=, val1, val2)
#define TF_CHECK_LT(val1, val2) TF_CHECK_OP(Check_LT, <, val1, val2)

// Check_GEImpl and Check_GTImpl are actually implemented in terms of their
// less-than versions. So, here, we do the same so that the error message
// is consistent.
#define TF_CHECK_GE(val1, val2) TF_CHECK_LE(val2, val1)
#define TF_CHECK_GT(val1, val2) TF_CHECK_LT(val2, val1)

// Streamable, single-evaluation status check. The argument is evaluated exactly
// once (some call sites pass function calls, e.g. XLA_CHECK_OK(Fn())), and
// callers may append context with operator<<. Backs XLA_CHECK_OK (see
// debug_macros.h). The loop body is [[noreturn]] (ErrorGenerator throws), so it
// runs at most once.
#define TF_CHECK_OK(val)                              \
  for (auto&& _tf_check_status = (val);               \
       ABSL_PREDICT_FALSE(!(_tf_check_status).ok());) \
  TF_ERROR_STREAM() << "Check failed: " #val " is not OK: "

#define TF_CHECK_NOTNULL(val) TF_CHECK((val) != nullptr)

}  // namespace internal
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_TF_LOGGING_H_
