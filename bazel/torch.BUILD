package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "headers",
    hdrs = glob(
        ["torch/include/**/*.h"],
        ["torch/include/google/protobuf/**/*.h"],
    ),
    strip_include_prefix = "torch/include",
)

# Runtime headers, for importing <torch/torch.h>.
cc_library(
    name = "runtime_headers",
    hdrs = glob(["torch/include/torch/csrc/api/include/**/*.h"]),
    strip_include_prefix = "torch/include/torch/csrc/api/include",
)

# In-tree PyTorch headers (c10/, caffe2/, torch/csrc/, aten/) reached via the
# `-isystem external/torch` copt are declared by no cc_library, so the strict
# include validator rejects them. Declare them here; they are hardlinked to
# torch/include/ by the PyTorch build, so declaring both is inode-safe.
cc_library(
    name = "source_headers",
    hdrs = glob(
        [
            "c10/**/*.h",
            "caffe2/**/*.h",
            "torch/csrc/**/*.h",
            "aten/src/**/*.h",
        ],
        allow_empty = True,
    ),
    # `includes` (not just `hdrs`) is required: it registers external/torch as
    # the include dir the validator matches these headers against.
    includes = ["."],
)

filegroup(
    name = "torchgen_deps",
    srcs = [
        "aten/src/ATen/native/native_functions.yaml",
        "aten/src/ATen/native/tags.yaml",
        "aten/src/ATen/native/ts_native_functions.yaml",
        "aten/src/ATen/templates/DispatchKeyNativeFunctions.cpp",
        "aten/src/ATen/templates/DispatchKeyNativeFunctions.h",
        "aten/src/ATen/templates/LazyIr.h",
        "aten/src/ATen/templates/LazyNonNativeIr.h",
        "aten/src/ATen/templates/RegisterDispatchDefinitions.ini",
        "aten/src/ATen/templates/RegisterDispatchKey.cpp",
        "torch/csrc/lazy/core/shape_inference.h",
        "torch/csrc/lazy/ts_backend/ts_native_functions.cpp",
    ],
)

cc_import(
    name = "libtorch",
    shared_library = "build/lib/libtorch.so",
)

cc_import(
    name = "libtorch_cpu",
    shared_library = "build/lib/libtorch_cpu.so",
)

cc_import(
    name = "libtorch_python",
    shared_library = "build/lib/libtorch_python.so",
)

cc_import(
    name = "libc10",
    shared_library = "build/lib/libc10.so",
)
