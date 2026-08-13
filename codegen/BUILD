# Use rules_python's py_binary, not the native rule: rules_python substitutes the
# bootstrap template's %interpreter_args% placeholder, which the native rule
# leaves in place and crashes this generator with a SyntaxError.
load("@rules_python//python:py_binary.bzl", "py_binary")

exports_files(["xla_native_functions.yaml"])

# Requires `torchgen` locally, via pip `torch` package or local builds.
py_binary(
    name = "lazy_tensor_generator",
    srcs = ["lazy_tensor_generator.py"],
    data = [
        "//torch_xla/csrc:aten_xla_type.cpp",
        "@torch//:torchgen_deps",
    ],
    deps = [
        "@pypi_torch//:pkg",
        "@pypi_pyyaml//:pkg",
    ],
    tags = [
        "local",
        "no-remote-exec",
    ],
)

sh_binary(
    name = "fix_includes",
    srcs = ["fix_includes.sh"],
)
