# Debian's python3-dev splits pyconfig.h: the copy pybind11_bazel places in
# @local_config_python is only a wrapper that includes the real
# <x86_64-linux-gnu/python3.12/pyconfig.h>, which is never copied and is thus
# undeclared under strict include validation. Provide the real header here.
# Non-Debian toolchains (e.g. the pyenv Python used by the release build) don't
# split pyconfig.h and never include the multiarch path, so emit an empty,
# never-included stub when the multiarch header is absent.
genrule(
    name = "multiarch_pyconfig_h",
    outs = ["python_include/x86_64-linux-gnu/python3.12/pyconfig.h"],
    cmd = "cp -f /usr/include/x86_64-linux-gnu/python3.12/pyconfig.h $@ 2>/dev/null || touch $@",
)

cc_library(
    name = "multiarch_pyconfig",
    hdrs = [":multiarch_pyconfig_h"],
    includes = ["python_include"],
    visibility = ["//visibility:public"],
)
