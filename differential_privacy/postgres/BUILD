#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The files in this directory implement a postgres extension adding
# anonymous aggregate functions.

licenses(["notice"])  # Apache v2.0

package(
    default_visibility = ["//visibility:private"],
)

config_setting(
    name = "darwin_build",
    values = {"cpu": "darwin"},
)

cc_binary(
    name = "anon_func.so",
    srcs = ["anon_func.cc"],
    copts = [
        "-Wno-sign-compare",
        "-fPIC",
    ] + select({
        ":darwin_build": [
            "-isystem bazel-out/darwin-fastbuild/bin/external/postgres/postgres/include/server",
            "-isystem bazel-out/darwin-fastbuild/bin/external/postgres/postgres/include/internal",
            "-isystem bazel-out/darwin-fastbuild/genfiles/external/postgres/postgres/include/server",
            "-isystem bazel-out/darwin-fastbuild/genfiles/external/postgres/postgres/include/internal",
        ],
        # These paths may need to be changed depending on the system building the binary.
        # eg. "k8-fastbuild" may need to be changed to "darwin-fastbuild".
        "//conditions:default": [
            "-isystem bazel-out/k8-fastbuild/bin/external/postgres/postgres/include/server",
            "-isystem bazel-out/k8-fastbuild/bin/external/postgres/postgres/include/internal",
            "-isystem bazel-out/k8-fastbuild/genfiles/external/postgres/postgres/include/server",
            "-isystem bazel-out/k8-fastbuild/genfiles/external/postgres/postgres/include/internal",
        ],
    }),
    linkopts = [
        "-fno-strict-aliasing",
        "-fwrapv",
        "-fexcess-precision=standard",
        "-fPIC",
        "-shared",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":dp_func",
        "@postgres//:pg_headers",
    ],
)

cc_library(
    name = "dp_func",
    srcs = ["dp_func.cc"],
    hdrs = ["dp_func.h"],
    copts = ["-Wno-sign-compare"],
    deps = [
        "//differential_privacy/algorithms:algorithm",
        "//differential_privacy/algorithms:bounded-mean",
        "//differential_privacy/algorithms:bounded-standard-deviation",
        "//differential_privacy/algorithms:bounded-sum",
        "//differential_privacy/algorithms:bounded-variance",
        "//differential_privacy/algorithms:count",
        "//differential_privacy/algorithms:order-statistics",
        "//differential_privacy/algorithms:util",
    ],
)

cc_test(
    name = "dp_func_test",
    srcs = ["dp_func_test.cc"],
    copts = ["-Wno-sign-compare"],
    deps = [
        ":dp_func",
        "@com_google_googletest//:gtest_main",
    ],
)
