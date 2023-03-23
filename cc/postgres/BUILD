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

licenses(["notice"])

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
        "-fPIC",
    ],
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
    deps = [
        "//algorithms:algorithm",
        "//algorithms:bounded-mean",
        "//algorithms:bounded-standard-deviation",
        "//algorithms:bounded-sum",
        "//algorithms:bounded-variance",
        "//algorithms:count",
        "//algorithms:order-statistics",
    ],
)

cc_test(
    name = "dp_func_test",
    srcs = ["dp_func_test.cc"],
    deps = [
        ":dp_func",
        "@com_google_googletest//:gtest_main",
    ],
)
