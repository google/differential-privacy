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

package(
    default_visibility = ["//visibility:public"],
)
config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

cc_library(
    name = "sequence",
    hdrs = ["sequence.h"],
    copts = select({
        ":windows": [],
        "//conditions:default": ["-Wno-sign-compare"],
    }),
    deps = [
        "//base:logging",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random:distributions",
    ],
)

cc_test(
    name = "sequence_test",
    size = "small",
    srcs = ["sequence_test.cc"],
    copts = select({
        ":windows": [],
        "//conditions:default": ["-Wno-sign-compare"],
    }),
    deps = [
        ":sequence",
        "@com_google_googletest//:gtest_main",
        "//algorithms:util",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "stochastic_tester",
    hdrs = ["stochastic_tester.h"],
    copts = select({
        ":windows": [],
        "//conditions:default": ["-Wno-sign-compare"],
    }),
    deps = [
        ":density_estimation",
        ":sequence",
        "//base:logging",
        "//algorithms:algorithm",
        "//algorithms:util",
        "//proto:util-lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "density_estimation",
    hdrs = ["density_estimation.h"],
    copts = select({
        ":windows": [],
        "//conditions:default": ["-Wno-sign-compare"],
    }),
    deps = [
        "//base:status",
        "//base:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "stochastic_tester_test",
    size = "small",
    timeout = "long",
    srcs = ["stochastic_tester_test.cc"],
    copts = select({
        ":windows": [],
        "//conditions:default": ["-Wno-sign-compare"],
    }),
    shard_count = 10,
    deps = [
        ":sequence",
        ":stochastic_tester",
        "//base:statusor",
        "@com_google_googletest//:gtest_main",
        "//algorithms:algorithm",
        "//algorithms:bounded-sum",
        "//algorithms:count",
        "//algorithms:numerical-mechanisms",
        "//algorithms:numerical-mechanisms-testing",
        "//algorithms:util",
        "@com_google_absl//absl/random:distributions",
    ],
)

cc_test(
    name = "density_estimation_test",
    srcs = ["density_estimation_test.cc"],
    copts = select({
        ":windows": [],
        "//conditions:default": ["-Wno-sign-compare"],
    }),
    deps = [
        ":density_estimation",
        "//base:status",
        "//base/testing:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
