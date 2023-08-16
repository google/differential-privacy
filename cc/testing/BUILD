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

cc_library(
    name = "sequence",
    hdrs = ["sequence.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random:distributions",
    ],
)

cc_test(
    name = "sequence_test",
    size = "small",
    srcs = ["sequence_test.cc"],
    deps = [
        ":sequence",
        "//algorithms:util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "stochastic_tester",
    hdrs = ["stochastic_tester.h"],
    deps = [
        ":density_estimation",
        ":sequence",
        "//algorithms:algorithm",
        "//algorithms:util",
        "//proto:util-lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "density_estimation",
    hdrs = ["density_estimation.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "stochastic_tester_test",
    size = "small",
    timeout = "long",
    srcs = ["stochastic_tester_test.cc"],
    shard_count = 10,
    deps = [
        ":sequence",
        ":stochastic_tester",
        "//algorithms:algorithm",
        "//algorithms:bounded-sum",
        "//algorithms:count",
        "//algorithms:numerical-mechanisms",
        "//algorithms:numerical-mechanisms-testing",
        "//algorithms:util",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "density_estimation_test",
    srcs = ["density_estimation_test.cc"],
    deps = [
        ":density_estimation",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "statistical_tests_utils",
    testonly = 1,
    srcs = ["statistical_tests_utils.cc"],
    hdrs = ["statistical_tests_utils.h"],
    deps = [
        "//algorithms:rand",
        "//algorithms:util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_cc_differential_privacy//base:status_macros",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "statistical_tests_utils_test",
    srcs = [
        "statistical_tests_utils_test.cc",
    ],
    deps = [
        ":statistical_tests_utils",
        "//algorithms:util",
        "//base/testing:proto_matchers",
        "@com_google_differential_privacy//proto/testing:statistical_tests_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "quantile_tree_dp_tests",
    timeout = "long",
    srcs = [
        "quantile_tree_dp_test.cc",
    ],
    data = [
        "@com_google_differential_privacy//proto/testing:bounded_quantiles_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in bounded_quantiles_dp_test_cases.textproto
    shard_count = 39,
    deps = [
        ":statistical_tests_utils",
        "//algorithms:numerical-mechanisms",
        "//algorithms:quantile-tree",
        "@com_google_differential_privacy//proto/testing:statistical_tests_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)
