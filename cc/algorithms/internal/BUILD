#
# Copyright 2021 Google LLC
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
    default_visibility = ["//algorithms:__subpackages__"],
)

cc_library(
    name = "clamped-calculation-without-bounds",
    hdrs = ["clamped-calculation-without-bounds.h"],
    deps = [
        "//algorithms:util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_cc_differential_privacy//base:status_macros",
    ],
)

cc_test(
    name = "clamped-calculation-without-bounds_test",
    srcs = ["clamped-calculation-without-bounds_test.cc"],
    deps = [
        ":clamped-calculation-without-bounds",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "count-tree",
    srcs = ["count-tree.cc"],
    hdrs = ["count-tree.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_cc_differential_privacy//base:status_macros",
        "@com_google_differential_privacy//proto:summary_cc_proto",
    ],
)

cc_test(
    name = "count-tree_test",
    srcs = ["count-tree_test.cc"],
    deps = [
        ":count-tree",
        "//base/testing:proto_matchers",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bounded-mean-ci",
    srcs = ["bounded-mean-ci.cc"],
    hdrs = ["bounded-mean-ci.h"],
    deps = [
        "//algorithms:numerical-mechanisms",
        "//algorithms:util",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
    ],
)

cc_test(
    name = "bounded-mean-ci_test",
    srcs = ["bounded-mean-ci_test.cc"],
    deps = [
        ":bounded-mean-ci",
        "//algorithms:numerical-mechanisms",
        "//base/testing:proto_matchers",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gaussian-stddev-calculator",
    srcs = ["gaussian-stddev-calculator.cc"],
    hdrs = ["gaussian-stddev-calculator.h"],
)

cc_test(
    name = "gaussian-stddev-calculator_test",
    srcs = ["gaussian-stddev-calculator_test.cc"],
    deps = [
        ":gaussian-stddev-calculator",
        "@com_google_googletest//:gtest_main",
    ],
)
