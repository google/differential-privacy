#
# Copyright 2020 Google LLC
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

cc_binary(
    name = "part1",
    srcs = ["main.cc"],
    deps = [
        ":insufficient_noise_algorithms",
        ":create_samples_count",
        ":create_samples_sum",
        ":create_samples_mean",
        "@com_google_absl//absl/strings",
        "@com_google_cc_differential_privacy//algorithms:bounded-mean",
        "@com_google_cc_differential_privacy//algorithms:bounded-sum",
        "@com_google_cc_differential_privacy//algorithms:count",
        "@com_google_cc_differential_privacy//algorithms:order-statistics",
        "@com_google_cc_differential_privacy//base:statusor",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_cc_differential_privacy//algorithms:util",
        "@com_google_cc_differential_privacy//proto:util-lib",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_google_cc_differential_privacy//algorithms:algorithm",
        "@com_google_cc_differential_privacy//algorithms:numerical-mechanisms",
        "@com_google_absl//absl/random:distributions",
        "@com_google_cc_differential_privacy//base:status",
        "@com_google_cc_differential_privacy//testing:sequence",
        "@com_google_cc_differential_privacy//testing:stochastic_tester",
    ],
)

cc_library(
    name = "create_samples_count",
    srcs = ["create_samples_count.cc"],
    hdrs = ["create_samples_count.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_cc_differential_privacy//algorithms:bounded-mean",
        "@com_google_cc_differential_privacy//algorithms:bounded-sum",
        "@com_google_cc_differential_privacy//algorithms:count",
        "@com_google_cc_differential_privacy//algorithms:order-statistics",
        "@com_google_cc_differential_privacy//base:statusor",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_cc_differential_privacy//algorithms:util",
        "@com_google_cc_differential_privacy//proto:util-lib",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_google_cc_differential_privacy//algorithms:algorithm",
        "@com_google_cc_differential_privacy//algorithms:numerical-mechanisms",
        "@com_google_absl//absl/random:distributions",
        "@com_google_cc_differential_privacy//base:status",
        "@com_google_cc_differential_privacy//testing:sequence",
    ],
)

cc_library(
    name = "create_samples_sum",
    srcs = ["create_samples_sum.cc"],
    hdrs = ["create_samples_sum.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_cc_differential_privacy//algorithms:bounded-mean",
        "@com_google_cc_differential_privacy//algorithms:bounded-sum",
        "@com_google_cc_differential_privacy//algorithms:count",
        "@com_google_cc_differential_privacy//algorithms:order-statistics",
        "@com_google_cc_differential_privacy//base:statusor",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_cc_differential_privacy//algorithms:util",
        "@com_google_cc_differential_privacy//proto:util-lib",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_google_cc_differential_privacy//algorithms:algorithm",
        "@com_google_cc_differential_privacy//algorithms:numerical-mechanisms",
        "@com_google_absl//absl/random:distributions",
        "@com_google_cc_differential_privacy//base:status",
        "@com_google_cc_differential_privacy//testing:sequence",
    ],
)

cc_library(
    name = "create_samples_mean",
    srcs = ["create_samples_mean.cc"],
    hdrs = ["create_samples_mean.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_cc_differential_privacy//algorithms:bounded-mean",
        "@com_google_cc_differential_privacy//algorithms:bounded-sum",
        "@com_google_cc_differential_privacy//algorithms:count",
        "@com_google_cc_differential_privacy//algorithms:order-statistics",
        "@com_google_cc_differential_privacy//base:statusor",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_cc_differential_privacy//algorithms:util",
        "@com_google_cc_differential_privacy//proto:util-lib",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_google_cc_differential_privacy//algorithms:algorithm",
        "@com_google_cc_differential_privacy//algorithms:numerical-mechanisms",
        "@com_google_absl//absl/random:distributions",
        "@com_google_cc_differential_privacy//base:status",
        "@com_google_cc_differential_privacy//testing:sequence",
    ],
)

cc_library(
    name = "insufficient_noise_algorithms",
    srcs = ["insufficient_noise_algorithms.cc"],
    hdrs = ["insufficient_noise_algorithms.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_cc_differential_privacy//algorithms:bounded-mean",
        "@com_google_cc_differential_privacy//algorithms:bounded-sum",
        "@com_google_cc_differential_privacy//algorithms:count",
        "@com_google_cc_differential_privacy//algorithms:order-statistics",
        "@com_google_cc_differential_privacy//base:statusor",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_cc_differential_privacy//algorithms:util",
        "@com_google_cc_differential_privacy//proto:util-lib",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_google_cc_differential_privacy//algorithms:algorithm",
        "@com_google_cc_differential_privacy//algorithms:numerical-mechanisms",
        "@com_google_absl//absl/random:distributions",
        "@com_google_cc_differential_privacy//base:status",
        "@com_google_cc_differential_privacy//testing:sequence",
        "@com_google_cc_differential_privacy//testing:stochastic_tester",
    ],
)