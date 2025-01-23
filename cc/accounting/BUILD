#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package_group(
    name = "accounting",
    packages = [
        "//accounting/...",
    ],
)

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "pld",
    srcs = [
        "privacy_loss_distribution.cc",
        "privacy_loss_mechanism.cc",
    ],
    hdrs = [
        "privacy_loss_distribution.h",
        "privacy_loss_mechanism.h",
    ],
    deps = [
        ":convolution",
        "//accounting/common",
        "@boost//:math",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_cc_differential_privacy//base:status_macros",
        "@com_google_differential_privacy//proto/accounting:privacy_loss_distribution_cc_proto",
    ],
)

cc_library(
    name = "convolution",
    srcs = [
        "convolution.cc",
    ],
    hdrs = [
        "convolution.h",
    ],
    deps = [
        ":kiss_fft_wrapper",
        "//accounting/common",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_test(
    name = "pld_test",
    srcs = [
        "privacy_loss_distribution_test.cc",
        "privacy_loss_mechanism_test.cc",
    ],
    deps = [
        ":pld",
        "//accounting/common",
        "//accounting/common:test_util",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "@com_google_differential_privacy//proto/accounting:privacy_loss_distribution_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "convolution_test",
    srcs = [
        "convolution_test.cc",
    ],
    tags = ["not_run:arm"],
    deps = [
        ":convolution",
        "//accounting/common:test_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "example",
    srcs = ["example.cc"],
    deps = [
        ":pld",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "kiss_fft_wrapper",
    hdrs = ["kiss_fft_wrapper.h"],
    deps = [
        "//accounting/kissfft",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_test(
    name = "kiss_fft_wrapper_test",
    srcs = ["kiss_fft_wrapper_test.cc"],
    deps = [
        ":kiss_fft_wrapper",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "accountant",
    srcs = ["accountant.cc"],
    hdrs = ["accountant.h"],
    deps = [
        ":pld",
        "//accounting/common",
        "@boost//:math",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "accountant_test",
    srcs = ["accountant_test.cc"],
    deps = [
        ":accountant",
        ":pld",
        "//accounting/common",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)
