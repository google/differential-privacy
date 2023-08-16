#
# Copyright 2019 Google LLC
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

# The files in this directory offer an example of how to use the C++
# Differential Privacy library.

licenses(["notice"])

cc_binary(
    name = "report_the_carrots",
    srcs = ["report_the_carrots.cc"],
    deps = [
        ":animals_and_carrots",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_cc_differential_privacy//proto:util-lib",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_differential_privacy//proto:data_cc_proto",
    ],
)

cc_library(
    name = "animals_and_carrots",
    srcs = ["animals_and_carrots.cc"],
    hdrs = ["animals_and_carrots.h"],
    data = ["animals_and_carrots.csv"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_cc_differential_privacy//algorithms:bounded-mean",
        "@com_google_cc_differential_privacy//algorithms:bounded-sum",
        "@com_google_cc_differential_privacy//algorithms:count",
        "@com_google_cc_differential_privacy//algorithms:quantiles",
        "@com_google_cc_differential_privacy//base:status_macros",
        "@com_google_differential_privacy//proto:data_cc_proto",
    ],
)

cc_test(
    name = "animals_and_carrots_test",
    srcs = ["animals_and_carrots_test.cc"],
    deps = [
        ":animals_and_carrots",
        "@com_google_absl//absl/status",
        "@com_google_cc_differential_privacy//base/testing:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
