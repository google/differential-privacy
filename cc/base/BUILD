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
#

# Dependencies for C++ Differential Privacy Library OSS

licenses(["notice"])

package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "percentile",
    hdrs = ["percentile.h"],
    deps = [
        "//proto:util-lib",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:log_severity",
    ],
)

cc_library(
    name = "status",
    hdrs = [
        "status.h",
        "status_macros.h",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "statusor",
    hdrs = ["statusor.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "percentile_test",
    srcs = ["percentile_test.cc"],
    deps = [
        ":percentile",
        "@com_google_differential_privacy//proto:summary_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "status_macros_test",
    srcs = ["status_macros_test.cc"],
    deps = [
        ":status",
        ":statusor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
