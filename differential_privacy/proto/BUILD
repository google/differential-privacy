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
#

# Differential Privacy related proto definitions.

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "data-proto",
    srcs = ["data.proto"],
    deps = [
        "//differential_privacy/algorithms:confidence-interval_proto",
    ],
)

cc_proto_library(
    name = "data_cc_proto",
    deps = [":data-proto"],
)

java_proto_library(
    name = "data_java_proto",
    deps = [":data-proto"],
)

proto_library(
    name = "summary-proto",
    srcs = ["summary.proto"],
    deps = [
        ":data-proto",
        "@com_google_protobuf//:any_proto",
    ],
)

cc_proto_library(
    name = "summary_cc_proto",
    deps = [":summary-proto"],
)

java_proto_library(
    name = "summary_java_proto",
    deps = [":summary-proto"],
)

cc_library(
    name = "util-lib",
    hdrs = ["util.h"],
    copts = ["-Wno-sign-compare"],
    deps = [":data_cc_proto"],
)

cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    copts = ["-Wno-sign-compare"],
    deps = [
        ":util-lib",
        "@com_google_googletest//:gtest_main",
    ],
)
