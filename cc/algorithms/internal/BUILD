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
    name = "count-tree",
    srcs = ["count-tree.cc"],
    hdrs = ["count-tree.h"],
    deps = [
        "//algorithms:algorithm",
        "//algorithms:bounded-algorithm",
        "//algorithms:numerical-mechanisms",
        "//proto:util-lib",
        "@com_google_absl//absl/status",
        "@com_google_cc_differential_privacy//base:status",
        "@com_google_cc_differential_privacy//base:statusor",
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
        "@com_google_googletest//:gtest_main",
    ],
)
