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
    copts = ["-Wno-sign-compare"],
    deps = [
        "//differential_privacy/base:logging",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random:distributions",
    ],
)

cc_test(
    name = "sequence_test",
    size = "small",
    srcs = ["sequence_test.cc"],
    copts = ["-Wno-sign-compare"],
    deps = [
        ":sequence",
        "@com_google_googletest//:gtest_main",
        "//differential_privacy/algorithms:util",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "stochastic_tester",
    hdrs = ["stochastic_tester.h"],
    copts = ["-Wno-sign-compare"],
    deps = [
        ":density_estimation",
        ":sequence",
        "//differential_privacy/base:logging",
        "//differential_privacy/algorithms:algorithm",
        "//differential_privacy/algorithms:util",
        "//differential_privacy/proto:util-lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "density_estimation",
    hdrs = ["density_estimation.h"],
    copts = ["-Wno-sign-compare"],
    deps = [
        "//differential_privacy/base:status",
        "//differential_privacy/base:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "stochastic_tester_test",
    size = "small",
    timeout = "long",
    srcs = ["stochastic_tester_test.cc"],
    copts = ["-Wno-sign-compare"],
    shard_count = 10,
    deps = [
        ":sequence",
        ":stochastic_tester",
        "@com_google_googletest//:gtest_main",
        "//differential_privacy/algorithms:algorithm",
        "//differential_privacy/algorithms:bounded-sum",
        "//differential_privacy/algorithms:count",
        "//differential_privacy/algorithms:numerical-mechanisms",
        "//differential_privacy/algorithms:numerical-mechanisms-testing",
        "//differential_privacy/algorithms:util",
        "@com_google_absl//absl/random:distributions",
    ],
)

cc_test(
    name = "density_estimation_test",
    srcs = ["density_estimation_test.cc"],
    copts = ["-Wno-sign-compare"],
    deps = [
        ":density_estimation",
        "//differential_privacy/base:status",
        "//differential_privacy/base/testing:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
