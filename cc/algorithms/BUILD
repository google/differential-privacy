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
    default_visibility = [
        "//algorithms:__subpackages__",
    ],
)

cc_library(
    name = "algorithm",
    hdrs = ["algorithm.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":numerical-mechanisms",
        ":util",
        "//proto:util-lib",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_cc_differential_privacy//base:status_macros",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_differential_privacy//proto:summary_cc_proto",
    ],
)

cc_test(
    name = "algorithm_test",
    size = "small",
    srcs = ["algorithm_test.cc"],
    deps = [
        ":algorithm",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "algorithm-stochastic-dp_test",
    timeout = "eternal",
    srcs = ["algorithm-stochastic-dp_test.cc"],
    shard_count = 16,
    deps = [
        ":approx-bounds",
        ":bounded-mean",
        ":bounded-standard-deviation",
        ":bounded-sum",
        ":bounded-variance",
        ":count",
        ":numerical-mechanisms-testing",
        ":order-statistics",
        ":util",
        "//testing:sequence",
        "//testing:stochastic_tester",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "binary-search",
    hdrs = ["binary-search.h"],
    deps = [
        ":algorithm",
        ":numerical-mechanisms",
        "//base:percentile",
        "//proto:util-lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_cc_differential_privacy//base:status_macros",
    ],
)

cc_test(
    name = "binary-search_test",
    size = "small",
    srcs = ["binary-search_test.cc"],
    deps = [
        ":algorithm",
        ":binary-search",
        ":numerical-mechanisms",
        ":numerical-mechanisms-testing",
        ":util",
        "//base/testing:status_matchers",
        "//proto:util-lib",
        "@com_google_absl//absl/random:distributions",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bounds-provider",
    hdrs = ["bounds-provider.h"],
    visibility = [
        "//algorithms/approx_count_distinct:__subpackages__",
    ],
    deps = [
        "//algorithms/internal:clamped-calculation-without-bounds",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_differential_privacy//proto:summary_cc_proto",
    ],
)

cc_library(
    name = "order-statistics",
    hdrs = ["order-statistics.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":algorithm",
        ":binary-search",
        ":bounded-algorithm",
        ":numerical-mechanisms",
        "//base:percentile",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "order-statistics_test",
    size = "medium",
    srcs = ["order-statistics_test.cc"],
    deps = [
        ":numerical-mechanisms-testing",
        ":order-statistics",
        ":util",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/random:distributions",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bounded-sum",
    hdrs = ["bounded-sum.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":algorithm",
        ":approx-bounds",
        ":approx-bounds-as-bounds-provider",
        ":bounds-provider",
        ":numerical-mechanisms",
        ":util",
        "//algorithms/internal:clamped-calculation-without-bounds",
        "//proto:util-lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_cc_differential_privacy//base:status_macros",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_differential_privacy//proto:summary_cc_proto",
    ],
)

cc_test(
    name = "bounded-sum_test",
    size = "small",
    srcs = ["bounded-sum_test.cc"],
    deps = [
        ":approx-bounds",
        ":approx-bounds-as-bounds-provider",
        ":bounded-sum",
        ":numerical-mechanisms",
        ":numerical-mechanisms-testing",
        "//base/testing:proto_matchers",
        "//base/testing:status_matchers",
        "//proto:util-lib",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bounded-mean",
    hdrs = ["bounded-mean.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":algorithm",
        ":approx-bounds",
        ":approx-bounds-as-bounds-provider",
        ":bounds-provider",
        ":numerical-mechanisms",
        ":util",
        "//algorithms/internal:bounded-mean-ci",
        "//algorithms/internal:clamped-calculation-without-bounds",
        "//proto:util-lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_cc_differential_privacy//base:status_macros",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_differential_privacy//proto:summary_cc_proto",
    ],
)

cc_test(
    name = "bounded-mean_test",
    size = "small",
    timeout = "moderate",
    srcs = ["bounded-mean_test.cc"],
    deps = [
        ":approx-bounds",
        ":approx-bounds-as-bounds-provider",
        ":bounded-mean",
        ":bounds-provider",
        ":numerical-mechanisms",
        ":numerical-mechanisms-testing",
        "//base/testing:proto_matchers",
        "//base/testing:status_matchers",
        "//proto:util-lib",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_differential_privacy//proto:summary_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "bounded-mean-ci-statistical_test",
    size = "medium",
    timeout = "long",
    srcs = ["bounded-mean-ci-statistical_test.cc"],
    deps = [
        ":approx-bounds",
        ":bounded-mean",
        ":numerical-mechanisms",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bounded-variance",
    hdrs = ["bounded-variance.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":algorithm",
        ":approx-bounds",
        ":approx-bounds-as-bounds-provider",
        ":bounds-provider",
        ":numerical-mechanisms",
        ":util",
        "//algorithms/internal:clamped-calculation-without-bounds",
        "//proto:util-lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_cc_differential_privacy//base:status_macros",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_differential_privacy//proto:summary_cc_proto",
    ],
)

cc_test(
    name = "bounded-variance_test",
    size = "small",
    srcs = ["bounded-variance_test.cc"],
    deps = [
        ":approx-bounds",
        ":approx-bounds-as-bounds-provider",
        ":bounded-variance",
        ":numerical-mechanisms",
        ":numerical-mechanisms-testing",
        "//base/testing:proto_matchers",
        "//base/testing:status_matchers",
        "//proto:util-lib",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_differential_privacy//proto:summary_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bounded-standard-deviation",
    hdrs = ["bounded-standard-deviation.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":algorithm",
        ":approx-bounds",
        ":bounded-variance",
        ":numerical-mechanisms",
        "//proto:util-lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_cc_differential_privacy//base:status_macros",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_differential_privacy//proto:summary_cc_proto",
    ],
)

cc_test(
    name = "bounded-standard-deviation_test",
    size = "small",
    srcs = ["bounded-standard-deviation_test.cc"],
    deps = [
        ":approx-bounds",
        ":bounded-standard-deviation",
        ":distributions",
        ":numerical-mechanisms-testing",
        "//base/testing:proto_matchers",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random:distributions",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "count",
    hdrs = ["count.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":algorithm",
        ":numerical-mechanisms",
        ":util",
        "//proto:util-lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_cc_differential_privacy//base:status_macros",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_differential_privacy//proto:summary_cc_proto",
    ],
)

cc_test(
    name = "count_test",
    size = "small",
    srcs = ["count_test.cc"],
    deps = [
        ":count",
        ":numerical-mechanisms",
        ":numerical-mechanisms-testing",
        "//base/testing:proto_matchers",
        "//base/testing:status_matchers",
        "//proto:util-lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_differential_privacy//proto:summary_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    visibility = [
        "//algorithms/internal:__pkg__",
        "//testing:__pkg__",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_cc_differential_privacy//base:status_macros",
    ],
)

cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    deps = [
        ":distributions",
        ":numerical-mechanisms-testing",
        ":util",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "distributions",
    srcs = ["distributions.cc"],
    hdrs = ["distributions.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":rand",
        ":util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_cc_differential_privacy//base:status_macros",
        "@com_google_differential_privacy//third_party/cephes:inverse_gaussian_cdf",
    ],
)

cc_test(
    name = "distributions_test",
    size = "large",
    srcs = ["distributions_test.cc"],
    shard_count = 4,
    deps = [
        ":distributions",
        ":numerical-mechanisms-testing",
        ":util",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "numerical-mechanisms",
    srcs = ["numerical-mechanisms.cc"],
    hdrs = ["numerical-mechanisms.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":distributions",
        ":rand",
        ":util",
        "//algorithms/internal:gaussian-stddev-calculator",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_cc_differential_privacy//base:status_macros",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_differential_privacy//proto:numerical_mechanism_cc_proto",
    ],
)

cc_test(
    name = "numerical-mechanisms_test",
    size = "large",
    srcs = ["numerical-mechanisms_test.cc"],
    deps = [
        ":distributions",
        ":numerical-mechanisms",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "numerical-mechanisms-testing",
    testonly = 1,
    hdrs = ["numerical-mechanisms-testing.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":distributions",
        ":numerical-mechanisms",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status:statusor",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "numerical-mechanisms-testing_test",
    size = "small",
    srcs = ["numerical-mechanisms-testing_test.cc"],
    deps = [
        ":numerical-mechanisms-testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "approx-bounds",
    hdrs = ["approx-bounds.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":algorithm",
        ":numerical-mechanisms",
        ":util",
        "//algorithms/internal:clamped-calculation-without-bounds",
        "//proto:util-lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_cc_differential_privacy//base:status_macros",
        "@com_google_differential_privacy//proto:data_cc_proto",
    ],
)

cc_test(
    name = "approx-bounds_test",
    size = "small",
    srcs = ["approx-bounds_test.cc"],
    deps = [
        ":approx-bounds",
        ":numerical-mechanisms",
        ":numerical-mechanisms-testing",
        "//algorithms/internal:clamped-calculation-without-bounds",
        "//base/testing:proto_matchers",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_differential_privacy//proto:data_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "approx-bounds-as-bounds-provider",
    hdrs = ["approx-bounds-as-bounds-provider.h"],
    visibility = [
        "//algorithms/approx_count_distinct:__subpackages__",
    ],
    deps = [
        ":approx-bounds",
        ":bounds-provider",
        "//algorithms/internal:clamped-calculation-without-bounds",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_cc_differential_privacy//base:status_macros",
    ],
)

cc_test(
    name = "approx-bounds-as-bounds-provider_test",
    srcs = ["approx-bounds-as-bounds-provider_test.cc"],
    deps = [
        ":approx-bounds",
        ":approx-bounds-as-bounds-provider",
        ":bounds-provider",
        "//algorithms/internal:clamped-calculation-without-bounds",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bounded-algorithm",
    hdrs = ["bounded-algorithm.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":algorithm",
        ":approx-bounds",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_cc_differential_privacy//base:status_macros",
    ],
)

cc_test(
    name = "bounded-algorithm_test",
    size = "small",
    srcs = ["bounded-algorithm_test.cc"],
    deps = [
        ":algorithm",
        ":approx-bounds",
        ":bounded-algorithm",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "rand",
    srcs = ["rand.cc"],
    hdrs = ["rand.h"],
    visibility = [
        "//testing:__pkg__",
    ],
    deps = [
        "@boringssl//:crypto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "rand_test",
    size = "small",
    srcs = ["rand_test.cc"],
    deps = [
        ":rand",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "distributions_benchmark_test",
    timeout = "eternal",
    srcs = ["distributions_benchmark_test.cc"],
    deps = [
        ":distributions",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_benchmark//:benchmark_main",
    ],
)

cc_library(
    name = "partition-selection",
    hdrs = ["partition-selection.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":distributions",
        ":numerical-mechanisms",
        ":rand",
        ":util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_cc_differential_privacy//base:status_macros",
    ],
)

cc_test(
    name = "partition-selection_test",
    srcs = ["partition-selection_test.cc"],
    shard_count = 2,
    deps = [
        ":numerical-mechanisms",
        ":partition-selection",
        ":partition-selection-testing",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "quantile-tree",
    hdrs = ["quantile-tree.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":numerical-mechanisms",
        "//algorithms/internal:count-tree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_cc_differential_privacy//base:status_macros",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_differential_privacy//proto:summary_cc_proto",
    ],
)

cc_test(
    name = "quantile-tree_test",
    timeout = "eternal",
    srcs = ["quantile-tree_test.cc"],
    shard_count = 16,
    deps = [
        ":numerical-mechanisms",
        ":numerical-mechanisms-testing",
        ":quantile-tree",
        "//base/testing:proto_matchers",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_differential_privacy//proto:confidence_interval_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "quantiles",
    hdrs = ["quantiles.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":algorithm",
        ":numerical-mechanisms",
        ":quantile-tree",
        ":util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_cc_differential_privacy//base:status_macros",
    ],
)

cc_test(
    name = "quantiles_test",
    srcs = ["quantiles_test.cc"],
    deps = [
        ":numerical-mechanisms-testing",
        ":quantiles",
        "//base/testing:proto_matchers",
        "//base/testing:status_matchers",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gaussian-dp-calculator",
    srcs = ["gaussian-dp-calculator.cc"],
    hdrs = ["gaussian-dp-calculator.h"],
    visibility = [
        # This target is not considered to be part of the public API (yet).
        "//algorithms:__subpackages__",
    ],
    deps = [
        "//algorithms/internal:gaussian-stddev-calculator",
    ],
)

cc_test(
    name = "gaussian-dp-calculator_test",
    srcs = ["gaussian-dp-calculator_test.cc"],
    deps = [
        ":gaussian-dp-calculator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "partition-selection-testing",
    testonly = 1,
    hdrs = ["partition-selection-testing.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":numerical-mechanisms",
        ":partition-selection",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)
