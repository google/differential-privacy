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

load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

java_test(
    name = "laplace_closeness_tests",
    size = "large",
    srcs = ["LaplaceClosenessTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:laplace_closeness_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in laplace_closeness_test_cases.textproto
    shard_count = 19,
    test_class = "com.google.privacy.differentialprivacy.statistical.LaplaceClosenessTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy",
        "//main/com/google/privacy/differentialprivacy:test_noise_factory",
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@com_google_protobuf//:protobuf_java",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "gaussian_closeness_tests",
    size = "large",
    srcs = ["GaussianClosenessTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:gaussian_closeness_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in gaussian_closeness_test_cases.textproto
    shard_count = 23,
    test_class = "com.google.privacy.differentialprivacy.statistical.GaussianClosenessTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy",
        "//main/com/google/privacy/differentialprivacy:test_noise_factory",
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@com_google_protobuf//:protobuf_java",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "count_dp_tests",
    size = "large",
    srcs = ["CountDpTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:count_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in count_dp_test_cases.textproto
    shard_count = 36,
    test_class = "com.google.privacy.differentialprivacy.statistical.CountDpTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy",
        "//main/com/google/privacy/differentialprivacy:test_noise_factory",
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@com_google_protobuf//:protobuf_java",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "bounded_mean_dp_tests",
    size = "large",
    srcs = ["BoundedMeanDpTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:bounded_mean_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in bounded_mean_dp_test_cases.textproto
    shard_count = 47,
    test_class = "com.google.privacy.differentialprivacy.statistical.BoundedMeanDpTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy",
        "//main/com/google/privacy/differentialprivacy:test_noise_factory",
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@com_google_protobuf//:protobuf_java",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "bounded_sum_dp_tests",
    size = "large",
    srcs = ["BoundedSumDpTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:bounded_sum_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in bounded_sum_dp_test_cases.textproto
    shard_count = 38,
    test_class = "com.google.privacy.differentialprivacy.statistical.BoundedSumDpTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy",
        "//main/com/google/privacy/differentialprivacy:test_noise_factory",
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@com_google_protobuf//:protobuf_java",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "long_bounded_sum_dp_tests",
    size = "large",
    srcs = ["LongBoundedSumDpTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:long_bounded_sum_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in long_bounded_sum_dp_test_cases.textproto
    shard_count = 38,
    test_class = "com.google.privacy.differentialprivacy.statistical.LongBoundedSumDpTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy",
        "//main/com/google/privacy/differentialprivacy:test_noise_factory",
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@com_google_protobuf//:protobuf_java",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "bounded_variance_dp_tests",
    size = "large",
    srcs = ["BoundedVarianceDpTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:bounded_stdv_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in bounded_stdv_dp_test_cases.textproto
    shard_count = 42,
    test_class = "com.google.privacy.differentialprivacy.statistical.BoundedVarianceDpTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy",
        "//main/com/google/privacy/differentialprivacy:test_noise_factory",
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@com_google_protobuf//:protobuf_java",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "approximate_bounds_dp_tests",
    size = "large",
    srcs = ["ApproximateBoundsDpTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:approximate_bounds_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in approximate_bounds_dp_test_cases.textproto
    shard_count = 14,
    test_class = "com.google.privacy.differentialprivacy.statistical.ApproximateBoundsDpTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@com_google_protobuf//:protobuf_java",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)
