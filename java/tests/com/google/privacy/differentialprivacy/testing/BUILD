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
    name = "reference_noise_util_test",
    size = "small",
    srcs = [
        "ReferenceNoiseUtilTest.java",
    ],
    test_class = "com.google.privacy.differentialprivacy.testing.ReferenceNoiseUtilTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "statistical_utils_test",
    size = "small",
    srcs = [
        "StatisticalTestsUtilTest.java",
    ],
    test_class = "com.google.privacy.differentialprivacy.testing.StatisticalTestsUtilTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "laplace_closeness_test_cases_validity_tests",
    size = "enormous",
    srcs = ["LaplaceClosenessTestCasesValidityTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:laplace_closeness_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in laplace_closeness_test_cases.textproto
    shard_count = 19,
    test_class = "com.google.privacy.differentialprivacy.testing.LaplaceClosenessTestCasesValidityTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_protobuf_protobuf_java",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "gaussian_closeness_test_cases_validity_tests",
    size = "enormous",
    srcs = ["GaussianClosenessTestCasesValidityTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:gaussian_closeness_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in gaussian_closeness_test_cases.textproto
    shard_count = 23,
    test_class = "com.google.privacy.differentialprivacy.testing.GaussianClosenessTestCasesValidityTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_protobuf_protobuf_java",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "count_dp_test_cases_validity_tests",
    size = "enormous",
    srcs = ["CountDpTestCasesValidityTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:count_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in count_dp_test_cases.textproto
    shard_count = 36,
    test_class = "com.google.privacy.differentialprivacy.testing.CountDpTestCasesValidityTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_protobuf_protobuf_java",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "bounded_mean_dp_test_cases_validity_tests",
    size = "enormous",
    srcs = ["BoundedMeanDpTestCasesValidityTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:bounded_mean_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in bounded_mean_dp_test_cases.textproto
    shard_count = 47,
    test_class = "com.google.privacy.differentialprivacy.testing.BoundedMeanDpTestCasesValidityTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_protobuf_protobuf_java",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "bounded_sum_dp_test_cases_validity_tests",
    size = "enormous",
    srcs = ["BoundedSumDpTestCasesValidityTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:bounded_sum_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in bounded_sum_dp_test_cases.textproto
    shard_count = 38,
    test_class = "com.google.privacy.differentialprivacy.testing.BoundedSumDpTestCasesValidityTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_protobuf_protobuf_java",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "long_bounded_sum_dp_test_cases_validity_tests",
    size = "enormous",
    srcs = ["LongBoundedSumDpTestCasesValidityTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:long_bounded_sum_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in long_bounded_sum_dp_test_cases.textproto
    shard_count = 38,
    test_class = "com.google.privacy.differentialprivacy.testing.LongBoundedSumDpTestCasesValidityTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_protobuf_protobuf_java",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "bounded_stdv_dp_test_cases_validity_tests",
    size = "medium",
    srcs = ["BoundedStdvDpTestCasesValidityTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:bounded_stdv_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in bounded_stdv_dp_test_cases.textproto
    shard_count = 42,
    test_class = "com.google.privacy.differentialprivacy.testing.BoundedStdvDpTestCasesValidityTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy/testing:reference_noise",
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_protobuf_protobuf_java",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "approximate_bounds_dp_test_cases_validity_tests",
    size = "medium",
    srcs = ["ApproximateBoundsDpTestCasesValidityTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:approximate_bounds_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in approximate_bounds_dp_test_cases.textproto
    shard_count = 14,
    test_class = "com.google.privacy.differentialprivacy.testing.ApproximateBoundsDpTestCasesValidityTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_protobuf_protobuf_java",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)
