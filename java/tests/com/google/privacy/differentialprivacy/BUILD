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

package(default_visibility = [
    "//visibility:public"
])

java_test(
    name = "AllTests",
    srcs = [
        "AllTests.java",
        "BoundedMeanBuilderTest.java",
        "BoundedMeanTest.java",
        "BoundedSumBuilderTest.java",
        "BoundedSumTest.java",
        "CountBuilderTest.java",
        "CountTest.java",
        "GaussianNoiseTest.java",
        "LaplaceNoiseTest.java",
    ],
    test_class = "com.google.privacy.differentialprivacy.AllTests",
    deps = [
               "//main/com/google/privacy/differentialprivacy",
               "@com_google_differential_privacy//proto:summary_java_proto",
               "@maven//:com_google_guava_guava",
               "@maven//:com_google_truth_truth",
               "@maven//:junit_junit",
               "@maven//:org_mockito_mockito_core",
               "@maven//:com_google_protobuf_protobuf_java",
    ],
)

java_test(
    name = "laplace_closeness_tests",
    size = "enormous",
    srcs = ["LaplaceClosenessTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:laplace_closeness_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in laplace_closeness_test_cases.textproto
    shard_count = 19,
    test_class = "com.google.privacy.differentialprivacy.LaplaceClosenessTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy",
        "//main/com/google/privacy/differentialprivacy/testing",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@com_google_protobuf//:protobuf_java",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "count_dp_tests",
    size = "enormous",
    srcs = ["CountDpTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:count_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in count_dp_test_cases.textproto
    shard_count = 36,
    test_class = "com.google.privacy.differentialprivacy.CountDpTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy",
        "//main/com/google/privacy/differentialprivacy/testing",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@com_google_protobuf//:protobuf_java",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)

java_test(
    name = "bounded_mean_dp_tests",
    size = "enormous",
    srcs = ["BoundedMeanDpTest.java"],
    resources = [
        "@com_google_differential_privacy//proto/testing:bounded_mean_dp_test_cases.textproto",
    ],
    # The shard count must match the number of test cases defined in bounded_mean_dp_test_cases.textproto
    shard_count = 47,
    test_class = "com.google.privacy.differentialprivacy.BoundedMeanDpTest",
    deps = [
        "//main/com/google/privacy/differentialprivacy",
        "//main/com/google/privacy/differentialprivacy/testing",
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        "@com_google_protobuf//:protobuf_java",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
    ],
)
