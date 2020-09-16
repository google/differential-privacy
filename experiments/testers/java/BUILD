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

java_plugin(
    name = "autovalue_plugin",
    processor_class = "com.google.auto.value.processor.AutoValueProcessor",
    deps = [
        "@maven//:com_google_auto_value_auto_value"
    ],
)

java_library(
    name = "autovalue",
    exported_plugins = [
        ":autovalue_plugin",
    ],
    exports = [
        "@maven//:com_google_auto_value_auto_value",
    ],
)

java_binary(
    name = "part2",
    srcs = ["StatisticalTester.java"],
    main_class = "StatisticalTester",
    deps = [
                ":autovalue",
                ":statistical_tester_count",
                ":statistical_tester_sum",
                ":statistical_tester_mean",
                ":statistical_utils",
               "@com_google_java_differential_privacy//main/com/google/privacy/differentialprivacy",
               "@com_google_java_differential_privacy//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
                "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
                "@com_google_protobuf//:protobuf_java",
                "@maven//:com_google_guava_guava",
                "@maven//:com_google_truth_truth",
                "@maven//:junit_junit",
               "@maven//:com_google_auto_value_auto_value_annotations",
    ],
)

java_library(
    name = "statistical_tester_count",
    srcs = ["StatisticalTesterCount.java"],
    deps = [
                ":autovalue",
                ":statistical_utils",
               "@com_google_java_differential_privacy//main/com/google/privacy/differentialprivacy",
               "@com_google_java_differential_privacy//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
                "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
                "@com_google_protobuf//:protobuf_java",
                "@maven//:com_google_guava_guava",
                "@maven//:com_google_truth_truth",
                "@maven//:junit_junit",
               "@maven//:com_google_auto_value_auto_value_annotations",
    ],
)

java_library(
    name = "statistical_tester_sum",
    srcs = ["StatisticalTesterSum.java"],
    deps = [
                ":autovalue",
                ":statistical_utils",
               "@com_google_java_differential_privacy//main/com/google/privacy/differentialprivacy",
               "@com_google_java_differential_privacy//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
                "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
                "@com_google_protobuf//:protobuf_java",
                "@maven//:com_google_guava_guava",
                "@maven//:com_google_truth_truth",
                "@maven//:junit_junit",
               "@maven//:com_google_auto_value_auto_value_annotations",
    ],
)

java_library(
    name = "statistical_tester_mean",
    srcs = ["StatisticalTesterMean.java"],
    deps = [
                ":autovalue",
                ":statistical_utils",
               "@com_google_java_differential_privacy//main/com/google/privacy/differentialprivacy",
               "@com_google_java_differential_privacy//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
                "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
                "@com_google_protobuf//:protobuf_java",
                "@maven//:com_google_guava_guava",
                "@maven//:com_google_truth_truth",
                "@maven//:junit_junit",
               "@maven//:com_google_auto_value_auto_value_annotations",
    ],
)

java_library(
    name = "statistical_utils",
    srcs = ["StatisticalUtils.java"],
    deps = [
                ":autovalue",
               "@com_google_java_differential_privacy//main/com/google/privacy/differentialprivacy",
               "@com_google_java_differential_privacy//main/com/google/privacy/differentialprivacy/testing:statistical_utils",
                "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
                "@com_google_protobuf//:protobuf_java",
                "@maven//:com_google_guava_guava",
                "@maven//:com_google_truth_truth",
                "@maven//:junit_junit",
               "@maven//:com_google_auto_value_auto_value_annotations",
    ],
)