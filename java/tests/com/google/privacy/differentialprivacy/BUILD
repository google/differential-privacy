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

load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

java_plugin(
    name = "autovalue_plugin",
    processor_class = "com.google.auto.value.processor.AutoValueProcessor",
    deps = [
        "@maven//:com_google_auto_value_auto_value",
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

java_test(
    name = "AllTests",
    timeout = "long",
    srcs = glob(
        [
            "*.java",
        ],
    ),
    shard_count = 16,
    test_class = "com.google.privacy.differentialprivacy.AllTests",
    deps = [
        ":autovalue",
        "//main/com/google/privacy/differentialprivacy",
        "@com_google_differential_privacy//proto:summary_java_proto",
        "@maven//:com_google_auto_value_auto_value",
        "@maven//:com_google_auto_value_auto_value_annotations",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_protobuf_protobuf_java",
        "@maven//:com_google_testparameterinjector_test_parameter_injector",
        "@maven//:com_google_truth_extensions_truth_java8_extension",
        "@maven//:com_google_truth_truth",
        "@maven//:junit_junit",
        "@maven//:org_mockito_mockito_core",
    ],
)
