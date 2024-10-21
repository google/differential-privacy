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
#

load("@rules_java//java:defs.bzl", "java_binary")

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

java_binary(
    name = "Main",
    srcs = glob(["*.java"]),
    main_class = "com.google.privacy.differentialprivacy.example.Main",
    resources = glob(["*.csv"]),
    deps = [
        ":autovalue",
        "@com_google_java_differential_privacy//main/com/google/privacy/differentialprivacy",
        "@maven//:com_google_auto_value_auto_value_annotations",
        "@maven//:com_google_guava_guava",
    ],
)
