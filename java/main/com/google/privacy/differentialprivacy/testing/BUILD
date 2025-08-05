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

load("@rules_java//java:defs.bzl", "java_library")
load("@rules_jvm_external//:defs.bzl", "artifact")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

java_library(
    name = "reference_noise",
    testonly = 1,
    srcs = ["ReferenceNoiseUtil.java"],
    deps = [
        "@com_google_differential_privacy//proto/testing:statistical_tests_java_proto",
        artifact(
            "com.google.guava:guava",
            repository_name = "differential_privacy_java_deps",
        ),
        artifact(
            "com.google.protobuf:protobuf-java",
            repository_name = "differential_privacy_java_deps",
        ),
        artifact(
            "org.apache.commons:commons-math3",
            repository_name = "differential_privacy_java_deps",
        ),
    ],
)

java_library(
    name = "statistical_utils",
    testonly = 1,
    srcs = [
        "StatisticalTestsUtil.java",
        "VotingUtil.java",
    ],
    deps = [
        artifact(
            "com.google.guava:guava",
            repository_name = "differential_privacy_java_deps",
        ),
    ],
)
