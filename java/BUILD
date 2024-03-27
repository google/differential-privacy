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

load("@bazel_common//tools/maven:pom_file.bzl", "pom_file")
load("@rules_jvm_external//:defs.bzl", "java_export")

# Update the following version for packaging of a new release.
_DP_LIB_VERSION = "3.0.0"

exports_files([
    "dp_java_deps.bzl",
    "dp_java_deps_preload.bzl",
])

pom_file(
    name = "differentialprivacy_pom",
    substitutions = {"DP_LIB_VERSION": _DP_LIB_VERSION},
    targets = [
        "//main/com/google/privacy/differentialprivacy",
    ],
    template_file = "pom.template",
)

java_export(
    name = "differentialprivacy-export",
    maven_coordinates = "com.google.privacy.differentialprivacy:differentialprivacy:%s" % _DP_LIB_VERSION,
    pom_template = ":differentialprivacy_pom",
    runtime_deps = [
        "//main/com/google/privacy/differentialprivacy",
    ],
)
