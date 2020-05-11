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
               "@com_google_differential_privacy//differential_privacy/proto:summary_java_proto",
               "@maven//:com_google_guava_guava",
               "@maven//:com_google_truth_truth",
               "@maven//:junit_junit",
               "@maven//:org_mockito_mockito_core",
               "@maven//:com_google_protobuf_protobuf_java",
    ],
)
