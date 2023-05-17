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

proto_library(
    name = "statistical_tests_proto",
    srcs = ["statistical_tests.proto"],
)

java_proto_library(
    name = "statistical_tests_java_proto",
    visibility = [
        "//visibility:public",
    ],
    deps = [":statistical_tests_proto"],
)

exports_files([
    "approximate_bounds_dp_test_cases.textproto",
    "bounded_mean_dp_test_cases.textproto",
    "bounded_quantiles_dp_test_cases.textproto",
    "bounded_stdv_dp_test_cases.textproto",
    "bounded_sum_dp_test_cases.textproto",
    "count_dp_test_cases.textproto",
    "gaussian_closeness_test_cases.textproto",
    "laplace_closeness_test_cases.textproto",
    "laplace_dp_test_cases.textproto",
    "long_bounded_sum_dp_test_cases.textproto",
])

cc_proto_library(
    name = "statistical_tests_cc_proto",
    visibility = [
        "//visibility:public",
    ],
    deps = [":statistical_tests_proto"],
)
