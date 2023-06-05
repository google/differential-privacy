#
# Copyright 2021 Google LLC
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

# Differential Privacy Accounting related proto definitions.

package(
    default_visibility = ["//visibility:public"],
)

proto_library(
    name = "privacy_loss_distribution_proto",
    srcs = ["privacy-loss-distribution.proto"],
)

cc_proto_library(
    name = "privacy_loss_distribution_cc_proto",
    deps = [":privacy_loss_distribution_proto"],
)
