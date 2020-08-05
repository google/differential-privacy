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
#

# Differential Privacy related proto utilities.

package(default_visibility = ["//visibility:public"])
config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)
cc_library(
    name = "util-lib",
    hdrs = ["util.h"],
    copts = select({
        ":windows": [],
        "//conditions:default": ["-Wno-sign-compare"],
    }),
    deps = ["@com_google_differential_privacy//proto:data_cc_proto"],
)

cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    copts = select({
        ":windows": [],
        "//conditions:default": ["-Wno-sign-compare"],
    }),
    deps = [
        ":util-lib",
        "@com_google_googletest//:gtest_main",
    ],
)
