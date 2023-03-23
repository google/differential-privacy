#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])  # Apache v2.0

load("@rules_foreign_cc//foreign_cc:configure.bzl", "configure_make")

package(
    default_visibility = ["//visibility:public"],
)

# We need to label this for configure_make.
filegroup(
    name = "all",
    srcs = glob(["**"]),
)

config_setting(
    name = "darwin_build",
    values = {"cpu": "darwin"},
)

configure_make(
    name = "postgres",
    configure_command = "configure",
    env = select({
        ":darwin_build": {
            # Neccessary to create a fully static linked library.
            "CXXFLAGS": "-fPIC",
            "CFLAGS": "-fPIC",
            # OSX crosstools specific flags
            "LDFLAGS": "-undefined error",
            "AR": "",
        },
        "//conditions:default": {
            # Neccessary to create a fully static linked library.
            "CXXFLAGS": "-fPIC",
            "CFLAGS": "-fPIC",
        },
    }),
    lib_source = "@postgres//:all",
    out_headers_only = True,
)

cc_library(
    name = "pg_headers",
    hdrs = glob(["**/*.h"]),
    includes = [
        "postgres/include/internal",
        "postgres/include/server",
    ],
    deps = [
        "postgres",
    ],
)
