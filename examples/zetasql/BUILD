#
# Copyright 2021 Google LLC
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

# The files in this directory offer an example of how to use the C++
# Differential Privacy library.

cc_binary(
    name = "execute_query",
    srcs = [
        "execute_query.cc",
    ],
    visibility = ["//visibility:public"],
    # TODO: Replace @com_google_cc_differential_privacy//base:status with
    # @com_google_cc_differential_privacy//base:status_macros once ZetaSQL
    # depends on the newer DP lib version.
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_cc_differential_privacy//base:status",
        "@com_google_protobuf//:protobuf",
        "@com_google_zetasql//zetasql/public:analyzer_options",
        "@com_google_zetasql//zetasql/public:builtin_function_options",
        "@com_google_zetasql//zetasql/public:catalog",
        "@com_google_zetasql//zetasql/public:language_options",
        "@com_google_zetasql//zetasql/public:options_cc_proto",
        "@com_google_zetasql//zetasql/public:simple_catalog",
        "@com_google_zetasql//zetasql/public:type_cc_proto",
        "@com_google_zetasql//zetasql/public:value",
        "@com_google_zetasql//zetasql/resolved_ast",
        "@com_google_zetasql//zetasql/resolved_ast:resolved_node_kind_cc_proto",
        "@com_google_zetasql//zetasql/tools/execute_query:execute_query_tool",
    ],
)
