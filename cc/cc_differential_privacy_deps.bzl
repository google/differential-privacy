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

""" Declares dependencies of the differential privacy library """

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def cc_differential_privacy_deps():
    """ Macro to include the differential privacy library's critical dependencies in a WORKSPACE.

    """

    # Abseil
    http_archive(
        name = "com_google_absl",
        url = "https://github.com/abseil/abseil-cpp/archive/20210324.0.tar.gz",
        sha256 = "dd7db6815204c2a62a2160e32c55e97113b0a0178b2f090d6bab5ce36111db4b",
        strip_prefix = "abseil-cpp-20210324.0",
    )

    # Common bazel rules
    http_archive(
        name = "bazel_skylib",
        url = "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
        sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
    )

    # GoogleTest/GoogleMock framework. Used by most unit-tests.
    http_archive(
        name = "com_google_googletest",
        # Commit date: 2021-01-27
        urls = ["https://github.com/google/googletest/archive/df7fee587d442b372ef43bd66c6a2f5c9af8c5eb.tar.gz"],
        strip_prefix = "googletest-df7fee587d442b372ef43bd66c6a2f5c9af8c5eb",
        sha256 = "4a6673769eefb799bc0db0d7cf48ad9cf22dc5e55106f54bf9f4e43a40f425ac",
    )

    # Benchmarks for testing.
    http_archive(
        name = "com_google_benchmark",
        url = "https://github.com/google/benchmark/archive/refs/tags/v1.5.2.tar.gz",
        sha256 = "dccbdab796baa1043f04982147e67bb6e118fe610da2c65f88912d73987e700c",
        strip_prefix = "benchmark-1.5.2",
    )

    # BoringSSL for cryptographic PRNG
    git_repository(
        name = "boringssl",
        # Commit date: 2021-08-30
        # Note for updating: we need to use a commit from the main-with-bazel branch.
        commit = "e410957010208f95586d432031397bffef678ba4",
        remote = "https://boringssl.googlesource.com/boringssl",
        shallow_since = "1630347097 +0000",
    )

    # Supports `./configure && make` style packages to become dependencies.
    http_archive(
        name = "rules_foreign_cc",
        strip_prefix = "rules_foreign_cc-0.2.0",
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.2.0.zip",
        sha256 = "e60cfd0a8426fa4f5fd2156e768493ca62b87d125cb35e94c44e79a3f0d8635f",
    )

    # Postgres depends on rules_foreign_cc. Use postgres 11.
    http_archive(
        name = "postgres",
        url = "https://github.com/postgres/postgres/archive/REL_11_7.tar.gz",
        build_file = "@com_google_differential_privacy//cc/postgres:postgres.BUILD",
        strip_prefix = "postgres-REL_11_7",
        sha256 = "8c427e10a5f8b6be76353e83c7cf0171ac0e85308d352b8c129612002bb342eb",
    )
