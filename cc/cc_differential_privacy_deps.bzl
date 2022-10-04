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

def cc_differential_privacy_deps():
    """ Macro to include the differential privacy library's critical dependencies in a WORKSPACE.

    """

    # Abseil
    http_archive(
        name = "com_google_absl",
        url = "https://github.com/abseil/abseil-cpp/archive/20220623.0.tar.gz",
        sha256 = "4208129b49006089ba1d6710845a45e31c59b0ab6bff9e5788a87f55c5abd602",
        strip_prefix = "abseil-cpp-20220623.0",
    )

    # Common bazel rules.  Also required for Abseil.
    http_archive(
        name = "bazel_skylib",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
        ],
        sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",
    )

    # GoogleTest/GoogleMock framework. Used by most unit-tests.
    http_archive(
        name = "com_google_googletest",
        url = "https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz",
        strip_prefix = "googletest-release-1.12.1",
        sha256 = "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2",
    )

    # Re2 is a requirement for GoogleTest
    #
    # Note this must use a commit from the `abseil` branch of the RE2 project.
    # https://github.com/google/re2/tree/abseil
    http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "0a890c2aa0bb05b2ce906a15efb520d0f5ad4c7d37b8db959c43772802991887",
        strip_prefix = "re2-a427f10b9fb4622dd6d8643032600aa1b50fbd12",
        # Commit date: 2022-06-09
        url = "https://github.com/google/re2/archive/a427f10b9fb4622dd6d8643032600aa1b50fbd12.zip",
    )

    # Benchmarks for testing.
    http_archive(
        name = "com_google_benchmark",
        url = "https://github.com/google/benchmark/archive/refs/tags/v1.7.0.tar.gz",
        sha256 = "3aff99169fa8bdee356eaa1f691e835a6e57b1efeadb8a0f9f228531158246ac",
        strip_prefix = "benchmark-1.7.0",
    )

    # BoringSSL for cryptographic PRNG
    http_archive(
        name = "boringssl",
        # Commit date: 2022-09-23
        # Note for updating: we need to use a commit from the `master-with-bazel` branch.
        url = "https://github.com/google/boringssl/archive/3a3d0b5c7fddeea312b5ce032d9b84a2be399b32.tar.gz",
        sha256 = "be8231e5f3b127d83eb156354dfa28c110e3c616c11ae119067c8184ef7a257f",
        strip_prefix = "boringssl-3a3d0b5c7fddeea312b5ce032d9b84a2be399b32",
    )

    # Supports `./configure && make` style packages to become dependencies.
    http_archive(
        name = "rules_foreign_cc",
        strip_prefix = "rules_foreign_cc-0.9.0",
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.9.0.tar.gz",
        sha256 = "2a4d07cd64b0719b39a7c12218a3e507672b82a97b98c6a89d38565894cf7c51",
    )

    # Postgres depends on rules_foreign_cc. Use postgres 12.
    http_archive(
        name = "postgres",
        url = "https://github.com/postgres/postgres/archive/REL_12_9.tar.gz",
        build_file = "@com_google_differential_privacy//cc/postgres:postgres.BUILD",
        strip_prefix = "postgres-REL_12_9",
        sha256 = "64f6da47aab9ac65d07b31abd40445b4a0413d4265d25b82ed738abad8a98349",
    )
