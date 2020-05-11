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

def differential_privacy_deps():
    """ Macro to include the differential privacy library's critical dependencies in a WORKSPACE.

    """

    # Abseil
    http_archive(
        name = "com_google_absl",
        url = "https://github.com/abseil/abseil-cpp/archive/20200225.2.tar.gz",
        sha256 = "f41868f7a938605c92936230081175d1eae87f6ea2c248f41077c8f88316f111",
        strip_prefix = "abseil-cpp-20200225.2",
    )

    # Common bazel rules
    http_archive(
        name = "bazel_skylib",
        url = "https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz",
        sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    )

    # Protobuf
    http_archive(
        name = "com_google_protobuf",
        urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.11.4.tar.gz"],
        sha256 = "a79d19dcdf9139fa4b81206e318e33d245c4c9da1ffed21c87288ed4380426f9",
        strip_prefix = "protobuf-3.11.4",
    )

    # GoogleTest/GoogleMock framework. Used by most unit-tests.
    http_archive(
        name = "com_google_googletest",
        url = "https://github.com/google/googletest/archive/a53e931dcd00c2556ee181d832e699c9f3c29036.tar.gz",
        sha256 = "7850caaf8149a6aded637f472415f84e4246a21d979d3866d71b1e56242f8de2",
        strip_prefix = "googletest-a53e931dcd00c2556ee181d832e699c9f3c29036",
    )

    # Benchmarks for testing.
    http_archive(
        name = "com_google_benchmark",
        url = "https://github.com/google/benchmark/archive/v1.5.0.tar.gz",
        sha256 = "3c6a165b6ecc948967a1ead710d4a181d7b0fbcaa183ef7ea84604994966221a",
        strip_prefix = "benchmark-1.5.0",
    )

    # BoringSSL for cryptographic PRNG
    git_repository(
        name = "boringssl",
        # 2019-07-10
        commit = "776d803ffbb857b3a67c4ec14b671ff2b3ee65d2",
        remote = "https://boringssl.googlesource.com/boringssl",
        shallow_since = "1562793714 +0000",
    )

    # Supports `./configure && make` style packages to become dependencies.
    http_archive(
        name = "rules_foreign_cc",
        strip_prefix = "rules_foreign_cc-c29236959744be4d5ca47ac0b8fc4c454a04b852",
        # 2020-05-04
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/c29236959744be4d5ca47ac0b8fc4c454a04b852.tar.gz",
        sha256 = "c694abd387911f9750e7eddeff09baf10191e25193d93b8d77e35e554157615a",
    )

    # Postgres depends on rules_foreign_cc. Use postgres 11.
    http_archive(
        name = "postgres",
        url = "https://github.com/postgres/postgres/archive/REL_11_7.tar.gz",
        build_file = "@com_google_differential_privacy//differential_privacy/postgres:postgres.BUILD",
        strip_prefix = "postgres-REL_11_7",
        sha256 = "8c427e10a5f8b6be76353e83c7cf0171ac0e85308d352b8c129612002bb342eb",
    )
