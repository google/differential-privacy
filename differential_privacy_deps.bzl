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
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

def differential_privacy_deps():
    """ Macro to include the differential privacy library's critical dependencies in a WORKSPACE.

    """

    # Abseil
    if not native.existing_rule("com_google_absl"):
        git_repository(
            name = "com_google_absl",
            # Commit from 2019-08-15
            commit = "aae8143cf9aa611f70d7ea9b95b8b8b383b2271a",
            remote = "https://github.com/abseil/abseil-cpp",
        )

    # Common bazel rules
    if not native.existing_rule("bazel_skylib"):
        http_archive(
            name = "bazel_skylib",
            type = "tar.gz",
            url = "https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz",
            sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
        )

    # Protobuf
    if not native.existing_rule("com_google_protobuf"):
        http_archive(
            name = "com_google_protobuf",
            urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.8.0.tar.gz"],
            sha256 = "03d2e5ef101aee4c2f6ddcf145d2a04926b9c19e7086944df3842b1b8502b783",
            strip_prefix = "protobuf-3.8.0",
        )

    # GoogleTest/GoogleMock framework. Used by most unit-tests.
    if not native.existing_rule("com_google_googletest"):
        http_archive(
            name = "com_google_googletest",
            # Commit on 2019-04-18
            urls = [
                "https://github.com/google/googletest/archive/a53e931dcd00c2556ee181d832e699c9f3c29036.tar.gz",
            ],
            strip_prefix = "googletest-a53e931dcd00c2556ee181d832e699c9f3c29036",
            sha256 = "7850caaf8149a6aded637f472415f84e4246a21d979d3866d71b1e56242f8de2",
        )

    # Benchmarks for testing.
    if not native.existing_rule("com_google_benchmark"):
        git_repository(
            name = "com_google_benchmark",
            # Commit from 2019-07-22
            commit = "8e48105d465c586068dd8e248fe75a8971c6ba3a",
            remote = "https://github.com/google/benchmark",
        )

    # BoringSSL for cryptographic PRNG
    if not native.existing_rule("boringssl"):
        git_repository(
            name = "boringssl",
            # 2019-07-10
            commit = "776d803ffbb857b3a67c4ec14b671ff2b3ee65d2",
            remote = "https://boringssl.googlesource.com/boringssl",
        )

    # Supports `./configure && make` style packages to become dependencies.
    if not native.existing_rule("rules_foreign_cc"):
        git_repository(
            name = "rules_foreign_cc",
            # Commit last updated 17 July 2019.
            commit = "a209b642c7687a8894c19b3dd40e43e6d3f38e83",
            remote = "https://github.com/bazelbuild/rules_foreign_cc.git",
        )

    # Postgres depends on rules_foreign_cc.
    if not native.existing_rule("postgres"):
        new_git_repository(
            name = "postgres",
            # 2019-08-16 stable version 11.
            commit = "aed967d697de19a78a653926c72604f9b04c3b1e",
            remote = "https://github.com/postgres/postgres/",
            build_file = "@com_google_differential_privacy//differential_privacy/postgres:postgres.BUILD",
        )
