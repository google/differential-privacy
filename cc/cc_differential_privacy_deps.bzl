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
        url = "https://github.com/abseil/abseil-cpp/archive/refs/tags/20240722.1.tar.gz",
        sha256 = "40cee67604060a7c8794d931538cb55f4d444073e556980c88b6c49bb9b19bb7",
        strip_prefix = "abseil-cpp-20240722.1",
    )

    # Common bazel rules.  Also required for Abseil.
    http_archive(
        name = "bazel_skylib",
        sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        ],
    )

    # GoogleTest/GoogleMock framework. Used by most unit-tests.
    http_archive(
        name = "com_google_googletest",
        url = "https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz",
        strip_prefix = "googletest-1.15.2",
        sha256 = "7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926",
    )

    # RE2 is a requirement for GoogleTest
    http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "eb2df807c781601c14a260a507a5bb4509be1ee626024cb45acbd57cb9d4032b",
        strip_prefix = "re2-2024-07-02",
        url = "https://github.com/google/re2/releases/download/2024-07-02/re2-2024-07-02.tar.gz",
    )

    # Benchmarks for testing.
    http_archive(
        name = "com_google_benchmark",
        url = "https://github.com/google/benchmark/archive/refs/tags/v1.8.5.tar.gz",
        sha256 = "d26789a2b46d8808a48a4556ee58ccc7c497fcd4c0af9b90197674a81e04798a",
        strip_prefix = "benchmark-1.8.5",
    )

    # BoringSSL for cryptographic PRNG
    http_archive(
        name = "boringssl",
        # Commit date: 2025-02-18
        # Note for updating: we need to use a commit from the `master-with-bazel` branch.
        url = "https://github.com/google/boringssl/archive/9802ee3a03f1a601527d724d72173d405615c6aa.tar.gz",
        strip_prefix = "boringssl-9802ee3a03f1a601527d724d72173d405615c6aa",
        sha256 = "a0d0c144f7f934932dafe5b341881f389c60b7749e1745f5b2877db62dc8145b",
    )

    # Supports `./configure && make` style packages to become dependencies.
    http_archive(
        name = "rules_foreign_cc",
        strip_prefix = "rules_foreign_cc-0.10.1",
        url = "https://github.com/bazelbuild/rules_foreign_cc/releases/download/0.10.1/rules_foreign_cc-0.10.1.tar.gz",
        sha256 = "476303bd0f1b04cc311fc258f1708a5f6ef82d3091e53fd1977fa20383425a6a",
    )

    # Postgres depends on rules_foreign_cc. Use postgres 12.
    http_archive(
        name = "postgres",
        url = "https://github.com/postgres/postgres/archive/REL_12_9.tar.gz",
        build_file = "@com_google_differential_privacy//cc/postgres:postgres.BUILD",
        strip_prefix = "postgres-REL_12_9",
        sha256 = "64f6da47aab9ac65d07b31abd40445b4a0413d4265d25b82ed738abad8a98349",
    )
