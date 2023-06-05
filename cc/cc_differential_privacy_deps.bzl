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
        url = "https://github.com/abseil/abseil-cpp/archive/refs/tags/20230125.3.tar.gz",
        sha256 = "5366d7e7fa7ba0d915014d387b66d0d002c03236448e1ba9ef98122c13b35c36",
        strip_prefix = "abseil-cpp-20230125.3",
    )

    # Common bazel rules.  Also required for Abseil.
    http_archive(
        name = "bazel_skylib",
        sha256 = "b8a1527901774180afc798aeb28c4634bdccf19c4d98e7bdd1ce79d1fe9aaad7",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.4.1/bazel-skylib-1.4.1.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.1/bazel-skylib-1.4.1.tar.gz",
        ],
    )

    # GoogleTest/GoogleMock framework. Used by most unit-tests.
    http_archive(
        name = "com_google_googletest",
        url = "https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz",
        strip_prefix = "googletest-1.13.0",
        sha256 = "ad7fdba11ea011c1d925b3289cf4af2c66a352e18d4c7264392fead75e919363",
    )

    # RE2 is a requirement for GoogleTest
    http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "1726508efc93a50854c92e3f7ac66eb28f0e57652e413f11d7c1e28f97d997ba",
        strip_prefix = "re2-03da4fc0857c285e3a26782f6bc8931c4c950df4",
        # release 2023-06-01
        url = "https://github.com/google/re2/archive/03da4fc0857c285e3a26782f6bc8931c4c950df4.zip",
    )

    # Benchmarks for testing.
    http_archive(
        name = "com_google_benchmark",
        url = "https://github.com/google/benchmark/archive/refs/tags/v1.7.1.tar.gz",
        sha256 = "6430e4092653380d9dc4ccb45a1e2dc9259d581f4866dc0759713126056bc1d7",
        strip_prefix = "benchmark-1.7.1",
    )

    # BoringSSL for cryptographic PRNG
    http_archive(
        name = "boringssl",
        # Commit date: 2023-03-17
        # Note for updating: we need to use a commit from the `master-with-bazel` branch.
        url = "https://github.com/google/boringssl/archive/e0648e015f039ef88801ff0cf84dcb5944b8b5ab.tar.gz",
        sha256 = "b9ba36d3c309cfee56df70da8e8700f9ac65d4c0460f78bf8a4e580300b7f59d",
        strip_prefix = "boringssl-e0648e015f039ef88801ff0cf84dcb5944b8b5ab",
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
