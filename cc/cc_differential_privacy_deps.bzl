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
        url = "https://github.com/abseil/abseil-cpp/archive/refs/tags/20240722.0.tar.gz",
        sha256 = "f50e5ac311a81382da7fa75b97310e4b9006474f9560ac46f54a9967f07d4ae3",
        strip_prefix = "abseil-cpp-20240722.0",
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
        url = "https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz",
        strip_prefix = "googletest-1.14.0",
        sha256 = "8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7",
    )

    # RE2 is a requirement for GoogleTest
    http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "4e6593ac3c71de1c0f322735bc8b0492a72f66ffccfad76e259fa21c41d27d8a",
        strip_prefix = "re2-2023-11-01",
        # release 2023-06-01
        url = "https://github.com/google/re2/releases/download/2023-11-01/re2-2023-11-01.tar.gz",
    )

    # Benchmarks for testing.
    http_archive(
        name = "com_google_benchmark",
        url = "https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz",
        sha256 = "6bc180a57d23d4d9515519f92b0c83d61b05b5bab188961f36ac7b06b0d9e9ce",
        strip_prefix = "benchmark-1.8.3",
    )

    # BoringSSL for cryptographic PRNG
    http_archive(
        name = "boringssl",
        # Commit date: 2023-10-26
        # Note for updating: we need to use a commit from the `master-with-bazel` branch.
        url = "https://github.com/google/boringssl/archive/add3674f646bcc3dfa828f308454fb3b37919512.tar.gz",
        sha256 = "f8b81f1741667e4a5aa9f0cc3e873875f7f832b3b141b8ee3a5d5863f992b8ba",
        strip_prefix = "boringssl-add3674f646bcc3dfa828f308454fb3b37919512",
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
