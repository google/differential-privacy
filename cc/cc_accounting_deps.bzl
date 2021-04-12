#
# Copyright 2021 Google LLC
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
"""Declares dependencies of the differential privacy accounting."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def cc_accounting_deps():
    """Loads required dependencies."""
    http_archive(
        name = "kissfft",
        build_file_content = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])""",
        strip_prefix = "kissfft-8f47a67f595a6641c566087bf5277034be64f24d",
        urls = ["https://github.com/mborgerding/kissfft/archive/8f47a67f595a6641c566087bf5277034be64f24d.tar.gz"],
        sha256 = "93cfa11a344ad552472f7d93c228d55969ac586275692d73d5e7ce73a69b047f",
    )

    # Begin GLog
    http_archive(
        name = "com_github_gflags_gflags",
        sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
        strip_prefix = "gflags-2.2.2",
        urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
    )

    http_archive(
        name = "com_github_google_glog",
        sha256 = "62efeb57ff70db9ea2129a16d0f908941e355d09d6d83c9f7b18557c0a7ab59e",
        strip_prefix = "glog-d516278b1cd33cd148e8989aec488b6049a4ca0b",
        urls = ["https://github.com/google/glog/archive/d516278b1cd33cd148e8989aec488b6049a4ca0b.zip"],
    )
    # End Glog

    # Begin Boost
    git_repository(
        name = "com_github_nelhage_rules_boost",
        commit = "1e3a69bf2d5cd10c34b74f066054cd335d033d71",
        remote = "https://github.com/nelhage/rules_boost",
        shallow_since = "1591047380 -0700",
    )
    # End boost
