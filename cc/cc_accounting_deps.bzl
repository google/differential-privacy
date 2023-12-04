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

    # Begin Boost
    git_repository(
        name = "com_github_nelhage_rules_boost",
        commit = "f2494bf3b9de990889ae05a484e5f0fabf1fbdc9",
        remote = "https://github.com/nelhage/rules_boost",
        shallow_since = "1679023729 +0000",
    )
    # End boost
