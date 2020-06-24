#
# Copyright 2020 Google LLC
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

"""Dependencies of Python Privacy Loss Distribution."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def accounting_py_deps():
    """ Loads dependencies of Python Privacy Loss Distribution.
    """

    if not native.existing_rule("rules_python"):
        # Commit from 2020-03-05
        http_archive(
            name = "rules_python",
            strip_prefix = "rules_python-748aa53d7701e71101dfd15d800e100f6ff8e5d1",
            url = "https://github.com/bazelbuild/rules_python/archive/748aa53d7701e71101dfd15d800e100f6ff8e5d1.zip",
            sha256 = "d3e40ca3b7e00b72d2b1585e0b3396bcce50f0fc692e2b7c91d8b0dc471e3eaf",
        )
