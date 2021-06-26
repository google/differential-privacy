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

def differential_privacy_deps():
    """ Macro to include the differential privacy library's critical dependencies in a WORKSPACE.

    """

    # Protobuf
    http_archive(
        name = "com_google_protobuf",
        urls = ["https://github.com/protocolbuffers/protobuf/archive/refs/tags/v3.17.3.tar.gz"],
        sha256 = "c6003e1d2e7fefa78a3039f19f383b4f3a61e81be8c19356f85b6461998ad3db",
        strip_prefix = "protobuf-3.17.3",
    )
