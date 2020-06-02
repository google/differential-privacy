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
        urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.11.4.tar.gz"],
        sha256 = "a79d19dcdf9139fa4b81206e318e33d245c4c9da1ffed21c87288ed4380426f9",
        strip_prefix = "protobuf-3.11.4",
    )
