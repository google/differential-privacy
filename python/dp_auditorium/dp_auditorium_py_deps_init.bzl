#
# Copyright 2024 Google LLC
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

"""Initialization of the Differential Privacy Tester Library."""

load("@rules_python//python:pip.bzl", "pip_parse")

def dp_auditorium_py_deps_init(workspace_name):
    pip_parse(
        name = "dp_auditorium_py_pip_deps",
        requirements_lock = "@" + workspace_name + "//:requirements.txt",
    )
