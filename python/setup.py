# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup for DP Accounting package."""

import os
import setuptools

here = os.path.dirname(os.path.abspath(__file__))


def _get_version():
  """Parses the version number from VERSION file."""
  with open(os.path.join(here, "VERSION")) as f:
    try:
      version_line = next(
          line for line in f if not line.startswith("\"\"\""))
    except StopIteration:
      raise ValueError("Version not defined in VERSION")
    else:
      return version_line.strip('\n \'"')


def _parse_requirements(path):
  """Parses requirements from file."""
  with open(os.path.join(here, path)) as f:
    deps = []
    for line in f:
      if line.startswith("dataclasses"):
        # For python version 3.7 onwards, dataclasses module is already included
        # as part of the core library.
        deps.append("dataclasses; python_version < \'3.7\'")
      elif not (line.isspace() or line.startswith("#")):
        deps.append(line.rstrip())
    return deps


def _read_description(path):
  """Read the description from README file."""
  # Exclude example from package description.
  return open(os.path.join(here, path)).read().split("## Examples")[0]

setuptools.setup(
    name="dp-accounting",
    version=_get_version(),
    author="Google Differential Privacy Team",
    author_email="dp-open-source@google.com",
    description="Tools for tracking differential privacy budgets",
    long_description=_read_description("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/google/differential-privacy/",
    packages=setuptools.find_packages(),
    install_requires=_parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    license="Apache 2.0",
    keywords="differential-privacy accounting",
)
