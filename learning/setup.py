# Copyright 2021 Google LLC.
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
"""Setup for DP Learning package."""

import os
import setuptools

here = os.path.dirname(os.path.abspath(__file__))


def _parse_requirements(path):
  """Parses requirements from file."""
  with open(os.path.join(here, path)) as f:
    return [line.rstrip() for line in f] + ["dp-accounting"]


setuptools.setup(
    name="dp-learning",
    author="Google Differential Privacy Team",
    author_email="dp-open-source@google.com",
    description="Differential privacy learning algorithms",
    long_description_content_type="text/markdown",
    url="https://github.com/google/differential-privacy/",
    packages=setuptools.find_packages(),
    install_requires=_parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    license="Apache 2.0",
    keywords="differential-privacy clustering",
)
