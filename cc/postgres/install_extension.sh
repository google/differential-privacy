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

#!/bin/bash

echo "Currently set postgres directory:" $PG_DIR

set -ex

WORKSPACE_DIR=`bazel info workspace`
BIN_DIR=`bazel info -c opt bazel-bin`

LIB_DIR=`ppg_config --pkglibdir`
SHARE_DIR=`pg_config --sharedir`

bazel build -c opt //postgres:anon_func.so
sudo install -c -m 755 $BIN_DIR/postgres/anon_func.so $LIB_DIR
sudo install -c -m 644 $WORKSPACE_DIR/postgres/anon_func.control $SHARE_DIR/extension/
sudo install -c -m 644 $WORKSPACE_DIR/postgres/anon_func--1.0.0.sql  $SHARE_DIR/extension/
