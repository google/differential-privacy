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

TOOL='bazel'
DP_DIR='differential_privacy/postgres'

$TOOL build $DP_DIR:anon_func.so
/bin/mkdir -p $PG_DIR/lib
/bin/mkdir -p $PG_DIR/share/extension
/usr/bin/install -c -m 755 $TOOL-bin/$DP_DIR/anon_func.so $PG_DIR/lib/
/usr/bin/install -c -m 644 $DP_DIR/anon_func.control $PG_DIR/share/extension/
/usr/bin/install -c -m 644 $DP_DIR/anon_func--1.0.0.sql  $PG_DIR/share/extension/
