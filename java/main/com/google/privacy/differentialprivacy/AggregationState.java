//
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package com.google.privacy.differentialprivacy;

/** Represents the state of the aggregation. */
public enum AggregationState {
  /** Object hasn't been serialized and computeResult() hasn't been called. */
  DEFAULT(""),
  /** Object has been serialized. */
  SERIALIZED("Object has been already serialized."),
  /** computeResult() was called. */
  RESULT_RETURNED("DP result was already computed and returned.");
  private final String errorMessage;

  AggregationState(String errorMessage) {
    this.errorMessage = errorMessage;
  }

  public String getErrorMessage() {
    return errorMessage;
  }
}
