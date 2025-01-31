/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.privacy.differentialprivacy.pipelinedp4j.api

import com.google.privacy.differentialprivacy.pipelinedp4j.core.ExecutionMode

/**
 * The test mode to apply in tests to make them deterministic.
 *
 * Always use NONE in production code.
 *
 * @property NONE default, no test mode, the only mode that can be used in production, applies all
 *   privacy measures requested by the pipeline code.
 * @property FULL no contribution bounding, no noise and no groups selection. The pipeline will
 *   calculate metrics as if no privacy measures were applied.
 */
enum class TestMode {
  NONE,
  FULL,
}

/**
 * Converts the [TestMode] to the [ExecutionMode] which is used internally.
 *
 * We delibaretly do not expose the internal classes in the public API to limit the surface of the
 * API. This will give us more flexibility to change the implementation.
 */
internal fun TestMode.toExecutionMode() =
  when (this) {
    TestMode.NONE -> ExecutionMode.PRODUCTION
    TestMode.FULL -> ExecutionMode.FULL_TEST_MODE
  }
