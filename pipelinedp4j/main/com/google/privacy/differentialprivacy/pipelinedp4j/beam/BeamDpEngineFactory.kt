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

package com.google.privacy.differentialprivacy.pipelinedp4j.beam

import com.google.privacy.differentialprivacy.pipelinedp4j.core.DpEngine
import com.google.privacy.differentialprivacy.pipelinedp4j.core.DpEngineBudgetSpec

// TODO: add tests either in respective test class or preferably delete this file and
// and parameterize the dp engine end2end tests with all different types of backends.

/** Creates a [DpEngine] that runs DP aggregations on Beam. */
fun DpEngine.Factory.createBeamEngine(budgetSpec: DpEngineBudgetSpec) =
  create(BeamEncoderFactory(), budgetSpec)
