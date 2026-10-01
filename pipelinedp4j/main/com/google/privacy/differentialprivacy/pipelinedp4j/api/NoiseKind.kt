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

import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind as InternalNoiseKind

/**
 * The kind of noise that will be applied to the data.
 *
 * @property LAPLACE noise is sampled from a Laplace distribution. To use it, you need to specify
 *   epsilon, delta is not required if groups are public.
 * @property GAUSSIAN noise is sampled from a Gaussian distribution. To use it, you need to specify
 *   both epsilon and delta in absolute budgets for aggregations and total budgets.
 * @property AUTO Let the system decide the noise kind. The system will choose the noise kind that
 *   gives the best utility (e.g., adds less noise) for the same level of privacy based on the
 *   parameters you provide and the aggregation you perform.
 */
enum class NoiseKind {
  LAPLACE,
  GAUSSIAN,
  AUTO,
}

/**
 * Converts the [NoiseKind] to the [InternalNoiseKind].
 *
 * We delibaretly do not expose the internal classes in the public API to limit the surface of the
 * API. This will give us more flexibility to change the implementation.
 */
internal fun NoiseKind.toInternalNoiseKind() =
  when (this) {
    NoiseKind.LAPLACE -> InternalNoiseKind.LAPLACE
    NoiseKind.GAUSSIAN -> InternalNoiseKind.GAUSSIAN
    NoiseKind.AUTO -> InternalNoiseKind.AUTO
  }
