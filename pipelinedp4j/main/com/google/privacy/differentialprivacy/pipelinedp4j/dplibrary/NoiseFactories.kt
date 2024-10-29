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

package com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary

import com.google.privacy.differentialprivacy.GaussianNoise
import com.google.privacy.differentialprivacy.LaplaceNoise
import com.google.privacy.differentialprivacy.Noise
import com.google.privacy.differentialprivacy.ZeroNoise
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.LAPLACE
import java.io.Serializable

/** Generates a [Noise] instance with the given [NoiseKind]. */
class NoiseFactory : (NoiseKind) -> Noise, Serializable {
  override fun invoke(noiseKind: NoiseKind) =
    when (noiseKind) {
      LAPLACE -> LaplaceNoise()
      GAUSSIAN -> GaussianNoise()
    }
}

/** For any [NoiseKind] returns ZeroNoise (i.e. no noise at all). */
class ZeroNoiseFactory : (NoiseKind) -> Noise, Serializable {
  override fun invoke(noiseKind: NoiseKind) = ZeroNoise()
}
