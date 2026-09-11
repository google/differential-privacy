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

package com.google.privacy.differentialprivacy.pipelinedp4j.core

import com.google.privacy.differentialprivacy.ConfidenceInterval
import com.google.privacy.differentialprivacy.GaussianNoise
import com.google.privacy.differentialprivacy.LaplaceNoise
import com.google.privacy.differentialprivacy.Noise
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType
import java.io.Serializable
import kotlin.math.pow

/**
 * A [Noise] implementation that automatically selects between Gaussian and Laplace noise based on
 * which one provides lower variance for the given parameters.
 */
class AutoNoise : Noise, Serializable {
  private val laplaceNoise = LaplaceNoise()
  private val gaussianNoise = GaussianNoise()

  override fun addNoise(
    x: Double,
    l0Sensitivity: Int,
    lInfSensitivity: Double,
    epsilon: Double,
    delta: Double,
  ): Double {
    if (shouldUseGaussian(l0Sensitivity, lInfSensitivity, epsilon, delta)) {
      return gaussianNoise.addNoise(x, l0Sensitivity, lInfSensitivity, epsilon, delta)
    }
    return laplaceNoise.addNoise(x, l0Sensitivity, lInfSensitivity, epsilon, 0.0)
  }

  override fun addNoise(
    x: Long,
    l0Sensitivity: Int,
    lInfSensitivity: Long,
    epsilon: Double,
    delta: Double,
  ): Long {
    if (shouldUseGaussian(l0Sensitivity, lInfSensitivity.toDouble(), epsilon, delta)) {
      return gaussianNoise.addNoise(x, l0Sensitivity, lInfSensitivity, epsilon, delta)
    }
    return laplaceNoise.addNoise(x, l0Sensitivity, lInfSensitivity, epsilon, 0.0)
  }

  override fun computeConfidenceInterval(
    noisedX: Double,
    l0Sensitivity: Int,
    lInfSensitivity: Double,
    epsilon: Double,
    delta: Double?,
    alpha: Double,
  ): ConfidenceInterval {
    val d = delta ?: 0.0
    if (shouldUseGaussian(l0Sensitivity, lInfSensitivity, epsilon, d)) {
      return gaussianNoise.computeConfidenceInterval(
        noisedX,
        l0Sensitivity,
        lInfSensitivity,
        epsilon,
        delta,
        alpha,
      )
    }
    return laplaceNoise.computeConfidenceInterval(
      noisedX,
      l0Sensitivity,
      lInfSensitivity,
      epsilon,
      0.0,
      alpha,
    )
  }

  override fun computeConfidenceInterval(
    noisedX: Long,
    l0Sensitivity: Int,
    lInfSensitivity: Long,
    epsilon: Double,
    delta: Double?,
    alpha: Double,
  ): ConfidenceInterval {
    val d = delta ?: 0.0
    if (shouldUseGaussian(l0Sensitivity, lInfSensitivity.toDouble(), epsilon, d)) {
      return gaussianNoise.computeConfidenceInterval(
        noisedX,
        l0Sensitivity,
        lInfSensitivity,
        epsilon,
        delta,
        alpha,
      )
    }
    return laplaceNoise.computeConfidenceInterval(
      noisedX,
      l0Sensitivity,
      lInfSensitivity,
      epsilon,
      0.0,
      alpha,
    )
  }

  override fun computeQuantile(
    rank: Double,
    x: Double,
    l0Sensitivity: Int,
    lInfSensitivity: Double,
    epsilon: Double,
    delta: Double?,
  ): Double {
    val d = delta ?: 0.0
    if (shouldUseGaussian(l0Sensitivity, lInfSensitivity, epsilon, d)) {
      return gaussianNoise.computeQuantile(rank, x, l0Sensitivity, lInfSensitivity, epsilon, delta)
    }
    return laplaceNoise.computeQuantile(rank, x, l0Sensitivity, lInfSensitivity, epsilon, 0.0)
  }

  override fun getMechanismType(): MechanismType {
    return MechanismType.MECHANISM_NONE
  }

  private fun shouldUseGaussian(
    l0Sensitivity: Int,
    lInfSensitivity: Double,
    epsilon: Double,
    delta: Double,
  ): Boolean {
    if (delta == 0.0) return false
    val l1 = Noise.getL1Sensitivity(l0Sensitivity, lInfSensitivity)
    val l2 = Noise.getL2Sensitivity(l0Sensitivity, lInfSensitivity)

    val laplaceVar = 2.0 * (l1 / epsilon).pow(2)
    val sigma = GaussianNoise.getSigma(l2, epsilon, delta)
    val gaussianVar = sigma * sigma

    return gaussianVar < laplaceVar
  }
}

/** Generates an [AutoNoise] instance. */
class AutoNoiseFactory : (NoiseKind) -> Noise, Serializable {
  override fun invoke(noiseKind: NoiseKind): Noise {
    return AutoNoise()
  }
}
