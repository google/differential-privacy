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

import com.google.common.truth.Truth.assertThat
import com.google.privacy.differentialprivacy.GaussianNoise
import com.google.privacy.differentialprivacy.LaplaceNoise
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class AutoNoiseTest {

  @Test
  fun addNoise_deltaZero_usesLaplace() {
    val autoNoise = AutoNoise()
    // This would throw if it tried to use Gaussian with delta=0.
    // We expect it to use Laplace .
    val unused = autoNoise.addNoise(0.0, 1, 1.0, 1.0, 0.0)
  }

  @Test
  fun computeQuantile_gaussianBetter_matchesGaussian() {
    val autoNoise = AutoNoise()
    // High l0 sensitivity favors Gaussian.
    val l0 = 100
    val lInf = 1.0
    val epsilon = 1.0
    val delta = 1e-5

    val rank = 0.9
    val result = autoNoise.computeQuantile(rank, 0.0, l0, lInf, epsilon, delta)

    val gaussianNoise = GaussianNoise()
    val expected = gaussianNoise.computeQuantile(rank, 0.0, l0, lInf, epsilon, delta)

    assertThat(result).isEqualTo(expected)
  }

  @Test
  fun computeQuantile_laplaceBetter_matchesLaplace() {
    val autoNoise = AutoNoise()
    // Low l0 sensitivity and small epsilon favors Laplace.
    val l0 = 1
    val lInf = 1.0
    val epsilon = 0.1
    val delta = 1e-10

    val rank = 0.9
    val result = autoNoise.computeQuantile(rank, 0.0, l0, lInf, epsilon, delta)

    val laplaceNoise = LaplaceNoise()
    val expected = laplaceNoise.computeQuantile(rank, 0.0, l0, lInf, epsilon, 0.0)

    assertThat(result).isEqualTo(expected)
  }

  @Test
  fun computeConfidenceInterval_gaussianBetter_matchesGaussian() {
    val autoNoise = AutoNoise()
    val l0 = 100
    val lInf = 1.0
    val epsilon = 1.0
    val delta = 1e-5
    val alpha = 0.05
    val noisedX = 10.0

    val result = autoNoise.computeConfidenceInterval(noisedX, l0, lInf, epsilon, delta, alpha)

    val gaussianNoise = GaussianNoise()
    val expected = gaussianNoise.computeConfidenceInterval(noisedX, l0, lInf, epsilon, delta, alpha)

    assertThat(result.lowerBound()).isEqualTo(expected.lowerBound())
    assertThat(result.upperBound()).isEqualTo(expected.upperBound())
  }

  @Test
  fun computeConfidenceInterval_laplaceBetter_matchesLaplace() {
    val autoNoise = AutoNoise()
    val l0 = 1
    val lInf = 1.0
    val epsilon = 0.1
    val delta = 1e-10
    val alpha = 0.05
    val noisedX = 10.0

    val result = autoNoise.computeConfidenceInterval(noisedX, l0, lInf, epsilon, delta, alpha)

    val laplaceNoise = LaplaceNoise()
    val expected = laplaceNoise.computeConfidenceInterval(noisedX, l0, lInf, epsilon, 0.0, alpha)

    assertThat(result.lowerBound()).isEqualTo(expected.lowerBound())
    assertThat(result.upperBound()).isEqualTo(expected.upperBound())
  }
}
