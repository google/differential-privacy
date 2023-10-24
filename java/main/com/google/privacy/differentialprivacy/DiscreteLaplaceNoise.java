//
// Copyright 2022 Google LLC
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

import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType;
import java.security.SecureRandom;
import java.util.Random;
import javax.annotation.Nullable;

/**
 * Generates and adds Discrete Laplace noise to a raw piece of numerical data such that the result
 * is securely differentially private.
 */
public class DiscreteLaplaceNoise implements Noise {
  private final Random random;

  /** Returns a Noise instance initialized with a secure randomness source. */
  public DiscreteLaplaceNoise() {
    this(new SecureRandom());
  }

  private DiscreteLaplaceNoise(Random random) {
    this.random = random;
  }

  /**
   * Returns a Noise instance initialized with a specified randomness source. This should only be
   * used for testing and may only be called via the static methods in {@link TestNoiseFactory}.
   *
   * <p>This method is package-private for use by the factory.
   */
  static DiscreteLaplaceNoise createForTesting(Random random) {
    return new DiscreteLaplaceNoise(random);
  }

  @Override
  public double addNoise(
      double x, int l0Sensitivity, double lInfSensitivity, double epsilon, double delta) {
    throw new IllegalArgumentException("Discrete Laplace Mechanism only applies to integers.");
  }

  /**
   * Adds Discrete Laplace noise to the integer {@code x} such that the output is {@code
   * epsilon}-differentially private, with respect to the specified L_0 and L_inf sensitivities.
   * Note that {@code delta} must be set to {@code null} because it does not parameterize Laplace
   * noise. Moreover, {@code epsilon} must be at least 2^-50.
   */
  @Override
  public long addNoise(
      long x, int l0Sensitivity, long lInfSensitivity, double epsilon, double delta) {
    DpPreconditions.checkSensitivities(l0Sensitivity, lInfSensitivity);

    return addNoise(
        x, (long) Noise.getL1Sensitivity(l0Sensitivity, lInfSensitivity), epsilon, delta);
  }

  /**
   * @deprecated Use {@link #addNoise(long, long, double, double)} instead. Set delta to 0.0.
   */
  @Deprecated
  public long addNoise(long x, long l1Sensitivity, double epsilon, @Nullable Double delta) {
    double primitiveDelta = delta == null ? 0.0 : delta;
    return addNoise(x, l1Sensitivity, epsilon, primitiveDelta);
  }

  /**
   * See {@link #addNoise(long, int, long, double, double)}.
   *
   * <p>As opposed to the latter method, this accepts the L_1 sensitivity of {@code x} directly
   * instead of the L_0 and L_Inf proxies. This should be used in settings where it is feasible or
   * more convenient to calculate the L_1 sensitivity directly.
   */
  public long addNoise(long x, long l1Sensitivity, double epsilon, double delta) {
    checkParameters(l1Sensitivity, epsilon, delta);

    return x + SamplingUtil.sampleTwoSidedGeometric(random, epsilon / l1Sensitivity);
  }

  @Override
  public MechanismType getMechanismType() {
    return MechanismType.DISCRETE_LAPLACE;
  }

  @Override
  public ConfidenceInterval computeConfidenceInterval(
      long noisedX,
      int l0Sensitivity,
      long lInfSensitivity,
      double epsilon,
      @Nullable Double delta,
      double alpha) {
    throw new UnsupportedOperationException("Not implemented yet.");
  }

  @Override
  public ConfidenceInterval computeConfidenceInterval(
      double noisedX,
      int l0Sensitivity,
      double lInfSensitivity,
      double epsilon,
      @Nullable Double delta,
      double alpha) {
    throw new IllegalArgumentException("Discrete Laplace Mechanism only outputs integers.");
  }

  @Override
  public double computeQuantile(
      double rank,
      double x,
      int l0Sensitivity,
      double lInfSensitivity,
      double epsilon,
      @Nullable Double delta) {
    throw new UnsupportedOperationException("Not implemented yet.");
  }

  private void checkParameters(double l1Sensitivity, double epsilon, @Nullable Double delta) {
    DpPreconditions.checkEpsilon(epsilon);
    DpPreconditions.checkNoiseDelta(delta, this);
    DpPreconditions.checkL1Sensitivity(l1Sensitivity);
  }
}
