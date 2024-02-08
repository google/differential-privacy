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

import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType;
import javax.annotation.Nullable;

/**
 * Interface for primitives that add noise to numerical data, for use in differential privacy
 * operations.
 */
public interface Noise {

  /**
   * @deprecated Use {@link #addNoise(double, int, double, double, double)} instead. Set delta to
   *     0.0 if it isn't used.
   */
  @Deprecated
  default double addNoise(
      double x, int l0Sensitivity, double lInfSensitivity, double epsilon, @Nullable Double delta) {
    double primitiveDelta = delta == null ? 0.0 : delta;
    return addNoise(x, l0Sensitivity, lInfSensitivity, epsilon, primitiveDelta);
  }

  double addNoise(
      double x, int l0Sensitivity, double lInfSensitivity, double epsilon, double delta);

  /**
   * @deprecated Use {{@link #addNoise(long, int, long, double, double)}} instead. Set delta to 0.0
   *     if it isn't used.
   */
  @Deprecated
  default long addNoise(
      long x, int l0Sensitivity, long lInfSensitivity, double epsilon, @Nullable Double delta) {
    double primitiveDelta = delta == null ? 0.0 : delta;
    return addNoise(x, l0Sensitivity, lInfSensitivity, epsilon, primitiveDelta);
  }

  long addNoise(long x, int l0Sensitivity, long lInfSensitivity, double epsilon, double delta);

  ConfidenceInterval computeConfidenceInterval(
      double noisedX,
      int l0Sensitivity,
      double lInfSensitivity,
      double epsilon,
      @Nullable Double delta,
      double alpha);

  ConfidenceInterval computeConfidenceInterval(
      long noisedX,
      int l0Sensitivity,
      long lInfSensitivity,
      double epsilon,
      @Nullable Double delta,
      double alpha);

  MechanismType getMechanismType();

  static double getL1Sensitivity(int l0Sensitivity, double lInfSensitivity) {
    return l0Sensitivity * lInfSensitivity;
  }

  static double getL2Sensitivity(int l0Sensitivity, double lInfSensitivity) {
    return Math.sqrt(l0Sensitivity) * lInfSensitivity;
  }

  /**
   * Computes the quantile z satisfying Pr[Y <= z] = {@code rank} for a random variable Y
   * whose distribution is given by applying the Noise mechanism to the raw value {@code x} using
   * the specified privacy parameters {@code epsilon}, {@code delta}, {@code l0Sensitivity}, and
   * {@code lInfSensitivity}.
   */
  double computeQuantile(
      double rank,
      double x,
      int l0Sensitivity,
      double lInfSensitivity,
      double epsilon,
      @Nullable Double delta);
}
