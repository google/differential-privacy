//
// Copyright 2023 Google LLC
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

/** {@link Noise} implementation that adds 0 noise. Should be used in tests only. */
public final class ZeroNoise implements Noise {

  @Override
  public double addNoise(
      double x, int l0Sensitivity, double lInfSensitivity, double epsilon, double delta) {
    return x;
  }

  @Override
  public long addNoise(
      long x, int l0Sensitivity, long lInfSensitivity, double epsilon, double delta) {
    return x;
  }

  @Override
  public ConfidenceInterval computeConfidenceInterval(
      double noisedX,
      int l0Sensitivity,
      double lInfSensitivity,
      double epsilon,
      @Nullable Double delta,
      double alpha) {
    return ConfidenceInterval.create(noisedX, noisedX);
  }

  @Override
  public ConfidenceInterval computeConfidenceInterval(
      long noisedX,
      int l0Sensitivity,
      long lInfSensitivity,
      double epsilon,
      @Nullable Double delta,
      double alpha) {
    return ConfidenceInterval.create(noisedX, noisedX);
  }

  @Override
  public MechanismType getMechanismType() {
    return MechanismType.MECHANISM_NONE;
  }

  @Override
  public double computeQuantile(
      double rank,
      double x,
      int l0Sensitivity,
      double lInfSensitivity,
      double epsilon,
      @Nullable Double delta) {
    return x;
  }
}
