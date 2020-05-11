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

import com.google.differentialprivacy.SummaryOuterClass.MechanismType;
import javax.annotation.Nullable;

/**
 * Interface for primitives that add noise to numerical data, for use in differential privacy
 * operations.
 */
public interface Noise {

  double addNoise(
      double x, int l0Sensitivity, double lInfSensitivity, double epsilon, @Nullable Double delta);

  long addNoise(
      long x, int l0Sensitivity, long lInfSensitivity, double epsilon, @Nullable Double delta);

  MechanismType getMechanismType();

  static double getL1Sensitivity(int l0Sensitivity, double lInfSensitivity) {
    return l0Sensitivity * lInfSensitivity;
  }

  static double getL2Sensitivity(int l0Sensitivity, double lInfSensitivity) {
    return Math.sqrt(l0Sensitivity) * lInfSensitivity;
  }
}
