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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static java.lang.Double.POSITIVE_INFINITY;
import static java.lang.Double.isFinite;

import com.google.differentialprivacy.SummaryOuterClass.MechanismType;
import java.util.Objects;
import javax.annotation.Nullable;

/** Utilities which validate the correctness of DP parameters. */
public class DpPreconditions {

  private DpPreconditions() {}

  static void checkEpsilon(double epsilon) {
    checkArgument(epsilon >= 1.0 / (1L << 50)
       && epsilon < POSITIVE_INFINITY,
        "epsilon must be > 0 and < infinity. Provided value: %s", epsilon);
  }

  static void checkNoiseDelta(Double delta, Noise noise) {
    if (noise instanceof LaplaceNoise) {
      checkArgument(
          delta == null,
          "delta should not be set when Laplace noise is used. Provided value: %s",
          delta);
    } else {
      checkNotNull(delta);
      checkDelta(delta);
    }
  }

  static void checkDelta(double delta) {
    checkArgument(delta > 0 && delta < 1, "delta must be > 0 and < 1. Provided value: %s", delta);
  }

  static void checkSensitivities(int l0Sensitivity, double lInfSensitivity) {
    checkL0Sensitivity(l0Sensitivity);
    checkArgument(
        lInfSensitivity > 0, "lInfSensitivity must be > 0. Provided value: %s", lInfSensitivity);
  }

  static void checkL0Sensitivity(int l0Sensitivity) {
    checkArgument(
        l0Sensitivity > 0, "l0Sensitivity must be > 0. Provided value: %s", l0Sensitivity);
  }

  static void checkL1Sensitivity(double l1Sensitivity) {
    checkArgument(
        l1Sensitivity > 0, "l1Sensitivity must be > 0. Provided value: %s", l1Sensitivity);
  }

  static void checkMaxPartitionsContributed(int maxPartitionsContributed) {
    // maxPartitionsContributed is the user-facing parameter, which is technically the same as
    // L0 sensitivity used by the noise internally.
    checkL0Sensitivity(maxPartitionsContributed);
  }

  static void checkMaxContributionsPerPartition(int maxContributionsPerPartition) {
    checkArgument(
        maxContributionsPerPartition > 0,
        "maxContributionsPerPartitions must be > 0. Provided value: %s",
        maxContributionsPerPartition);
  }

  static void checkBounds(double lower, double upper) {
    checkArgument(
        upper >= lower,
        "The upper bound should be greater than the lower bound. Provided values: "
            + "lower = %s upper = %s",
        lower,
        upper);
    checkArgument(isFinite(lower) && isFinite(upper),
        "Lower and upper bounds should be finite. Provided values: "
            + "lower = %s upper = %s",
        lower,
        upper);
  }

  static void checkMergeDeltaAreEqual(@Nullable Double delta1, double delta2) {
    if (delta1 != null) {
      checkArgument(Double.compare(delta1, delta2) == 0,
          "Failed to merge: unequal values of delta. "
              + "delta1 = %s, delta2 = %s", delta1, delta2);
    } else {
      checkArgument(Double.compare(delta2, 0.0) == 0,
          "Failed to merge: unequal values of delta. "
              + "delta1 = %s, delta2 = %s", delta1, delta2);
    }
  }

  static void checkMergeEpsilonAreEqual(double epsilon1, double epsilon2) {
    checkArgument(Double.compare(epsilon1, epsilon2) == 0,
        "Failed to merge: unequal values of epsilon. "
            + "epsilon1 = %s, epsilon2 = %s", epsilon1, epsilon2);
  }

  static void checkMergeBoundsAreEqual(
      double lower1, double lower2, double upper1, double upper2) {
    checkArgument(Double.compare(lower1, lower2) == 0,
        "Failed to merge: unequal lower bounds. "
            + "lower1 = %s, lower2 = %s", lower1, lower2);
    checkArgument(Double.compare(upper1, upper2) == 0,
        "Failed to merge: unequal upper bounds. "
            + "upper1 = %s, upper2 = %s", upper1, upper2);
  }

  static void checkMergeMaxContributionsPerPartitionAreEqual(
      int maxContributionsPerPartition1, int maxContributionsPerPartition2) {
    checkArgument(maxContributionsPerPartition1 == maxContributionsPerPartition2,
        "Failed to merge: unequal values of maxContributionsPerPartition. "
            + "maxContributionsPerPartition1 = %s, maxContributionsPerPartition2 = %s",
        maxContributionsPerPartition1, maxContributionsPerPartition2);
  }

  static void checkMergeMaxPartitionsContributedAreEqual(
      int maxPartitionsContributed1, int maxPartitionsContributed2) {
    checkArgument(maxPartitionsContributed1 == maxPartitionsContributed2,
        "Failed to merge: unequal values of maxPartitionsContributed. "
            + "maxPartitionsContributed1 = %s, maxPartitionsContributed2 = %s",
        maxPartitionsContributed1, maxPartitionsContributed2);
  }

  static void checkMergeMechanismTypesAreEqual(MechanismType type1, MechanismType type2) {
    checkArgument(Objects.equals(type1, type2),
        "Failed to merge: unequal mechanism types. type1 = %s, type2 = %s", type1, type2);
  }
}
