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
import static java.lang.Double.isFinite;

import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType;
import java.util.Objects;
import javax.annotation.Nullable;

/** Utilities which validate the correctness of DP parameters. */
public class DpPreconditions {

  private DpPreconditions() {}

  static void checkEpsilon(double epsilon) {
    double epsilonLowerBound = 1.0 / (1L << 50);
    checkArgument(
        Double.isFinite(epsilon) && epsilon >= epsilonLowerBound,
        "epsilon must be >= %s and < infinity. Provided value: %s",
        epsilonLowerBound,
        epsilon);
  }

  static void checkRho(double rho) {
    double rhoLowerBound = 1.0 / (1L << 50);
    checkArgument(
        Double.isFinite(rho) && rho >= rhoLowerBound,
        "rho must be >= %s and < infinity. Provided value: %s",
        rhoLowerBound,
        rho);
  }

  static void checkNoiseDelta(Double delta, Noise noise) {
    if (noise.getMechanismType() == MechanismType.LAPLACE
        || noise.getMechanismType() == MechanismType.DISCRETE_LAPLACE) {
      checkArgument(
          delta == null || delta == 0.0,
          "delta should not be set when (Discrete) Laplace noise is used. Provided value: %s",
          delta);
    } else if (noise.getMechanismType() == MechanismType.GAUSSIAN) {
      checkNotNull(delta, "delta should not be null when Gaussian noise is used.");
      checkDelta(delta);
      // For unknown noise, delta may or may not be null, but if it is not null it should be between
      // 0 and 1.
    } else if (delta != null) {
      checkArgument(
          delta >= 0 && delta < 1, "delta must be >= 0 and < 1. Provided value: %s", delta);
    }
  }

  static void checkDelta(double delta) {
    checkArgument(delta > 0 && delta < 1, "delta must be > 0 and < 1. Provided value: %s", delta);
  }

  static void checkSensitivities(int l0Sensitivity, double lInfSensitivity) {
    checkL0Sensitivity(l0Sensitivity);
    checkArgument(
        Double.isFinite(lInfSensitivity) && lInfSensitivity > 0,
        "lInfSensitivity must be > 0 and finite. Provided value: %s",
        lInfSensitivity);
  }

  static void checkL0Sensitivity(int l0Sensitivity) {
    checkArgument(
        l0Sensitivity > 0, "l0Sensitivity must be > 0. Provided value: %s", l0Sensitivity);
  }

  static void checkL1Sensitivity(double l1Sensitivity) {
    checkArgument(
        Double.isFinite(l1Sensitivity) && l1Sensitivity > 0,
        "l1Sensitivity must be > 0 and finite. Provided value: %s",
        l1Sensitivity);
  }

  static void checkL2Sensitivity(double l2Sensitivity) {
    checkArgument(
        Double.isFinite(l2Sensitivity) && l2Sensitivity > 0,
        "l2Sensitivity must be > 0 and finite. Provided value: %s",
        l2Sensitivity);
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
    checkArgument(
        isFinite(lower) && isFinite(upper),
        "Lower and upper bounds should be finite. Provided values: lower = %s upper = %s",
        lower,
        upper);
  }

  static void checkBounds(long lower, long upper) {
    checkArgument(
        upper >= lower,
        "The upper bound should be greater than the lower bound. Provided values: "
            + "lower = %s upper = %s",
        lower,
        upper);
  }

  static void checkBoundsNotEqual(double lower, double upper) {
    checkArgument(
        upper != lower,
        "Lower and upper bounds cannot be equal to each other. Provided values: "
            + "lower = %s upper = %s",
        lower,
        upper);
  }

  static void checkMergeDeltaAreEqual(@Nullable Double delta1, double delta2) {
    if (delta1 != null) {
      checkArgument(
          Double.compare(delta1, delta2) == 0,
          "Failed to merge: unequal values of delta. delta1 = %s, delta2 = %s",
          delta1,
          delta2);
    } else {
      checkArgument(
          Double.compare(delta2, 0.0) == 0,
          "Failed to merge: unequal values of delta. delta1 = %s, delta2 = %s",
          delta1,
          delta2);
    }
  }

  static void checkMergeEpsilonAreEqual(double epsilon1, double epsilon2) {
    checkArgument(
        Double.compare(epsilon1, epsilon2) == 0,
        "Failed to merge: unequal values of epsilon. epsilon1 = %s, epsilon2 = %s",
        epsilon1,
        epsilon2);
  }

  static void checkMergeBoundsAreEqual(double lower1, double lower2, double upper1, double upper2) {
    checkArgument(
        Double.compare(lower1, lower2) == 0,
        "Failed to merge: unequal lower bounds. lower1 = %s, lower2 = %s",
        lower1,
        lower2);
    checkArgument(
        Double.compare(upper1, upper2) == 0,
        "Failed to merge: unequal upper bounds. upper1 = %s, upper2 = %s",
        upper1,
        upper2);
  }

  static void checkMergeBoundsAreEqual(long lower1, long lower2, long upper1, long upper2) {
    checkArgument(
        Long.compare(lower1, lower2) == 0,
        "Failed to merge: unequal lower bounds. lower1 = %s, lower2 = %s",
        lower1,
        lower2);
    checkArgument(
        Long.compare(upper1, upper2) == 0,
        "Failed to merge: unequal upper bounds. upper1 = %s, upper2 = %s",
        upper1,
        upper2);
  }

  static void checkMergeMaxContributionsPerPartitionAreEqual(
      int maxContributionsPerPartition1, int maxContributionsPerPartition2) {
    checkArgument(
        maxContributionsPerPartition1 == maxContributionsPerPartition2,
        "Failed to merge: unequal values of maxContributionsPerPartition. "
            + "maxContributionsPerPartition1 = %s, maxContributionsPerPartition2 = %s",
        maxContributionsPerPartition1,
        maxContributionsPerPartition2);
  }

  static void checkMergeMaxPartitionsContributedAreEqual(
      int maxPartitionsContributed1, int maxPartitionsContributed2) {
    checkArgument(
        maxPartitionsContributed1 == maxPartitionsContributed2,
        "Failed to merge: unequal values of maxPartitionsContributed. "
            + "maxPartitionsContributed1 = %s, maxPartitionsContributed2 = %s",
        maxPartitionsContributed1,
        maxPartitionsContributed2);
  }

  static void checkMergePreThresholdAreEqual(int preThreshold1, int preThreshold2) {
    checkArgument(
        preThreshold1 == preThreshold2,
        "Failed to merge: unequal values of preThreshold. "
            + "preThreshold1 = %s, preThreshold2 = %s",
        preThreshold1,
        preThreshold2);
  }

  static void checkMergeMechanismTypesAreEqual(MechanismType type1, MechanismType type2) {
    checkArgument(
        Objects.equals(type1, type2),
        "Failed to merge: unequal mechanism types. type1 = %s, type2 = %s",
        type1,
        type2);
  }

  static void checkAlpha(double alpha) {
    checkArgument(
        0 < alpha && alpha < 1,
        "alpha should be strictly between 0 and 1. Provided value: %s",
        alpha);
  }

  static void checkNoiseComputeQuantileArguments(
      Noise noise,
      double rank,
      int l0Sensitivity,
      double lInfSensitivity,
      double epsilon,
      @Nullable Double delta) {
    checkSensitivities(l0Sensitivity, lInfSensitivity);
    checkEpsilon(epsilon);
    checkNoiseDelta(delta, noise);
    checkArgument(rank > 0 && rank < 1, "rank must be > 0 and < 1. Provided value: %s", rank);
  }

  static void checkPreThreshold(int preThreshold) {
    checkArgument(
        preThreshold >= 1,
        "preThreshold must be greater than or equal to 1. Provided value: %s",
        preThreshold);
  }
}
