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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static java.lang.Double.NaN;
import static java.lang.Math.max;
import static org.junit.Assert.assertThrows;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class LaplaceNoiseQuantileTest {
  private static final double TOLERANCE = 1E-7;
  private static final LaplaceNoise NOISE = new LaplaceNoise();
  private static final double LN_3 = Math.log(3);
  private static final double DEFAULT_X = 0.0;
  private static final double DEFAULT_EPSILON = LN_3;
  private static final double DEFAULT_RANK = 1E-5;
  private static final int DEFAULT_L_0_SENSITIVITY = 1;
  private static final double DEFAULT_L_INF_SENSITIVITY = 1.0;
  private static final double[] RANKS = {0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999};

  @Test
  public void computeQuantile_defaultParameters_returnsExactResult() {
    // Exact quantiles rounded to the 8 most significant digits.
    double[] expectedQuantiles = {
      -5.6567801, -3.5608768, -0.63092975, 0.0, 0.63092975, 3.5608768, 5.6567801
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta= */ null);
      assertWithMessage(
              "Quantile of rank %s is not precise: actual = %s, expected = %s",
              RANKS[i], actualQuantile, expectedQuantiles[i])
          .that(approxEqual(actualQuantile, expectedQuantiles[i]))
          .isTrue();
    }
  }

  @Test
  public void computeQuantile_positiveX_returnsExactResult() {
    // Exact quantiles rounded to the 8 most significant digits.
    double[] expectedQuantiles = {
      802.36064, 804.45655, 807.38649, 808.01742, 808.64835, 811.57830, 813.67420
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              /* x= */ 808.017424,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta= */ null);
      assertWithMessage(
              "Quantile of rank %s is not precise: actual = %s, expected = %s",
              RANKS[i], actualQuantile, expectedQuantiles[i])
          .that(approxEqual(actualQuantile, expectedQuantiles[i]))
          .isTrue();
    }
  }

  @Test
  public void computeQuantile_negativeX_returnsExactResult() {
    // Exact quantiles rounded to the 8 most significant digits.
    double[] expectedQuantiles = {
      -813.67420, -811.57830, -808.64835, -808.01742, -807.38649, -804.45655, -802.36064
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              /* x= */ -808.017424,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta= */ null);
      assertWithMessage(
              "Quantile of rank %s is not precise: actual = %s, expected = %s",
              RANKS[i], actualQuantile, expectedQuantiles[i])
          .that(approxEqual(actualQuantile, expectedQuantiles[i]))
          .isTrue();
    }
  }

  @Test
  public void computeQuantile_differentL0Sensitivity_returnsExactResult() {
    // Exact quantiles rounded to the 8 most significant digits.
    double[] expectedQuantiles = {
      -16.970340, -10.682630, -1.8927893, 0.0, 1.8927893, 10.682630, 16.970340
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              DEFAULT_X,
              /* l0Sensitivity= */ 3,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta= */ null);
      assertWithMessage(
              "Quantile of rank %s is not precise: actual = %s, expected = %s",
              RANKS[i], actualQuantile, expectedQuantiles[i])
          .that(approxEqual(actualQuantile, expectedQuantiles[i]))
          .isTrue();
    }
  }

  @Test
  public void computeQuantile_differentLInfSensitivity_returnsExactResult() {
    // Exact quantiles rounded to the 8 most significant digits.
    double[] expectedQuantiles = {
      -4.1011656, -2.5816357, -0.45742407, 0.0, 0.45742407, 2.5816357, 4.1011656
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity= */ 0.725,
              DEFAULT_EPSILON,
              /* delta= */ null);
      assertWithMessage(
              "Quantile of rank %s is not precise: actual = %s, expected = %s",
              RANKS[i], actualQuantile, expectedQuantiles[i])
          .that(approxEqual(actualQuantile, expectedQuantiles[i]))
          .isTrue();
    }
  }

  @Test
  public void computeQuantile_smallEpsilon_returnsExactResult() {
    // Exact quantiles rounded to the 8 most significant digits.
    double[] expectedQuantiles = {
      -621.46081, -391.20230, -69.314718, 0.0, 69.314718, 391.20230, 621.46081
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon= */ 0.01,
              /* delta= */ null);
      assertWithMessage(
              "Quantile of rank %s is not precise: actual = %s, expected = %s",
              RANKS[i], actualQuantile, expectedQuantiles[i])
          .that(approxEqual(actualQuantile, expectedQuantiles[i]))
          .isTrue();
    }
  }

  @Test
  public void computeQuantile_largeEpsilon_returnsExactResult() {
    // Exact quantiles rounded to the 8 most significant digits.
    double[] expectedQuantiles = {
      -2.8283900, -1.7804384, -0.31546488, 0.0, 0.31546488, 1.7804384, 2.8283900
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              2 * DEFAULT_EPSILON,
              /* delta= */ null);
      assertWithMessage(
              "Quantile of rank %s is not precise: actual = %s, expected = %s",
              RANKS[i], actualQuantile, expectedQuantiles[i])
          .that(approxEqual(actualQuantile, expectedQuantiles[i]))
          .isTrue();
    }
  }

  @Test
  public void computeQuantile_epsilonNegative_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    /* epsilon= */ -0.1,
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("epsilon must be");
  }

  @Test
  public void computeQuantile_epsilonZero_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    /* epsilon= */ 0.0,
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("epsilon must be");
  }

  @Test
  public void computeQuantile_epsilonTooSmall_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    /* epsilon= */ 1.0 / (1L << 51),
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("epsilon must be");
  }

  @Test
  public void computeQuantile_epsilonPosInfinity_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    /* epsilon= */ Double.POSITIVE_INFINITY,
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("epsilon must be");
  }

  @Test
  public void computeQuantile_epsilonNan_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    /* epsilon= */ Double.NaN,
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("epsilon must be");
  }

  @Test
  public void computeQuantile_deltaNonnul_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    /* epsilon= */ DEFAULT_EPSILON,
                    /* delta= */ 0.1));
    assertThat(e).hasMessageThat().startsWith("delta should not be set");
  }

  @Test
  public void computeQuantile_rankNegative_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    /* rank= */ -1.0,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("rank must be");
  }

  @Test
  public void computeQuantile_rankZero_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    /* rank= */ 0.0,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("rank must be");
  }

  @Test
  public void computeQuantile_rankOne_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    /* rank= */ 1.0,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("rank must be");
  }

  @Test
  public void computeQuantile_rankGreaterThanOne_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    /* rank= */ 2.0,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("rank must be");
  }

  @Test
  public void computeQuantile_rankNaN_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    /* rank= */ NaN,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("rank must be");
  }

  @Test
  public void computeQuantile_lInfSensitivityNan_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    /* lInfSensitivity= */ Double.NaN,
                    /* epsilon= */ 1.0,
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("lInfSensitivity must be");
  }

  @Test
  public void computeQuantile_lInfSensitivityNegative_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    /* lInfSensitivity= */ -1.0,
                    DEFAULT_EPSILON,
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("lInfSensitivity must be");
  }

  @Test
  public void computeQuantile_lInfSensitivityZero_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    /* lInfSensitivity= */ 0.0,
                    DEFAULT_EPSILON,
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("lInfSensitivity must be");
  }

  @Test
  public void computeQuantile_l0SensitivityNegative_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    /* l0Sensitivity= */ -1,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("l0Sensitivity must be");
  }

  @Test
  public void computeQuantile_l0SensitivityZero_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    /* l0Sensitivity= */ 0,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta= */ null));
    assertThat(e).hasMessageThat().startsWith("l0Sensitivity must be");
  }

  private static boolean approxEqual(double a, double b) {
    double maxMagnitude = max(Math.abs(a), Math.abs(b));
    return Math.abs(a - b) <= TOLERANCE * maxMagnitude;
  }
}
