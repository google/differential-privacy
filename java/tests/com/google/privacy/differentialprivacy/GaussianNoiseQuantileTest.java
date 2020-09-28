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
public final class GaussianNoiseQuantileTest {
  // Sigma computation is accurate to the 3 most significant digits, TOLERANCE is set accordingly.
  private static final double TOLERANCE = 1E-3;
  private static final GaussianNoise NOISE = new GaussianNoise();
  private static final double LN_3 = Math.log(3);
  private static final double DEFAULT_X = 0.0;
  private static final double DEFAULT_EPSILON = LN_3;
  private static final double DEFAULT_DELTA = 0.00001;
  private static final double DEFAULT_RANK = 1E-5;
  private static final int DEFAULT_L_0_SENSITIVITY = 1;
  private static final double DEFAULT_L_INF_SENSITIVITY = 1.0;
  private static final double[] RANKS = {0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999};

  @Test
  public void computeQuantile_defaultParameters_returnsExactResult() {
    // Exact quantiles rounded to the 8 most significant digits.
    double[] expectedQuantiles = {
      -10.583002, -7.9669561, -2.3098997, 0.0, 2.3098997, 7.9669561, 10.583002
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA);
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
      797.43442, 800.05047, 805.70752, 808.01742, 810.32732, 815.98438, 818.60043
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              /* x= */ 808.017424,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA);
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
      -818.60043, -815.98438, -810.32732, -808.01742, -805.70752, -800.05047, -797.43442
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              /* x= */ -808.017424,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA);
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
      -18.330298, -13.799173, -4.0008636, 0.0, 4.0008636, 13.799173, 18.330298
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              DEFAULT_X,
              /* l0Sensitivity= */ 3,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA);
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
      -7.6726767, -5.7760432, -1.6746773, 0.0, 1.6746773, 5.7760432, 7.6726767
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity= */ 0.725,
              DEFAULT_EPSILON,
              DEFAULT_DELTA);
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
      -753.35364, -567.12973, -164.43078, 0.0, 164.43078, 567.12973, 753.35364
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon= */ 0.01,
              DEFAULT_DELTA);
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
      -5.6644981, -4.2642727, -1.2363620, 0.0, 1.2363620, 4.2642727, 5.6644981
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              2 * DEFAULT_EPSILON,
              DEFAULT_DELTA);
      assertWithMessage(
              "Quantile of rank %s is not precise: actual = %s, expected = %s",
              RANKS[i], actualQuantile, expectedQuantiles[i])
          .that(approxEqual(actualQuantile, expectedQuantiles[i]))
          .isTrue();
    }
  }

  @Test
  public void computeQuantile_smallDelta_returnsExactResult() {
    // Exact quantiles rounded to the 8 most significant digits.
    double[] expectedQuantiles = {
      -16.568238, -12.472682, -3.6162674, 0.0, 3.6162674, 12.472682, 16.568238
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta= */ 1E-10);
      assertWithMessage(
              "Quantile of rank %s is not precise: actual = %s, expected = %s",
              RANKS[i], actualQuantile, expectedQuantiles[i])
          .that(approxEqual(actualQuantile, expectedQuantiles[i]))
          .isTrue();
    }
  }

  @Test
  public void computeQuantile_largeDelta_returnsExactResult() {
    // Exact quantiles rounded to the 8 most significant digits.
    double[] expectedQuantiles = {
      -7.3530635, -5.5354362, -1.6049169, 0.0, 1.6049169, 5.5354362, 7.3530635
    };
    for (int i = 0; i < RANKS.length; i++) {
      double actualQuantile =
          NOISE.computeQuantile(
              RANKS[i],
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta= */ 1E-3);
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
                    DEFAULT_DELTA));
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
                    DEFAULT_DELTA));
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
                    DEFAULT_DELTA));
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
                    DEFAULT_DELTA));
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
                    DEFAULT_DELTA));
    assertThat(e).hasMessageThat().startsWith("epsilon must be");
  }

  @Test
  public void computeQuantile_deltaNull_throwsException() {
    assertThrows(
        NullPointerException.class,
        () ->
            NOISE.computeQuantile(
                DEFAULT_RANK,
                DEFAULT_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                /* delta= */ null));
  }

  @Test
  public void computeQuantile_deltaNegative_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta= */ -0.1));
    assertThat(e).hasMessageThat().startsWith("delta must be");
  }

  @Test
  public void computeQuantile_deltaZero_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta= */ 0.0));
    assertThat(e).hasMessageThat().startsWith("delta must be");
  }

  @Test
  public void computeQuantile_deltaOne_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta= */ 1.0));
    assertThat(e).hasMessageThat().startsWith("delta must be");
  }

  @Test
  public void computeQuantile_deltaGreaterThanOne_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta= */ 2.0));
    assertThat(e).hasMessageThat().startsWith("delta must be");
  }

  @Test
  public void computeQuantile_deltaNaN_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta= */ NaN));
    assertThat(e).hasMessageThat().startsWith("delta must be");
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
                    DEFAULT_DELTA));
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
                    DEFAULT_DELTA));
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
                    DEFAULT_DELTA));
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
                    DEFAULT_DELTA));
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
                    DEFAULT_DELTA));
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
                    DEFAULT_DELTA));
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
                    DEFAULT_DELTA));
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
                    DEFAULT_DELTA));
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
                    DEFAULT_DELTA));
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
                    DEFAULT_DELTA));
    assertThat(e).hasMessageThat().startsWith("l0Sensitivity must be");
  }

  private static boolean approxEqual(double a, double b) {
    double maxMagnitude = max(Math.abs(a), Math.abs(b));
    return Math.abs(a - b) <= TOLERANCE * maxMagnitude;
  }
}
