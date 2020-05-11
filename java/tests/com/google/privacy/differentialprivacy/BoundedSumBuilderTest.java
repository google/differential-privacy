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
import static java.lang.Double.NEGATIVE_INFINITY;
import static java.lang.Double.NaN;
import static java.lang.Double.POSITIVE_INFINITY;
import static org.junit.Assert.assertThrows;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the validations performed by {@link BoundedSum#builder()}. */
@RunWith(JUnit4.class)
public class BoundedSumBuilderTest {
  private static final double EPSILON = 0.123;
  private static final double DELTA = 0.123;

  private BoundedSum.Params.Builder builder;

  @Before
  public void setup() {
    builder =
        BoundedSum.builder()
            .delta(DELTA)
            .epsilon(EPSILON)
            .noise(new GaussianNoise())
            .lower(0)
            .upper(1)
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1);
  }

  @Test
  public void epsilon_belowZero_throwsException() {
    builder.epsilon(-1);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void epsilon_zero_throwsException() {
    builder.epsilon(0);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void epsilon_posInfinity_throwsException() {
    builder.epsilon(POSITIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void epsilon_Nan_throwsException() {
    builder.epsilon(NaN);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaGaussian_belowZero_throwsException() {
    builder.delta(-1.0);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaGaussian_greaterThanOne_throwsException() {
    builder.delta(50.0);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaGaussian_zero_throwsException() {
    builder.delta(0.0);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaGaussian_null_throwsException() {
    builder.delta(null);
    assertThrows(NullPointerException.class, builder::build);
  }

  @Test
  public void deltaGaussian_notSet_throwsException() {
    BoundedSum.Params.Builder builder =
        BoundedSum.builder()
            .noise(new GaussianNoise())
            .epsilon(EPSILON)
            .lower(0)
            .upper(1)
            .maxPartitionsContributed(1);
    assertThrows(NullPointerException.class, builder::build);
  }

  @Test
  public void deltaGaussian_Nan_throwsException() {
    builder.delta(NaN);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaLaplace_nonnull_throwsException() {
    builder.noise(new LaplaceNoise());
    builder.delta(0.1);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void bounds_lowerGreaterThanUpper_throwsException() {
    builder.lower(1);
    builder.upper(0);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void bounds_lowerNegInfinity_throwsException() {
    builder.lower(NEGATIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void bounds_lowerPosInfinity_throwsException() {
    builder.lower(POSITIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void bounds_lowerNan_throwsException() {
    builder.lower(NaN);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void bounds_upperNegInfinity_throwsException() {
    builder.upper(NEGATIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void bounds_upperPosInfinity_throwsException() {
    builder.upper(POSITIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void bounds_upperNan_throwsException() {
    builder.upper(NaN);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  // It should be possible for lower and upper bounds to be equal.
  @Test
  public void bounds_equal_noException() {
    builder.lower(1);
    builder.upper(1);
    assertThat(builder.build()).isNotNull();
  }

  @Test
  public void maxPartitionsContributed_belowZero_throwsException() {
    builder.maxPartitionsContributed(-1);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void maxPartitionsContributed_zero_throwsException() {
    builder.maxPartitionsContributed(0);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void noise_notProvided_noException() {
    // No exception should be thrown if no Noise instance is provided because the builder
    // should automatically provide a default instance of Noise.
    BoundedSum.builder().epsilon(EPSILON).lower(0).upper(1).maxPartitionsContributed(1).build();
  }

  @Test
  public void noise_null_throwsException() {
    assertThrows(NullPointerException.class, () -> BoundedSum.builder().noise(null));
  }

  @Test
  public void maxContributionsPerPartition_belowZero_throwsException() {
    builder.maxContributionsPerPartition(-1);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void maxContributionsPerPartition_zero_throwsException() {
    builder.maxContributionsPerPartition(0);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void maxContributionsPerPartition_aboveZero_noException() {
    builder.maxContributionsPerPartition(5).build();
  }

  @Test
  public void maxContributionsPerPartition_notProvided_noException() {
    // No exception should be thrown if maxContributionsPerPartition is not provided, because
    // the builder should automatically provide a default value.
    BoundedSum.builder()
        .epsilon(EPSILON)
        .lower(0)
        .upper(1)
        .maxPartitionsContributed(1)
        .noise(new LaplaceNoise())
        .build();
  }

  @Test
  public void macContributionsPerPartitionAndUpper_tooHigh_throwsException() {
    builder.lower(1).upper(Double.MAX_VALUE).maxContributionsPerPartition(2);

    // An exception should be thrown because lInfSensitivity should overflow.
    // More precisely: lInfSensitivity = max(abs(lower), abs(upper)) * maxContributionsPerPartition
    // = max(abs(1), abs(Double.MAX_VALUE)) * 2 = Double.MAX_VALUE * 2 => Double overflow.
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void maxContributionsPerPartitionAndLower_tooHigh_throwsException() {
    builder.lower(-Double.MAX_VALUE).upper(1).maxContributionsPerPartition(2);

    // An exception shouldd be thrown because LInfSensitivity should overflow.
    // More precisely: LInfSensitivity = max(abs(lower), abs(upper)) * maxContributionsPerPartition
    // = max(abs(-Double.MAX_VALUE), abs(1)) * 2 = Double.MAX_VALUE * 2 => Double overflow.
    assertThrows(IllegalArgumentException.class, builder::build);
  }
}
