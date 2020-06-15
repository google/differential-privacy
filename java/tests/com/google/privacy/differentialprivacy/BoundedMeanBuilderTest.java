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

/** Tests validations done by {@link BoundedMean#builder()}. */
@RunWith(JUnit4.class)
public class BoundedMeanBuilderTest {
  private static final double EPSILON = 0.123;
  private static final double DELTA = 0.123;

  private BoundedMean.Params.Builder builder;

  @Before
  public void setup() {
    builder =
        BoundedMean.builder()
            .delta(DELTA)
            .epsilon(EPSILON)
            .noise(new GaussianNoise())
            .lower(0)
            .upper(1)
            .maxContributionsPerPartition(1)
            .maxPartitionsContributed(1);
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
  public void deltaGaussian_notProvided_throwsException() {
    BoundedMean.Params.Builder builder =
        BoundedMean.builder()
            .noise(new GaussianNoise())
            .epsilon(EPSILON)
            .lower(0)
            .upper(1)
            .maxContributionsPerPartition(1)
            .maxPartitionsContributed(1);
    assertThrows(NullPointerException.class, builder::build);
  }

  @Test
  public void maxContributionsPerPartition_notProvided_throwsException() {
    BoundedMean.Params.Builder builder =
        BoundedMean.builder()
            .noise(new GaussianNoise())
            .epsilon(EPSILON)
            .delta(DELTA)
            .maxPartitionsContributed(1)
            .lower(0)
            .upper(1);
    assertThrows(IllegalStateException.class, builder::build);
  }

  @Test
  public void deltaGaussian_Nan_throwsException() {
    builder.delta(NaN);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void maxContributionsPerPartition_belowZero_throwsException() {
    builder.maxContributionsPerPartition(-1);
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
  public void bounds_lower_negInfinity_throwsException() {
    builder.lower(NEGATIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void bounds_lower_posInfinity_throwsException() {
    builder.lower(POSITIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void bounds_lower_Nan_throwsException() {
    builder.lower(NaN);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void bounds_upper_negInfinity_throwsException() {
    builder.upper(NEGATIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void bounds_upper_posInfinity_throwsException() {
    builder.upper(POSITIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void bounds_upper_Nan_throwsException() {
    builder.lower(NaN);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  // Lower and upper bounds can be equal to each other.
  @Test
  public void bounds_equal_noException() {
    builder.lower(1);
    builder.upper(1);
    assertThat(builder.build()).isNotNull();
  }

  @Test
  public void l0Sensitivity_belowZero_throwsException() {
    builder.maxPartitionsContributed(-1);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void l0Sensitivity_zero_throwsException() {
    builder.maxPartitionsContributed(0);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void noise_notProvided_noException() {
    // No exception is thrown if Noise instance is not provided because a noise
    // instance will be defaulted.
    BoundedMean.builder()
        .epsilon(EPSILON)
        .lower(0)
        .upper(1)
        .maxPartitionsContributed(1)
        .maxContributionsPerPartition(1)
        .build();
  }

  @Test
  public void noise_null_throwsException() {
    assertThrows(NullPointerException.class, () -> BoundedMean.builder().noise(null));
  }
}
