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

import static java.lang.Double.NaN;
import static java.lang.Double.POSITIVE_INFINITY;
import static org.junit.Assert.assertThrows;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the validations performed by {@link Count#builder()}. */
@RunWith(JUnit4.class)
public class CountBuilderTest {

  private static final double EPSILON = 0.123;
  private static final double DELTA = 0.123;

  private Count.Params.Builder builder;

  @Before
  public void setup() {
    builder =
        Count.builder()
            .delta(DELTA)
            .epsilon(EPSILON)
            .noise(new GaussianNoise())
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
  public void deltaGaussian_Nan_throwsException() {
    builder.delta(NaN);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaGaussian_notSet_throwsException() {
    Count.Params.Builder builder =
        Count.builder().noise(new GaussianNoise()).epsilon(EPSILON).maxPartitionsContributed(1);
    assertThrows(NullPointerException.class, builder::build);
  }

  @Test
  public void deltaLaplace_nonnull_throwsException() {
    builder.delta(0.1);
    builder.noise(new LaplaceNoise());
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaLaplace_null_noException() {
    builder.delta(null);
    builder.noise(new LaplaceNoise());
    builder.build();
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
    Count.builder().epsilon(EPSILON).maxPartitionsContributed(1).build();
  }

  @Test
  public void noise_null_throwsException() {
    assertThrows(NullPointerException.class, () -> Count.builder().noise(null));
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
    Count.builder().epsilon(EPSILON).maxPartitionsContributed(1).noise(new LaplaceNoise()).build();
  }
}
