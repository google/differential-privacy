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
import static java.lang.Double.NaN;
import static java.lang.Double.POSITIVE_INFINITY;
import static org.junit.Assert.assertThrows;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests validations done by {@link Count#builder()}. */
@RunWith(JUnit4.class)
public class CountBuilderTest {
  private static final double DEFAULT_EPSILON = 0.5;
  private static final double DEFAULT_DELTA = 0.00001;
  private static final int DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION = 1;
  private static final int DEFAULT_MAX_PARTITIONS_CONTRIBUTED = 1;

  private Count.Params.Builder builder;

  @Before
  public void setup() {
    builder =
        Count.builder()
            .epsilon(DEFAULT_EPSILON)
            .delta(DEFAULT_DELTA)
            .noise(new GaussianNoise())
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .maxPartitionsContributed(DEFAULT_MAX_PARTITIONS_CONTRIBUTED);
  }

  @Test
  public void defaultParameters_buildsInstance() {
    assertThat(builder.build()).isNotNull();
  }

  @Test
  public void epsilon_lessThanZero_throwsException() {
    builder.epsilon(-1.0);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void epsilon_zero_throwsException() {
    builder.epsilon(0.0);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void epsilon_posInfinity_throwsException() {
    builder.epsilon(POSITIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void epsilon_nan_throwsException() {
    builder.epsilon(NaN);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void epsilon_notProvided_throwsException() {
    Count.Params.Builder builder =
        Count.builder()
            .delta(DEFAULT_DELTA)
            .noise(new GaussianNoise())
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .maxPartitionsContributed(DEFAULT_MAX_PARTITIONS_CONTRIBUTED);
    assertThrows(IllegalStateException.class, builder::build);
  }

  @Test
  public void deltaGaussian_lessThanZero_throwsException() {
    builder.delta(-1.0);
    builder.noise(new GaussianNoise());
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaGaussian_zero_throwsException() {
    builder.delta(0.0);
    builder.noise(new GaussianNoise());
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaGaussian_greaterThanOne_throwsException() {
    builder.delta(50.0);
    builder.noise(new GaussianNoise());
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaGaussian_one_throwsException() {
    builder.delta(1.0);
    builder.noise(new GaussianNoise());
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaGaussian_nan_throwsException() {
    builder.delta(NaN);
    builder.noise(new GaussianNoise());
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaGaussian_notProvided_throwsException() {
    Count.Params.Builder builder =
        Count.builder()
            .epsilon(DEFAULT_EPSILON)
            .noise(new GaussianNoise())
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .maxPartitionsContributed(DEFAULT_MAX_PARTITIONS_CONTRIBUTED);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaLaplace_notNull_throwsException() {
    builder.delta(DEFAULT_DELTA);
    builder.noise(new LaplaceNoise());
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaLaplace_zero_buildsInstance() {
    builder.noise(new LaplaceNoise());
    builder.delta(0.0);
    assertThat(builder.build()).isNotNull();
  }

  @Test
  public void deltaLaplace_notProvided_buildsInstance() {
    Count.Params.Builder builder =
        Count.builder()
            .epsilon(DEFAULT_EPSILON)
            .noise(new LaplaceNoise())
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .maxPartitionsContributed(DEFAULT_MAX_PARTITIONS_CONTRIBUTED);
    assertThat(builder.build()).isNotNull();
  }

  @Test
  public void maxContributionsPerPartition_lessThanZero_throwsException() {
    builder.maxContributionsPerPartition(-1);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void maxContributionsPerPartition_zero_throwsException() {
    builder.maxContributionsPerPartition(0);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void maxContributionsPerPartition_notProvided_buildsInstance() {
    // No exception is thrown because max contributions per partition is 1 by default.
    Count.Params.Builder builder =
        Count.builder()
            .epsilon(DEFAULT_EPSILON)
            .delta(DEFAULT_DELTA)
            .noise(new GaussianNoise())
            .maxPartitionsContributed(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION);
    assertThat(builder.build()).isNotNull();
  }

  @Test
  public void maxPartitionsContributed_lessThanZero_throwsException() {
    builder.maxPartitionsContributed(-1);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void maxPartitionsContributed_zero_throwsException() {
    builder.maxPartitionsContributed(0);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void maxPartitionsContributed_notProvided_throwsException() {
    Count.Params.Builder builder =
        Count.builder()
            .epsilon(DEFAULT_EPSILON)
            .delta(DEFAULT_DELTA)
            .noise(new GaussianNoise())
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION);
    assertThrows(IllegalStateException.class, builder::build);
  }

  @Test
  public void noise_null_throwsException() {
    assertThrows(NullPointerException.class, () -> builder.noise(null));
  }

  @Test
  public void noise_notProvided_delta_notProvided_buildsInstance() {
    // No exception is thrown because the noise parameter will be set to a default Laplace instance.
    Count.Params.Builder builder =
        Count.builder()
            .epsilon(DEFAULT_EPSILON)
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .maxPartitionsContributed(DEFAULT_MAX_PARTITIONS_CONTRIBUTED);
    assertThat(builder.build()).isNotNull();
  }
}
