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
import static org.mockito.Mockito.when;

import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Tests validations done by {@link BoundedSum#builder()}. */
@RunWith(JUnit4.class)
public class BoundedSumBuilderTest {
  private static final double DEFAULT_EPSILON = 0.5;
  private static final double DEFAULT_DELTA = 1e-5;
  private static final int DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION = 1;
  private static final int DEFAULT_MAX_PARTITIONS_CONTRIBUTED = 1;
  private static final double DEFAULT_LOWER = 0.0;
  private static final double DEFAULT_UPPER = 1.0;

  private BoundedSum.Params.Builder builder;

  @Mock private Noise unrecognizedNoise;

  @Rule public final MockitoRule mocks = MockitoJUnit.rule();

  @Before
  public void setup() {
    builder =
        BoundedSum.builder()
            .epsilon(DEFAULT_EPSILON)
            .delta(DEFAULT_DELTA)
            .noise(new GaussianNoise())
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .maxPartitionsContributed(DEFAULT_MAX_PARTITIONS_CONTRIBUTED)
            .lower(DEFAULT_LOWER)
            .upper(DEFAULT_UPPER);

    when(unrecognizedNoise.getMechanismType()).thenReturn(MechanismType.EMPTY);
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
    BoundedSum.Params.Builder builder =
        BoundedSum.builder()
            .delta(DEFAULT_DELTA)
            .noise(new GaussianNoise())
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .maxPartitionsContributed(DEFAULT_MAX_PARTITIONS_CONTRIBUTED)
            .lower(DEFAULT_LOWER)
            .upper(DEFAULT_UPPER);
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
    BoundedSum.Params.Builder builder =
        BoundedSum.builder()
            .epsilon(DEFAULT_EPSILON)
            .noise(new GaussianNoise())
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .maxPartitionsContributed(DEFAULT_MAX_PARTITIONS_CONTRIBUTED)
            .lower(DEFAULT_LOWER)
            .upper(DEFAULT_UPPER);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaLaplace_notZero_throwsException() {
    builder.delta(DEFAULT_DELTA);
    builder.noise(new LaplaceNoise());
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaLaplace_zero_buildsInstance() {
    builder.delta(0.0);
    builder.noise(new LaplaceNoise());
    assertThat(builder.build()).isNotNull();
  }

  @Test
  public void deltaUnrecognizedNoise_lessThanZero_throwsException() {
    builder.delta(-1.0);
    builder.noise(unrecognizedNoise);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaUnrecognizedNoise_zero_buildsInstance() {
    builder.delta(0.0);
    builder.noise(unrecognizedNoise);
    assertThat(builder.build()).isNotNull();
  }

  @Test
  public void deltaUnrecognizedNoise_one_throwsException() {
    builder.delta(1.0);
    builder.noise(unrecognizedNoise);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaUnrecognizedNoise_greaterThanOne_throwsException() {
    builder.delta(2.0);
    builder.noise(unrecognizedNoise);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaUnrecognizedNoise_nan_throwsException() {
    builder.delta(NaN);
    builder.noise(unrecognizedNoise);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void deltaUnrecognizedNoise_betweenZeroAndOne_buildsInstance() {
    builder.delta(DEFAULT_DELTA);
    builder.noise(unrecognizedNoise);
    assertThat(builder.build()).isNotNull();
  }

  @Test
  public void deltaUnrecognizedNoise_notProvided_buildsInstance() {
    BoundedSum.Params.Builder builder =
        BoundedSum.builder()
            .epsilon(DEFAULT_EPSILON)
            .noise(unrecognizedNoise)
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .maxPartitionsContributed(DEFAULT_MAX_PARTITIONS_CONTRIBUTED)
            .lower(DEFAULT_LOWER)
            .upper(DEFAULT_UPPER);
    assertThat(builder.build()).isNotNull();
  }

  @Test
  public void deltaLaplace_notProvided_buildsInstance() {
    BoundedSum.Params.Builder builder =
        BoundedSum.builder()
            .epsilon(DEFAULT_EPSILON)
            .noise(new LaplaceNoise())
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .maxPartitionsContributed(DEFAULT_MAX_PARTITIONS_CONTRIBUTED)
            .lower(DEFAULT_LOWER)
            .upper(DEFAULT_UPPER);
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
    BoundedSum.Params.Builder builder =
        BoundedSum.builder()
            .epsilon(DEFAULT_EPSILON)
            .delta(DEFAULT_DELTA)
            .noise(new GaussianNoise())
            .maxPartitionsContributed(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .lower(DEFAULT_LOWER)
            .upper(DEFAULT_UPPER);
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
    BoundedSum.Params.Builder builder =
        BoundedSum.builder()
            .epsilon(DEFAULT_EPSILON)
            .delta(DEFAULT_DELTA)
            .noise(new GaussianNoise())
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .lower(DEFAULT_LOWER)
            .upper(DEFAULT_UPPER);
    assertThrows(IllegalStateException.class, builder::build);
  }

  @Test
  public void lower_negInfinity_throwsException() {
    builder.lower(NEGATIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void lower_posInfinity_throwsException() {
    builder.lower(POSITIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void lower_nan_throwsException() {
    builder.lower(NaN);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void lower_notProvided_throwsException() {
    BoundedSum.Params.Builder builder =
        BoundedSum.builder()
            .epsilon(DEFAULT_EPSILON)
            .delta(DEFAULT_DELTA)
            .noise(new GaussianNoise())
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .maxPartitionsContributed(DEFAULT_MAX_PARTITIONS_CONTRIBUTED)
            .upper(DEFAULT_UPPER);
    assertThrows(IllegalStateException.class, builder::build);
  }

  @Test
  public void upper_negInfinity_throwsException() {
    builder.upper(NEGATIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void upper_posInfinity_throwsException() {
    builder.upper(POSITIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void upper_nan_throwsException() {
    builder.lower(NaN);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void upper_notProvided_throwsException() {
    BoundedSum.Params.Builder builder =
        BoundedSum.builder()
            .epsilon(DEFAULT_EPSILON)
            .delta(DEFAULT_DELTA)
            .noise(new GaussianNoise())
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .maxPartitionsContributed(DEFAULT_MAX_PARTITIONS_CONTRIBUTED)
            .lower(DEFAULT_LOWER);
    assertThrows(IllegalStateException.class, builder::build);
  }

  @Test
  public void bounds_lowerGreaterThanUpper_throwsException() {
    builder.lower(1.0);
    builder.upper(0.0);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void bounds_equal_buildsInstance() {
    builder.lower(1.0);
    builder.upper(1.0);
    assertThat(builder.build()).isNotNull();
  }

  @Test
  public void noise_null_throwsException() {
    assertThrows(NullPointerException.class, () -> builder.noise(null));
  }

  @Test
  public void noise_notProvided_delta_notProvided_buildsInstance() {
    // No exception is thrown because the noise parameter will be set to a default Laplace instance.
    BoundedSum.Params.Builder builder =
        BoundedSum.builder()
            .epsilon(DEFAULT_EPSILON)
            .maxContributionsPerPartition(DEFAULT_MAX_CONTRIBUTIONS_PER_PARTITION)
            .maxPartitionsContributed(DEFAULT_MAX_PARTITIONS_CONTRIBUTED)
            .lower(DEFAULT_LOWER)
            .upper(DEFAULT_UPPER);
    assertThat(builder.build()).isNotNull();
  }

  @Test
  public void parametersResultInL1SensitivityOverflow_throwsException() {
    BoundedSum.Params.Builder builder =
        BoundedSum.builder()
            .epsilon(DEFAULT_EPSILON)
            .noise(new LaplaceNoise())
            .maxContributionsPerPartition(1)
            .maxPartitionsContributed(4)
            .lower(-Double.MAX_VALUE / 2.0)
            .upper(Double.MAX_VALUE / 2.0);
    //   l_1 sensitivity (of the numerator of the mean)
    // = maxContributionsPerPartition * maxPartitionsContributed * |lower - upper| / 2
    // = 1 * 4 * Double.MAX_VALUE / 2
    // = 2 * Double.MAX_VALUE
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void parametersResultInL2SensitivityOverflow_throwsException() {
    BoundedSum.Params.Builder builder =
        BoundedSum.builder()
            .epsilon(DEFAULT_EPSILON)
            .delta(DEFAULT_DELTA)
            .noise(new GaussianNoise())
            .maxContributionsPerPartition(1)
            .maxPartitionsContributed(5)
            .lower(-Double.MAX_VALUE / 2.0)
            .upper(Double.MAX_VALUE / 2.0);
    //   l_2 sensitivity (of the numerator of the mean)
    // = maxContributionsPerPartition * maxPartitionsContributed^0.5 * |lower - upper| / 2
    // = 1 * 5^0.5 * Double.MAX_VALUE / 2
    // = (5/4)^0.5 * Double.MAX_VALUE
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void parametersResultInL1SensitivityOverflow_unrecognizedNoise_buildsInstance() {
    BoundedSum.Params.Builder builder =
        BoundedSum.builder()
            .epsilon(DEFAULT_EPSILON)
            .noise(unrecognizedNoise)
            .maxContributionsPerPartition(1)
            .maxPartitionsContributed(4)
            .lower(-Double.MAX_VALUE / 2.0)
            .upper(Double.MAX_VALUE / 2.0);
    //   l_1 sensitivity (of the numerator of the mean)
    // = maxContributionsPerPartition * maxPartitionsContributed * |lower - upper| / 2
    // = 1 * 4 * Double.MAX_VALUE / 2
    // = 2 * Double.MAX_VALUE
    // But the instance is still built because the noise type isn't recognized.
    assertThat(builder.build()).isNotNull();
  }

  @Test
  public void parametersResultInL2SensitivityOverflow_unrecognizedNoise_buildsInstance() {
    BoundedSum.Params.Builder builder =
        BoundedSum.builder()
            .epsilon(DEFAULT_EPSILON)
            .noise(unrecognizedNoise)
            .maxContributionsPerPartition(1)
            .maxPartitionsContributed(5)
            .lower(-Double.MAX_VALUE / 2.0)
            .upper(Double.MAX_VALUE / 2.0);
    //   l_2 sensitivity (of the numerator of the mean)
    // = maxContributionsPerPartition * maxPartitionsContributed^0.5 * |lower - upper| / 2
    // = 1 * 5^0.5 * Double.MAX_VALUE / 2
    // = (5/4)^0.5 * Double.MAX_VALUE
    // But the instance is still built because the noise type isn't recognized.
    assertThat(builder.build()).isNotNull();
  }
}
