//
// Copyright 2021 Google LLC
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
import static com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType.GAUSSIAN;
import static com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType.LAPLACE;
import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.BoundedQuantilesSummary;
import com.google.protobuf.InvalidProtocolBufferException;
import java.lang.reflect.Method;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.invocation.Invocation;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/**
 * Tests the accuracy of {@link BoundedQuantiles}. The test mocks {@link Noise} instance which
 * generates 0 noise. Statistical and DP properties of the algorithm are out of scope of the test.
 */
@RunWith(JUnit4.class)
public class BoundedQuantilesTest {

  // Epsilon, delta and the contribution bounds have no effect on the mocked noise so their choice
  // is arbitrary.
  private static final double ARBITRARY_EPSILON = 0.5;
  private static final double ARBITRARY_DELTA = 0.00001;
  private static final int ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION = 5;
  private static final int ARBITRARY_MAX_PARTITIONS_CONTRIBUTED = 12;
  private static final double ARBITRARY_LOWER = -2.68545;
  private static final double ARBITRARY_UPPER = 2.68545;
  private static final int ARBITRARY_TREE_HEIGHT = 10;
  private static final int ARBITRARY_BRANCHING_FACTOR = 3;
  private static final double ARBITRARY_RANK = 0.54321;

  @Rule public final MockitoRule mocks = MockitoJUnit.rule();

  private List<Double> entries;
  private List<Double> ranks;
  @Mock private Noise noise;

  private BoundedQuantiles.Params.Builder builder;

  @Before
  public void setUp() {
    ranks = new ArrayList<>();
    for (int i = 0; i <= 1000; i++) {
      ranks.add(i / 1000.0);
    }

    // Generate a dataset with arbitrary values between -25.0 and 25.0.
    SecureRandom random = new SecureRandom();
    entries = new ArrayList<>(1001); // one entry per rank
    for (int i = 0; i < 1001; i++) {
      entries.add(random.nextDouble() * 50.0 - 25);
    }

    // Mock a noise mechanism that adds no noise.
    when(noise.addNoise(anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> invocation.getArguments()[0]);
    // Tests that use serialization need to access to the type of the noise they use. Because the
    // tests don't rely on a specific noise type, we arbitrarily return Gaussian.
    when(noise.getMechanismType()).thenReturn(GAUSSIAN);

    builder =
        BoundedQuantiles.builder()
            .noise(noise)
            .epsilon(ARBITRARY_EPSILON)
            .delta(ARBITRARY_DELTA)
            .maxContributionsPerPartition(ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION)
            .maxPartitionsContributed(ARBITRARY_MAX_PARTITIONS_CONTRIBUTED)
            // Lower and upper bounds are large enough so that no input value used in this test will
            // be clamped.
            .lower(-50.0)
            .upper(50.0);
  }

  @Test
  public void computeResult_resultApproximatesTrueQuantile() {
    // When no noise is added, computeResult should return a value that differs from the true
    // quantile by no more than the size of the buckets the range is partitioned into, i.e.,
    // (upper - lower) / branchingFactor^treeHeight.
    BoundedQuantiles quantiles =
        builder.lower(-50.0).upper(50.0).treeHeight(4).branchingFactor(10).build();
    double tolerance = 0.01; // > (upper - lower) / branchingFactor^treeHeight

    quantiles.addEntries(entries);

    Collections.sort(entries);
    for (double rank : ranks) {
      double trueQuantile =
          entries.get((int) Math.round((entries.size() - 1) * BoundedQuantiles.adjustRank(rank)));
      double result = quantiles.computeResult(rank);
      assertThat(abs(result - trueQuantile)).isLessThan(tolerance);
    }
  }

  @Test
  public void computeResult_callsNoiseCorrectly() throws NoSuchMethodException {
    BoundedQuantiles quantiles =
        builder
            .epsilon(0.1)
            .delta(0.00001)
            .maxContributionsPerPartition(5)
            .maxPartitionsContributed(7)
            .treeHeight(3)
            .build();

    quantiles.addEntry(0.0);
    quantiles.computeResult(0.123);

    // A privacy unit can contribute at most treeHeight * maxContributionsPerPartition times to a
    // node of the internal quantile tree data structure. Morover, the privacy unit can contribute
    // to at most maxPartitionsContributed quantile trees in total. Thus, the resulting l_1
    // sensitivity is
    //    s_1: treeHeight * maxContributionsPerPartition * maxPartitionsContributed
    //
    // Considering that a privacy unit can contribute at most maxContributionsPerPartition times
    // to any particular node, furthermore yields a l_2 sensitivity of:
    //    s_2: maxContributionsPerPartition * sqrt(treeHeight * maxPartitionsContributed)
    //
    // Setting the l_0 and l_inf sensitivitiies to
    //    s_0: treeHeight * maxPartitionsContributed = 3 * 7 = 21
    //    s_inf: maxContributionsPerPartition = 5
    // respects both of these bounds.

    int addNoiseInvocationCount = 0;
    Method addNoiseMethod =
        Noise.class.getMethod(
            "addNoise", double.class, int.class, double.class, double.class, double.class);
    for (Invocation invoc : Mockito.mockingDetails(noise).getInvocations()) {
      if (invoc.getMethod().equals(addNoiseMethod)) {
        addNoiseInvocationCount++;
      }
    }
    verify(noise, times(addNoiseInvocationCount))
        .addNoise(
            /* x: depends on the count of the noised tree node, could be any value*/ anyDouble(),
            /*l0Sensitivity: treeHeight * maxPartitionsContributed*/ eq(21),
            /*lInfSensitivity: maxContributionsPerPartition*/ eq(5.0),
            /*epsilon: value should be passed on from input*/ eq(0.1),
            /*delta: value should be passed on from input*/ eq(0.00001));
  }

  @Test
  public void computeResult_emptySetOfEntries_resultIsLinearlyDistributed() {
    // When invoked on an empty data set, the bounded quantile mechanism should return values that
    // match a linear distribution. This is an arbitrary specification considering that no standard
    // definition of quantiles exists for empty data sets.
    BoundedQuantiles quantiles =
        builder.lower(-50.0).upper(50.0).treeHeight(4).branchingFactor(10).build();
    double tolerance = 0.01; // > (upper - lower) / branchingFactor^treeHeight

    for (double rank : ranks) {
      double quantileOfLinearDistribution =
          -50.0 * (1.0 - BoundedQuantiles.adjustRank(rank))
              + 50.0 * BoundedQuantiles.adjustRank(rank);
      double result = quantiles.computeResult(rank);
      assertThat(abs(result - quantileOfLinearDistribution)).isLessThan(tolerance);
    }
  }

  @Test
  public void computeResult_entriesSmallerThanLowerBound_resultGreaterOrEqualToLowerBound() {
    BoundedQuantiles quantiles = builder.lower(-1.0).build();

    for (int i = 0; i < 1000; i++) {
      quantiles.addEntry(-100.0);
    }

    for (double rank : ranks) {
      assertThat(quantiles.computeResult(rank)).isGreaterThan(-1.0);
    }
  }

  @Test
  public void computeResult_entriesGreaterThanUpperBound_resultLessOrEqualToUpperBound() {
    BoundedQuantiles quantiles = builder.upper(1.0).build();

    for (int i = 0; i < 1000; i++) {
      quantiles.addEntry(100.0);
    }

    for (double rank : ranks) {
      assertThat(quantiles.computeResult(rank)).isLessThan(1.0);
    }
  }

  @Test
  public void computeResult_invariantToPreClamping() {
    BoundedQuantiles quantiles1 = builder.lower(-5.0).upper(5.0).build();
    BoundedQuantiles quantiles2 = builder.build();

    for (double entry : entries) {
      quantiles1.addEntry(entry);
      quantiles2.addEntry(min(max(-5.0, entry), 5.0));
    }

    for (double rank : ranks) {
      assertThat(quantiles1.computeResult(rank)).isEqualTo(quantiles2.computeResult(rank));
    }
  }

  @Test
  public void computeResult_invariantToEntryOrder() {
    BoundedQuantiles quantiles1 = builder.build();
    BoundedQuantiles quantiles2 = builder.build();

    // The list of entries contains 1001 elements. However, we only add the first 997. The reason
    // is that 997 is a prime number, which allows us to shuffle the entires easily using modular
    // arithmetic.
    for (int i = 0; i < 997; i++) {
      quantiles1.addEntry(entries.get(i));
      // Adding entries with an arbitrary step length of 643. Because the two values are coprime,
      // all entries between 0 and 997 will be added.
      quantiles2.addEntry(entries.get((i * 643) % 997));
    }

    for (double rank : ranks) {
      assertThat(quantiles1.computeResult(rank)).isEqualTo(quantiles2.computeResult(rank));
    }
  }

  @Test
  public void computeResult_invariantToAddingEntriesInBulk() {
    BoundedQuantiles quantiles1 = builder.build();
    BoundedQuantiles quantiles2 = builder.build();

    quantiles1.addEntries(entries);
    for (double entry : entries) {
      quantiles2.addEntry(entry);
    }

    for (double rank : ranks) {
      assertThat(quantiles1.computeResult(rank)).isEqualTo(quantiles2.computeResult(rank));
    }
  }

  @Test
  public void computeResult_calledTwiceForTheSameRank_returnsSameResult() {
    // This property should hold even if noise is added.
    BoundedQuantiles quantiles = builder.noise(new GaussianNoise()).build();

    quantiles.addEntries(entries);

    for (double rank : ranks) {
      assertThat(quantiles.computeResult(rank)).isEqualTo(quantiles.computeResult(rank));
    }
  }

  @Test
  public void computeResult_increasesMonotonicallyAsRankIncreases() {
    // This property should hold even if noise is added.
    BoundedQuantiles quantiles = builder.noise(new GaussianNoise()).build();

    quantiles.addEntries(entries);

    double lastResult = Double.NEGATIVE_INFINITY;
    for (double rank = 0.0; rank <= 1.0; rank += 0.001) {
      assertThat(quantiles.computeResult(rank)).isAtLeast(lastResult);
      lastResult = quantiles.computeResult(rank);
    }
  }

  @Test
  public void computeResult_rankLessThanZero_throwsException() {
    BoundedQuantiles quantiles = builder.build();
    assertThrows(IllegalArgumentException.class, () -> quantiles.computeResult(-0.1));
  }

  @Test
  public void computeResult_rankGreaterThanOne_throwsException() {
    BoundedQuantiles quantiles = builder.build();
    assertThrows(IllegalArgumentException.class, () -> quantiles.computeResult(1.1));
  }

  @Test
  public void computeResult_calledAfterSerialize_throwsException() {
    BoundedQuantiles quantiles = builder.build();
    quantiles.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> quantiles.computeResult(ARBITRARY_RANK));
  }

  @Test
  public void addEntry_ignoresNanValues() {
    // The result should not be affected by entries of value Nan.
    BoundedQuantiles quantiles1 = builder.build();
    BoundedQuantiles quantiles2 = builder.build();

    for (double entry : Arrays.asList(10.0, 20.0, 3.0)) {
      quantiles1.addEntry(entry);
      quantiles2.addEntry(entry);
      quantiles2.addEntry(Double.NaN);
    }

    for (double rank : ranks) {
      assertThat(quantiles1.computeResult(rank)).isEqualTo(quantiles2.computeResult(rank));
    }
  }

  @Test
  public void addEntry_calledAfterComputeResult_throwsException() {
    BoundedQuantiles quantiles = builder.build();

    quantiles.computeResult(ARBITRARY_RANK);

    assertThrows(IllegalStateException.class, () -> quantiles.addEntry(0.0));
  }

  @Test
  public void addEntry_calledAfterSerialize_throwsException() {
    BoundedQuantiles quantiles = builder.build();

    quantiles.getSerializableSummary();

    assertThrows(IllegalStateException.class, () -> quantiles.addEntry(0.0));
  }

  @Test
  public void addEntries_ignoresNanValues() {
    // The result should not be affected by entries of value Nan.
    BoundedQuantiles quantiles1 = builder.build();
    BoundedQuantiles quantiles2 = builder.build();

    quantiles1.addEntries(Arrays.asList(10.0, 20.0, 3.0));
    quantiles2.addEntries(Arrays.asList(10.0, Double.NaN, 20.0, Double.NaN, 3.0, Double.NaN));

    for (double rank : ranks) {
      assertThat(quantiles1.computeResult(rank)).isEqualTo(quantiles2.computeResult(rank));
    }
  }

  @Test
  public void addEntries_accpetsEmptyCollections() {
    // The result should be the same as if addEntries had not been invoked.
    BoundedQuantiles quantiles1 = builder.build();
    BoundedQuantiles quantiles2 = builder.build();

    quantiles1.addEntries(new ArrayList<Double>());

    for (double rank : ranks) {
      assertThat(quantiles1.computeResult(rank)).isEqualTo(quantiles2.computeResult(rank));
    }
  }

  @Test
  public void addEntries_calledAfterComputeResult_throwsException() {
    BoundedQuantiles quantiles = builder.build();

    quantiles.computeResult(ARBITRARY_RANK);

    assertThrows(IllegalStateException.class, () -> quantiles.addEntry(0.0));
  }

  @Test
  public void addEntries_calledAfterSerialize_throwsException() {
    BoundedQuantiles quantiles = builder.build();

    quantiles.getSerializableSummary();

    assertThrows(IllegalStateException.class, () -> quantiles.addEntry(0.0));
  }

  @Test
  public void getSerializableSummary_copiesQuantileTreeCorrectly() {
    BoundedQuantiles quantiles =
        builder.lower(-0.5).upper(8.5).treeHeight(2).branchingFactor(3).build();
    quantiles.addEntry(-100.0); // maps to node 1 and 4
    quantiles.addEntry(-0.5); // maps to node 1 and 4
    quantiles.addEntry(0.4999); // maps to node 1 and 4
    quantiles.addEntry(0.5001); // maps to node 1 and 5
    quantiles.addEntry(3.4999); // maps to node 2 and 7
    quantiles.addEntry(3.5001); // maps to node 2 and 8
    quantiles.addEntry(4.0); // maps to node 2 and 8
    quantiles.addEntry(8.5); // maps to node 3 and 12
    quantiles.addEntry(100.0); // maps to node 3 and 12

    BoundedQuantilesSummary summary = getSummary(quantiles);

    assertThat(summary.getQuantileTreeOrDefault(/*node index*/ 1, /*default*/ -1))
        .isEqualTo(/*expected node count*/ 4);
    assertThat(summary.getQuantileTreeOrDefault(/*node index*/ 2, /*default*/ -1))
        .isEqualTo(/*expected node count*/ 3);
    assertThat(summary.getQuantileTreeOrDefault(/*node index*/ 3, /*default*/ -1))
        .isEqualTo(/*expected node count*/ 2);
    assertThat(summary.getQuantileTreeOrDefault(/*node index*/ 4, /*default*/ -1))
        .isEqualTo(/*expected node count*/ 3);
    assertThat(summary.getQuantileTreeOrDefault(/*node index*/ 5, /*default*/ -1))
        .isEqualTo(/*expected node count*/ 1);
    assertThat(summary.getQuantileTreeOrDefault(/*node index*/ 7, /*default*/ -1))
        .isEqualTo(/*expected node count*/ 1);
    assertThat(summary.getQuantileTreeOrDefault(/*node index*/ 8, /*default*/ -1))
        .isEqualTo(/*expected node count*/ 2);
    assertThat(summary.getQuantileTreeOrDefault(/*node index*/ 12, /*default*/ -1))
        .isEqualTo(/*expected node count*/ 2);

    // a total of 8 nodes should have a count greater than 0
    assertThat(summary.getQuantileTreeCount()).isEqualTo(8);
  }

  @Test
  public void getSerializableSummary_copiesEmptyQuantileTreeCorrectly() {
    BoundedQuantiles quantiles = builder.build();
    BoundedQuantilesSummary summary = getSummary(quantiles);
    assertThat(summary.getQuantileTreeCount()).isEqualTo(0);
  }

  @Test
  public void getSerializableSummary_calledAfterComputeResult_throwsException() {
    BoundedQuantiles quantiles = builder.build();
    quantiles.computeResult(ARBITRARY_RANK);
    assertThrows(IllegalStateException.class, quantiles::getSerializableSummary);
  }

  @Test
  public void getSerializableSummary_multipleCalls_returnsSameSummary() {
    BoundedQuantiles quantiles =
        BoundedQuantiles.builder()
            .epsilon(1.0)
            .noise(new LaplaceNoise())
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1)
            .lower(0.0)
            .upper(1.0)
            .build();
    quantiles.addEntry(0.5);
    byte[] summary1 = quantiles.getSerializableSummary();
    byte[] summary2 = quantiles.getSerializableSummary();
    assertThat(summary1).isEqualTo(summary2);
  }

  @Test
  public void getSerializableSummary_copiesEpsilonCorrectly() {
    BoundedQuantiles quantiles = builder.epsilon(ARBITRARY_EPSILON).build();
    BoundedQuantilesSummary summary = getSummary(quantiles);
    assertThat(summary.getEpsilon()).isEqualTo(ARBITRARY_EPSILON);
  }

  @Test
  public void getSerializableSummary_copiesDeltaCorrectly() {
    BoundedQuantiles quantiles = builder.delta(ARBITRARY_DELTA).build();
    BoundedQuantilesSummary summary = getSummary(quantiles);
    assertThat(summary.getDelta()).isEqualTo(ARBITRARY_DELTA);
  }

  @Test
  public void getSerializableSummary_copiesGaussianNoiseCorrectly() {
    BoundedQuantiles quantiles = builder.noise(new GaussianNoise()).build();
    BoundedQuantilesSummary summary = getSummary(quantiles);
    assertThat(summary.getMechanismType()).isEqualTo(GAUSSIAN);
  }

  @Test
  public void getSerializableSummary_copiesLaplaceNoiseCorrectly() {
    BoundedQuantiles quantiles = builder.noise(new LaplaceNoise()).delta(0.0).build();
    BoundedQuantilesSummary summary = getSummary(quantiles);
    assertThat(summary.getMechanismType()).isEqualTo(LAPLACE);
  }

  @Test
  public void getSerializableSummary_copiesMaxPartitionsContributedCorrectly() {
    BoundedQuantiles quantiles =
        builder.maxPartitionsContributed(ARBITRARY_MAX_PARTITIONS_CONTRIBUTED).build();
    BoundedQuantilesSummary summary = getSummary(quantiles);
    assertThat(summary.getMaxPartitionsContributed())
        .isEqualTo(ARBITRARY_MAX_PARTITIONS_CONTRIBUTED);
  }

  @Test
  public void getSerializableSummary_copiesMaxContributionsPerPartitionCorrectly() {
    BoundedQuantiles quantiles =
        builder.maxContributionsPerPartition(ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION).build();
    BoundedQuantilesSummary summary = getSummary(quantiles);
    assertThat(summary.getMaxContributionsPerPartition())
        .isEqualTo(ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION);
  }

  @Test
  public void getSerializableSummary_copiesLowerCorrectly() {
    BoundedQuantiles quantiles = builder.lower(ARBITRARY_LOWER).build();
    BoundedQuantilesSummary summary = getSummary(quantiles);
    assertThat(summary.getLower()).isEqualTo(ARBITRARY_LOWER);
  }

  @Test
  public void getSerializableSummary_copiesUpperCorrectly() {
    BoundedQuantiles quantiles = builder.upper(ARBITRARY_UPPER).build();
    BoundedQuantilesSummary summary = getSummary(quantiles);
    assertThat(summary.getUpper()).isEqualTo(ARBITRARY_UPPER);
  }

  @Test
  public void getSerializableSummary_copiesTreeHeightCorrectly() {
    BoundedQuantiles quantiles = builder.treeHeight(ARBITRARY_TREE_HEIGHT).build();
    BoundedQuantilesSummary summary = getSummary(quantiles);
    assertThat(summary.getTreeHeight()).isEqualTo(ARBITRARY_TREE_HEIGHT);
  }

  @Test
  public void getSerializableSummary_copiesBranchingFactorCorrectly() {
    BoundedQuantiles quantiles = builder.branchingFactor(ARBITRARY_BRANCHING_FACTOR).build();
    BoundedQuantilesSummary summary = getSummary(quantiles);
    assertThat(summary.getBranchingFactor()).isEqualTo(ARBITRARY_BRANCHING_FACTOR);
  }

  @Test
  public void mergeWith_distributedComputationMatchesCentralizedComputation() {
    BoundedQuantiles distributedQuantiles1 = builder.build();
    BoundedQuantiles distributedQuantiles2 = builder.build();
    BoundedQuantiles distributedQuantiles3 = builder.build();
    BoundedQuantiles centralizedQuantiles = builder.build();

    for (int i = 0; i < entries.size(); i++) {
      if (i % 3 == 0) {
        distributedQuantiles1.addEntry(entries.get(i));
      } else if (i % 3 == 1) {
        distributedQuantiles2.addEntry(entries.get(i));
      } else {
        distributedQuantiles3.addEntry(entries.get(i));
      }
    }
    centralizedQuantiles.addEntries(entries);

    distributedQuantiles1.mergeWith(distributedQuantiles2.getSerializableSummary());
    distributedQuantiles1.mergeWith(distributedQuantiles3.getSerializableSummary());

    for (double rank : ranks) {
      assertThat(distributedQuantiles1.computeResult(rank))
          .isEqualTo(centralizedQuantiles.computeResult(rank));
    }
  }

  @Test
  public void mergeWith_epsilonMismatch_throwsException() {
    BoundedQuantiles quantiles1 = builder.epsilon(ARBITRARY_EPSILON).build();
    BoundedQuantiles quantiles2 = builder.epsilon(ARBITRARY_EPSILON + 1.0).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> quantiles1.mergeWith(quantiles2.getSerializableSummary()));
  }

  @Test
  public void mergeWith_nullDelta_mergesWithoutException() {
    BoundedQuantiles quantiles1 = builder.noise(new LaplaceNoise()).delta(0.0).build();
    BoundedQuantiles quantiles2 = builder.noise(new LaplaceNoise()).delta(0.0).build();
    // No exception should be thrown.
    quantiles1.mergeWith(quantiles2.getSerializableSummary());
  }

  @Test
  public void mergeWith_deltaMismatch_throwsException() {
    BoundedQuantiles quantiles1 = builder.delta(ARBITRARY_DELTA).build();
    BoundedQuantiles quantiles2 = builder.delta(ARBITRARY_DELTA + 0.00001).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> quantiles1.mergeWith(quantiles2.getSerializableSummary()));
  }

  @Test
  public void mergeWith_noiseMismatch_throwsException() {
    BoundedQuantiles quantiles1 = builder.noise(new GaussianNoise()).build();
    BoundedQuantiles quantiles2 = builder.noise(new LaplaceNoise()).delta(0.0).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> quantiles1.mergeWith(quantiles2.getSerializableSummary()));
  }

  @Test
  public void mergeWith_maxPartitionsContributedMismatch_throwsException() {
    BoundedQuantiles quantiles1 =
        builder.maxPartitionsContributed(ARBITRARY_MAX_PARTITIONS_CONTRIBUTED).build();
    BoundedQuantiles quantiles2 =
        builder.maxPartitionsContributed(ARBITRARY_MAX_PARTITIONS_CONTRIBUTED + 1).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> quantiles1.mergeWith(quantiles2.getSerializableSummary()));
  }

  @Test
  public void mergeWith_maxContributionsPerPartitionMismatch_throwsException() {
    BoundedQuantiles quantiles1 =
        builder.maxContributionsPerPartition(ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION).build();
    BoundedQuantiles quantiles2 =
        builder.maxContributionsPerPartition(ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION + 1).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> quantiles1.mergeWith(quantiles2.getSerializableSummary()));
  }

  @Test
  public void mergeWith_lowerBoundsMismatch_throwsException() {
    BoundedQuantiles quantiles1 = builder.lower(ARBITRARY_LOWER).build();
    BoundedQuantiles quantiles2 = builder.lower(ARBITRARY_LOWER + 1.0).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> quantiles1.mergeWith(quantiles2.getSerializableSummary()));
  }

  @Test
  public void mergeWith_upperBoundsMismatch_throwsException() {
    BoundedQuantiles quantiles1 = builder.upper(ARBITRARY_UPPER).build();
    BoundedQuantiles quantiles2 = builder.upper(ARBITRARY_UPPER + 1.0).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> quantiles1.mergeWith(quantiles2.getSerializableSummary()));
  }

  @Test
  public void mergeWith_treeHeightMismatch_throwsException() {
    BoundedQuantiles quantiles1 = builder.treeHeight(ARBITRARY_TREE_HEIGHT).build();
    BoundedQuantiles quantiles2 = builder.treeHeight(ARBITRARY_TREE_HEIGHT + 1).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> quantiles1.mergeWith(quantiles2.getSerializableSummary()));
  }

  @Test
  public void mergeWith_branchingFactorMismatch_throwsException() {
    BoundedQuantiles quantiles1 = builder.treeHeight(ARBITRARY_BRANCHING_FACTOR).build();
    BoundedQuantiles quantiles2 = builder.treeHeight(ARBITRARY_BRANCHING_FACTOR + 1).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> quantiles1.mergeWith(quantiles2.getSerializableSummary()));
  }

  @Test
  public void mergeWith_calledAfterComputeResult_throwsException() {
    BoundedQuantiles quantiles1 = builder.build();
    BoundedQuantiles quantiles2 = builder.build();

    quantiles1.computeResult(ARBITRARY_RANK);
    byte[] summary = quantiles2.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> quantiles1.mergeWith(summary));
  }

  @Test
  public void mergeWith_calledAfterSerialization_throwsException() {
    BoundedQuantiles quantiles1 = builder.build();
    BoundedQuantiles quantiles2 = builder.build();

    quantiles1.getSerializableSummary();
    byte[] summary = quantiles2.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> quantiles1.mergeWith(summary));
  }

  /**
   * Note that {@link BoundedQuantilesSummary} isn't visible to the actual clients, who only see an
   * opaque {@code byte[]} blob. Here, we parse said blob to perform whitebox testing, to verify
   * some expectations of the blob's content. We do this because achieving good coverage with pure
   * behaviour testing (i.e., blackbox testing) isn't possible.
   */
  private static BoundedQuantilesSummary getSummary(BoundedQuantiles quantiles) {
    byte[] nonParsedSummary = quantiles.getSerializableSummary();
    try {
      return BoundedQuantilesSummary.parseFrom(nonParsedSummary);
    } catch (InvalidProtocolBufferException pbe) {
      throw new IllegalArgumentException(pbe);
    }
  }
}
