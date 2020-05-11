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
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

//import com.google.testing.mockito.Mocks;
import java.util.Collection;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/**
 * Tests behavior of {@link Count}. The test mocks a {@link Noise} instance to always
 * generate zero noise. Testing the statistical and DP properties of the algorithm are out of scope.
 */
@RunWith(JUnit4.class)
public class CountTest {

  private static final double EPSILON = 0.123;
  private static final double DELTA = 0.123;

  @Mock private Noise noise;
  @Mock private Collection<Double> hugeCollection;
  @Rule public final MockitoRule mocks = MockitoJUnit.rule();

  private Count count;

  @Before
  public void setUp() {
    // Mock the noise mechanism so that it does not add any noise.
    when(noise.addNoise(anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> invocation.getArguments()[0]);

    count =
        Count.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(1)
            .build();

    when(hugeCollection.size()).thenReturn(Integer.MAX_VALUE);
  }

  @Test
  public void increment() {
    int entriesCount = 10000;
    for (int i = 0; i < entriesCount; ++i) {
      count.increment();
    }

    assertThat(count.computeResult()).isEqualTo(entriesCount);
  }

  @Test
  public void incrementBy() {
    count.incrementBy(9);

    assertThat(count.computeResult()).isEqualTo(9);
  }

  @Test
  public void incrementAndIncrementByAllTogether() {
    count.increment();
    count.incrementBy(7);
    count.increment();

    assertThat(count.computeResult()).isEqualTo(9);
  }

  @Test
  public void computeResult_noIncrements_returnsZero() {
    assertThat(count.computeResult()).isEqualTo(0);
  }

  // An attempt to compute the count several times should result in an exception.
  @Test
  public void computeResult_multipleCalls_throwsException() {
    count.increment();

    count.computeResult();
    assertThrows(IllegalStateException.class, count::computeResult);
  }

  @Test
  public void incrementBy_hugeValues_dontOverflow() {
    count.incrementBy(Integer.MAX_VALUE);
    count.incrementBy(Integer.MAX_VALUE);
    count.increment();
    count.increment();
    count.increment();

    long expected = Integer.MAX_VALUE * 2L + 3L;
    long actualResult = count.computeResult();
    assertThat(actualResult).isEqualTo(expected);
  }

  @Test
  public void computeResult_callsNoiseCorrectly() {
    int l0Sensitivity = 1;
    count =
        Count.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(l0Sensitivity)
            .maxContributionsPerPartition(5)
            .build();
    count.increment();
    count.computeResult();

    verify(noise)
        .addNoise(
            eq(1L), // count of added entries = 1
            eq(l0Sensitivity),
            eq(/* lInfSensitivity = maxContributionsPerPartition = 5 */ 5L),
            eq(EPSILON),
            eq(DELTA));
  }

  @Test
  public void computeResult_addsNoise() {
    // Mock the noise mechanism so that it always generates 100.0.
    when(noise.addNoise(anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> (long) invocation.getArguments()[0] + 100);
    count =
        Count.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(1)
            .build();

    count.increment();
    assertThat(count.computeResult()).isEqualTo(101); // value (1) + noise (100) = 101
  }
}
