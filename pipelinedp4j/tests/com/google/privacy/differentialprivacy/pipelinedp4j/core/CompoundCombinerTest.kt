/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.privacy.differentialprivacy.pipelinedp4j.core

import com.google.common.collect.ImmutableList
import com.google.common.truth.extensions.proto.ProtoTruth.assertThat
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.MEAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.VARIANCE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.NoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.ZeroNoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.compoundAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.countAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.dpAggregates
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.meanAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdContributions
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdCountAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.sumAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.varianceAccumulator
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

// TODO: Update tests, they are a little bit outdated in terms of names.
@RunWith(JUnit4::class)
class CompoundCombinerTest {
  private val COUNT_PARAMS =
    AggregationParams(
      metrics = ImmutableList.of(MetricDefinition(COUNT)),
      noiseKind = NoiseKind.GAUSSIAN,
      maxPartitionsContributed = Int.MAX_VALUE,
      maxContributionsPerPartition = Int.MAX_VALUE,
    )
  private val COUNT_AND_SUM_PARAMS =
    AggregationParams(
      metrics = ImmutableList.of(MetricDefinition(COUNT), MetricDefinition(SUM)),
      noiseKind = NoiseKind.GAUSSIAN,
      maxPartitionsContributed = Int.MAX_VALUE,
      maxContributionsPerPartition = Int.MAX_VALUE,
      minTotalValue = -Double.MAX_VALUE,
      maxTotalValue = Double.MAX_VALUE,
    )
  private val COUNT_AND_MEAN_PARAMS =
    AggregationParams(
      metrics = ImmutableList.of(MetricDefinition(COUNT), MetricDefinition(MEAN)),
      noiseKind = NoiseKind.GAUSSIAN,
      maxPartitionsContributed = 100,
      maxContributionsPerPartition = 10,
      minValue = -100.0,
      maxValue = 100.0,
    )
  private val COUNT_AND_VARIANCE_PARAMS =
    AggregationParams(
      metrics = ImmutableList.of(MetricDefinition(COUNT), MetricDefinition(VARIANCE)),
      noiseKind = NoiseKind.GAUSSIAN,
      maxPartitionsContributed = 100,
      maxContributionsPerPartition = 10,
      minValue = -100.0,
      maxValue = 100.0,
    )
  private val UNUSED_ALLOCATED_BUDGET = AllocatedBudget()

  init {
    UNUSED_ALLOCATED_BUDGET.initialize(1.1, 1e-3)
  }

  @Test
  fun createAccumulator_oneMetric_createsOneAccumulator() {
    val compoundCombiner =
      CompoundCombiner(listOf(CountCombiner(COUNT_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory())))

    val accumulator =
      compoundCombiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(10.0, 10.0, 10.0) }
      )

    assertThat(accumulator)
      .isEqualTo(compoundAccumulator { countAccumulator = countAccumulator { count = 3 } })
  }

  @Test
  fun createAccumulator_allMetrics_createsAllAccumulators() {
    val compoundCombiner =
      CompoundCombiner(
        listOf(
          CountCombiner(COUNT_AND_SUM_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory()),
          SumCombiner(COUNT_AND_SUM_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory()),
        )
      )

    val accumulator =
      compoundCombiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(10.0, 10.0, 10.0) }
      )

    assertThat(accumulator)
      .isEqualTo(
        compoundAccumulator {
          countAccumulator = countAccumulator { count = 3 }
          sumAccumulator = sumAccumulator { sum = 30.0 }
        }
      )
  }

  @Test
  fun createAccumulator_meanCombiner_createsMeanAccumulator() {
    val compoundCombiner =
      CompoundCombiner(
        listOf(
          MeanCombiner(
            COUNT_AND_MEAN_PARAMS,
            UNUSED_ALLOCATED_BUDGET,
            UNUSED_ALLOCATED_BUDGET,
            NoiseFactory(),
          )
        )
      )

    val accumulator =
      compoundCombiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(5.0, 10.5, 19.0) }
      )

    assertThat(accumulator)
      .isEqualTo(
        compoundAccumulator {
          meanAccumulator = meanAccumulator {
            count = 3
            normalizedSum = 34.5
          }
        }
      )
  }

  @Test
  fun createAccumulator_varianceCombiner_createsVarianceAccumulator() {
    val compoundCombiner =
      CompoundCombiner(
        listOf(
          VarianceCombiner(
            COUNT_AND_MEAN_PARAMS,
            UNUSED_ALLOCATED_BUDGET,
            UNUSED_ALLOCATED_BUDGET,
            UNUSED_ALLOCATED_BUDGET,
            NoiseFactory(),
          )
        )
      )

    val accumulator =
      compoundCombiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(5.0, 10.5, 19.0) }
      )

    assertThat(accumulator)
      .isEqualTo(
        compoundAccumulator {
          varianceAccumulator = varianceAccumulator {
            count = 3
            normalizedSum = 34.5
            normalizedSumSquares = 496.25
          }
        }
      )
  }

  @Test
  fun mergeAccumulators_multipleMetrics_mergesAccumulators() {
    val compoundCombiner =
      CompoundCombiner(
        listOf(
          CountCombiner(COUNT_AND_SUM_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory()),
          SumCombiner(COUNT_AND_SUM_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory()),
        )
      )

    val mergedAccumulator =
      compoundCombiner.mergeAccumulators(
        compoundAccumulator {
          countAccumulator = countAccumulator { count = 1 }
          sumAccumulator = sumAccumulator { sum = 10.0 }
        },
        compoundAccumulator {
          countAccumulator = countAccumulator { count = 2 }
          sumAccumulator = sumAccumulator { sum = 20.0 }
        },
      )

    assertThat(mergedAccumulator)
      .isEqualTo(
        compoundAccumulator {
          countAccumulator = countAccumulator { count = 3 }
          sumAccumulator = sumAccumulator { sum = 30.0 }
        }
      )
  }

  @Test
  fun mergeAccumulators_oneMetric_mergesAccumulators() {
    val compoundCombiner =
      CompoundCombiner(listOf(CountCombiner(COUNT_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory())))

    val mergedAccumulator =
      compoundCombiner.mergeAccumulators(
        compoundAccumulator { countAccumulator = countAccumulator { count = 1 } },
        compoundAccumulator { countAccumulator = countAccumulator { count = 2 } },
      )

    assertThat(mergedAccumulator)
      .isEqualTo(compoundAccumulator { countAccumulator = countAccumulator { count = 3 } })
  }

  @Test
  fun mergeAccumulators_meanCombiner_mergesAccumulators() {
    val compoundCombiner =
      CompoundCombiner(
        listOf(
          MeanCombiner(
            COUNT_AND_MEAN_PARAMS,
            UNUSED_ALLOCATED_BUDGET,
            UNUSED_ALLOCATED_BUDGET,
            NoiseFactory(),
          )
        )
      )

    val mergedAccumulator =
      compoundCombiner.mergeAccumulators(
        compoundAccumulator {
          meanAccumulator = meanAccumulator {
            count = 1
            normalizedSum = 10.0
          }
        },
        compoundAccumulator {
          meanAccumulator = meanAccumulator {
            count = 2
            normalizedSum = 20.0
          }
        },
      )

    assertThat(mergedAccumulator)
      .isEqualTo(
        compoundAccumulator {
          meanAccumulator = meanAccumulator {
            count = 3
            normalizedSum = 30.0
          }
        }
      )
  }

  @Test
  fun mergeAccumulators_varianceCombiner_mergesAccumulators() {
    val compoundCombiner =
      CompoundCombiner(
        listOf(
          VarianceCombiner(
            COUNT_AND_VARIANCE_PARAMS,
            UNUSED_ALLOCATED_BUDGET,
            UNUSED_ALLOCATED_BUDGET,
            UNUSED_ALLOCATED_BUDGET,
            NoiseFactory(),
          )
        )
      )

    val mergedAccumulator =
      compoundCombiner.mergeAccumulators(
        compoundAccumulator {
          varianceAccumulator = varianceAccumulator {
            count = 1
            normalizedSum = 10.0
            normalizedSumSquares = 100.0
          }
        },
        compoundAccumulator {
          varianceAccumulator = varianceAccumulator {
            count = 2
            normalizedSum = 20.0
            normalizedSumSquares = 200.0
          }
        },
      )

    assertThat(mergedAccumulator)
      .isEqualTo(
        compoundAccumulator {
          varianceAccumulator = varianceAccumulator {
            count = 3
            normalizedSum = 30.0
            normalizedSumSquares = 300.0
          }
        }
      )
  }

  @Test
  fun computeMetrics_allMetrics_returnsAllMetrics() {
    val compoundCombiner =
      CompoundCombiner(
        listOf(
          CountCombiner(COUNT_AND_SUM_PARAMS, UNUSED_ALLOCATED_BUDGET, ZeroNoiseFactory()),
          SumCombiner(COUNT_AND_SUM_PARAMS, UNUSED_ALLOCATED_BUDGET, ZeroNoiseFactory()),
        )
      )

    val dpAggregates =
      compoundCombiner.computeMetrics(
        compoundAccumulator {
          countAccumulator = countAccumulator { count = 3 }
          sumAccumulator = sumAccumulator { sum = 30.0 }
        }
      )

    assertThat(dpAggregates)
      .isEqualTo(
        dpAggregates {
          count = 3.0
          sum = 30.0
        }
      )
  }

  @Test
  fun computeMetrics_oneMetric_returnsOneMetric() {
    val compoundCombiner =
      CompoundCombiner(
        listOf(CountCombiner(COUNT_PARAMS, UNUSED_ALLOCATED_BUDGET, ZeroNoiseFactory()))
      )

    val dpAggregates =
      compoundCombiner.computeMetrics(
        compoundAccumulator { countAccumulator = countAccumulator { count = 3 } }
      )

    assertThat(dpAggregates).isEqualTo(dpAggregates { count = 3.0 })
  }

  @Test
  fun computeMetrics_meanCombiner_returnsMeanMetric() {
    val compoundCombiner =
      CompoundCombiner(
        listOf(
          MeanCombiner(
            COUNT_AND_MEAN_PARAMS.copy(metrics = ImmutableList.of(MetricDefinition(MEAN))),
            UNUSED_ALLOCATED_BUDGET,
            UNUSED_ALLOCATED_BUDGET,
            ZeroNoiseFactory(),
          )
        )
      )

    val dpAggregates =
      compoundCombiner.computeMetrics(
        compoundAccumulator {
          meanAccumulator = meanAccumulator {
            count = 3
            normalizedSum = 30.0
          }
        }
      )

    assertThat(dpAggregates).isEqualTo(dpAggregates { mean = 10.0 })
  }

  @Test
  fun computeMetrics_meanCombiner_returnsCountSumMean() {
    val compoundCombiner =
      CompoundCombiner(
        listOf(
          MeanCombiner(
            COUNT_AND_MEAN_PARAMS.copy(
              metrics =
                ImmutableList.of(
                  MetricDefinition(MEAN),
                  MetricDefinition(COUNT),
                  MetricDefinition(SUM),
                )
            ),
            UNUSED_ALLOCATED_BUDGET,
            UNUSED_ALLOCATED_BUDGET,
            ZeroNoiseFactory(),
          )
        )
      )

    val dpAggregates =
      compoundCombiner.computeMetrics(
        compoundAccumulator {
          meanAccumulator = meanAccumulator {
            count = 3
            normalizedSum = 30.0
          }
        }
      )

    assertThat(dpAggregates)
      .isEqualTo(
        dpAggregates {
          count = 3.0
          sum = 30.0
          mean = 10.0
        }
      )
  }

  @Test
  fun computeMetrics_varianceCombiner_returnsVarianceMetric() {
    val compoundCombiner =
      CompoundCombiner(
        listOf(
          VarianceCombiner(
            COUNT_AND_VARIANCE_PARAMS.copy(metrics = ImmutableList.of(MetricDefinition(VARIANCE))),
            UNUSED_ALLOCATED_BUDGET,
            UNUSED_ALLOCATED_BUDGET,
            UNUSED_ALLOCATED_BUDGET,
            ZeroNoiseFactory(),
          )
        )
      )

    val dpAggregates =
      compoundCombiner.computeMetrics(
        compoundAccumulator {
          varianceAccumulator = varianceAccumulator {
            count = 10
            normalizedSum = 120.0
            normalizedSumSquares = 1500.0
          }
        }
      )

    assertThat(dpAggregates).isEqualTo(dpAggregates { variance = 6.0 })
  }

  @Test
  fun computeMetrics_varianceCombiner_returnsCountSumMeanVariance() {
    val compoundCombiner =
      CompoundCombiner(
        listOf(
          VarianceCombiner(
            COUNT_AND_VARIANCE_PARAMS.copy(
              metrics =
                ImmutableList.of(
                  MetricDefinition(MEAN),
                  MetricDefinition(COUNT),
                  MetricDefinition(SUM),
                  MetricDefinition(VARIANCE),
                )
            ),
            UNUSED_ALLOCATED_BUDGET,
            UNUSED_ALLOCATED_BUDGET,
            UNUSED_ALLOCATED_BUDGET,
            ZeroNoiseFactory(),
          )
        )
      )

    val dpAggregates =
      compoundCombiner.computeMetrics(
        compoundAccumulator {
          varianceAccumulator = varianceAccumulator {
            count = 10
            normalizedSum = 120.0
            normalizedSumSquares = 1500.0
          }
        }
      )

    assertThat(dpAggregates)
      .isEqualTo(
        dpAggregates {
          count = 10.0
          sum = 120.0
          mean = 12.0
          variance = 6.0
        }
      )
  }

  @Test
  fun createAccumulator_exactPrivacyIdCountCombiner_createsAccumulator() {
    val compoundCombiner = CompoundCombiner(listOf(ExactPrivacyIdCountCombiner()))

    val accumulator =
      compoundCombiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(10.0, 5.0) }
      )

    assertThat(accumulator)
      .isEqualTo(
        compoundAccumulator { privacyIdCountAccumulator = privacyIdCountAccumulator { count = 1 } }
      )
  }

  @Test
  fun mergeAccumulator_exactPrivacyIdCountCombiner_mergesAccumulators() {
    val compoundCombiner = CompoundCombiner(listOf(ExactPrivacyIdCountCombiner()))

    val mergedAccumulator =
      compoundCombiner.mergeAccumulators(
        compoundAccumulator { privacyIdCountAccumulator = privacyIdCountAccumulator { count = 4 } },
        compoundAccumulator { privacyIdCountAccumulator = privacyIdCountAccumulator { count = 5 } },
      )

    assertThat(mergedAccumulator)
      .isEqualTo(
        compoundAccumulator { privacyIdCountAccumulator = privacyIdCountAccumulator { count = 9 } }
      )
  }

  @Test
  fun computeMetrics_exactPrivacyIdCountCombiner_returnsEmptyMetrics() {
    val compoundCombiner = CompoundCombiner(listOf(ExactPrivacyIdCountCombiner()))

    val dpAggregates =
      compoundCombiner.computeMetrics(
        compoundAccumulator { privacyIdCountAccumulator = privacyIdCountAccumulator { count = 3 } }
      )

    assertThat(dpAggregates).isEqualTo(dpAggregates {}) // No Exact Privacy Id metric
  }
}
