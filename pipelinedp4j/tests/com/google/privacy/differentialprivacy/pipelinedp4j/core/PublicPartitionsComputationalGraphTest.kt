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
import com.google.common.truth.Truth.assertThat
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.PRIVACY_ID_COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.PartitionsBalance.UNKNOWN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.ZeroNoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalEncoderFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalTable
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.DpAggregates
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.dpAggregates
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import com.google.testing.junit.testparameterinjector.TestParameters
import com.google.testing.junit.testparameterinjector.TestParametersValuesProvider
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(TestParameterInjector::class)
class PublicPartitionsComputationalGraphTest {
  companion object {
    private val COUNT_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(COUNT)),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = Int.MAX_VALUE,
        maxContributionsPerPartition = Int.MAX_VALUE,
      )
    private val PRIVACY_ID_COUNT_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(PRIVACY_ID_COUNT)),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = Int.MAX_VALUE,
      )
    private val SUM_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(SUM)),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = Int.MAX_VALUE,
        minTotalValue = -Double.MAX_VALUE,
        maxTotalValue = Double.MAX_VALUE,
      )
    private val COUNT_SUM_AND_ID_COUNT_PARAMS =
      AggregationParams(
        metrics =
          ImmutableList.of(
            MetricDefinition(COUNT),
            MetricDefinition(SUM),
            MetricDefinition(PRIVACY_ID_COUNT),
          ),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 100,
        maxContributionsPerPartition = 100,
        minTotalValue = -100.0,
        maxTotalValue = 100.0,
      )
    private val ALLOCATED_BUDGET = AllocatedBudget()

    init {
      ALLOCATED_BUDGET.initialize(1.1, 1e-3)
    }

    private val LOCAL_EF = LocalEncoderFactory()
    private val COUNT_SUM_AND_ID_COUNT_COMBINER =
      CompoundCombiner(
        listOf(
          CountCombiner(COUNT_SUM_AND_ID_COUNT_PARAMS, ALLOCATED_BUDGET, ZeroNoiseFactory()),
          SumCombiner(COUNT_SUM_AND_ID_COUNT_PARAMS, ALLOCATED_BUDGET, ZeroNoiseFactory()),
          PrivacyIdCountCombiner(
            COUNT_SUM_AND_ID_COUNT_PARAMS,
            ALLOCATED_BUDGET,
            ZeroNoiseFactory(),
          ),
        )
      )
  }

  @Test
  fun aggregate_appliesPublicPartitions() {
    val inputData =
      LocalCollection(
        sequenceOf(
          TestDataRow("Alice", "public_present_in_data", 1.0),
          TestDataRow("Alice", "not_public", 1.0),
        )
      )
    val publicPartitions =
      LocalCollection(sequenceOf("public_present_in_data", "public_not_present_in_data"))
    val computationalGraph =
      PublicPartitionsComputationalGraph(
        PartitionSampler(
          maxPartitionsContributed = 5,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        ),
        publicPartitions,
        PartitionsBalance.UNKNOWN,
        COUNT_SUM_AND_ID_COUNT_COMBINER,
        testDataExtractors,
        LOCAL_EF,
      )

    val dpAggregates = computationalGraph.aggregate(inputData) as LocalTable<String, DpAggregates>

    assertThat(dpAggregates.data.toMap().keys)
      .containsExactly("public_present_in_data", "public_not_present_in_data")
    // Check that the value corresponding to the public partition not present in data is noisy
    assertThat(dpAggregates.data.toMap().get("public_present_in_data")!!.count).isNotEqualTo(0.0)
    assertThat(dpAggregates.data.toMap().get("public_present_in_data")!!.sum).isNotEqualTo(0.0)
    assertThat(dpAggregates.data.toMap().get("public_present_in_data")!!.privacyIdCount)
      .isNotEqualTo(0.0)
  }

  // The test uses a value provider instead of a canonical enum because an enum must be immutable,
  // and it cannot be immutable because it stores a CompoundCombiner, which is not immutable.
  class MetricsTestValueProvider : TestParametersValuesProvider() {
    override fun provideValues(context: Context): MutableList<TestParameters.TestParametersValues> {
      return ImmutableList.of(
        TestParameters.TestParametersValues.builder()
          .name("COUNT")
          .addParameter("params", COUNT_PARAMS)
          .addParameter(
            "combiner",
            CompoundCombiner(
              listOf(CountCombiner(COUNT_PARAMS, ALLOCATED_BUDGET, ZeroNoiseFactory()))
            ),
          )
          .addParameter(
            "inputData",
            sequenceOf(
              TestDataRow("Alice", "red", 10.0),
              TestDataRow("Alice", "red", 10.0),
              TestDataRow("Alice", "green", 10.0),
              TestDataRow("Bob", "green", 10.0),
              TestDataRow("Bob", "blue", 10.0),
            ),
          )
          .addParameter("publicPartitions", sequenceOf("red", "green", "blue"))
          .addParameter(
            "expectedResult",
            arrayOf(
              Pair("red", dpAggregates { count = 2.0 }),
              Pair("green", dpAggregates { count = 2.0 }),
              Pair("blue", dpAggregates { count = 1.0 }),
            ),
          )
          .build(),
        TestParameters.TestParametersValues.builder()
          .name("SUM")
          .addParameter("params", SUM_PARAMS)
          .addParameter(
            "combiner",
            CompoundCombiner(listOf(SumCombiner(SUM_PARAMS, ALLOCATED_BUDGET, ZeroNoiseFactory()))),
          )
          .addParameter(
            "inputData",
            sequenceOf(
              TestDataRow("Alice", "red", 10.0),
              TestDataRow("Alice", "red", 10.0),
              TestDataRow("Alice", "green", 10.0),
              TestDataRow("Bob", "green", 10.0),
              TestDataRow("Bob", "blue", 10.0),
            ),
          )
          .addParameter("publicPartitions", sequenceOf("red", "green", "blue"))
          .addParameter(
            "expectedResult",
            arrayOf(
              Pair("red", dpAggregates { sum = 20.0 }),
              Pair("green", dpAggregates { sum = 20.0 }),
              Pair("blue", dpAggregates { sum = 10.0 }),
            ),
          )
          .build(),
        TestParameters.TestParametersValues.builder()
          .name("PRIVACY_ID_COUNT")
          .addParameter("params", PRIVACY_ID_COUNT_PARAMS)
          .addParameter(
            "combiner",
            CompoundCombiner(
              listOf(
                PrivacyIdCountCombiner(
                  PRIVACY_ID_COUNT_PARAMS,
                  ALLOCATED_BUDGET,
                  ZeroNoiseFactory(),
                )
              )
            ),
          )
          .addParameter(
            "inputData",
            sequenceOf(
              TestDataRow("Alice", "red", 10.0),
              TestDataRow("Alice", "red", 10.0),
              TestDataRow("Alice", "green", 10.0),
              TestDataRow("Bob", "green", 10.0),
              TestDataRow("Bob", "blue", 10.0),
            ),
          )
          .addParameter("publicPartitions", sequenceOf("red", "green", "blue"))
          .addParameter(
            "expectedResult",
            arrayOf(
              Pair("red", dpAggregates { privacyIdCount = 1.0 }),
              Pair("green", dpAggregates { privacyIdCount = 2.0 }),
              Pair("blue", dpAggregates { privacyIdCount = 1.0 }),
            ),
          )
          .build(),
        TestParameters.TestParametersValues.builder()
          .name("COUNT_SUM_PRIVACY_ID_COUNT")
          .addParameter("params", COUNT_SUM_AND_ID_COUNT_PARAMS)
          .addParameter("combiner", COUNT_SUM_AND_ID_COUNT_COMBINER)
          .addParameter(
            "inputData",
            sequenceOf(
              TestDataRow("Alice", "red", 10.0),
              TestDataRow("Alice", "red", 10.0),
              TestDataRow("Alice", "green", 10.0),
              TestDataRow("Bob", "green", 10.0),
              TestDataRow("Bob", "blue", 10.0),
            ),
          )
          .addParameter("publicPartitions", sequenceOf("red", "green", "blue"))
          .addParameter(
            "expectedResult",
            arrayOf(
              Pair(
                "red",
                dpAggregates {
                  count = 2.0
                  sum = 20.0
                  privacyIdCount = 1.0
                },
              ),
              Pair(
                "green",
                dpAggregates {
                  count = 2.0
                  sum = 20.0
                  privacyIdCount = 2.0
                },
              ),
              Pair(
                "blue",
                dpAggregates {
                  count = 1.0
                  sum = 10.0
                  privacyIdCount = 1.0
                },
              ),
            ),
          )
          .build(),
      )
    }
  }

  @Test
  @TestParameters(valuesProvider = MetricsTestValueProvider::class)
  fun aggregate_computesMetricsDefinedInCombiner(
    params: AggregationParams,
    combiner: CompoundCombiner,
    inputData: Sequence<TestDataRow>,
    publicPartitions: Sequence<String>,
    expectedResult: Array<Pair<String, DpAggregates>>,
  ) {
    val computationalGraph =
      PublicPartitionsComputationalGraph(
        PartitionSampler(
          params.maxPartitionsContributed!!,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        ),
        LocalCollection(publicPartitions),
        PartitionsBalance.UNKNOWN,
        combiner,
        testDataExtractors,
        LOCAL_EF,
      )

    val dpAggregates =
      computationalGraph.aggregate(LocalCollection(inputData)) as LocalTable<String, DpAggregates>

    assertThat(dpAggregates.data.toList()).containsExactlyElementsIn(expectedResult)
  }

  @Test
  fun aggregate_withPartitionSampler_appliesPartitionSampling() {
    val inputData =
      LocalCollection(
        sequenceOf(
          TestDataRow("Alice", "red", 10.0),
          TestDataRow("Alice", "green", 10.0),
          TestDataRow("Alice", "blue", 10.0),
        )
      )
    val publicPartitions = LocalCollection(sequenceOf("red", "green", "blue"))
    val computationalGraph =
      PublicPartitionsComputationalGraph(
        PartitionSampler(
          maxPartitionsContributed = 2,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        ),
        publicPartitions,
        PartitionsBalance.UNKNOWN,
        COUNT_SUM_AND_ID_COUNT_COMBINER,
        testDataExtractors,
        LOCAL_EF,
      )

    val dpAggregates = computationalGraph.aggregate(inputData) as LocalTable<String, DpAggregates>

    // The user contributed to 3 partitions but maxPartitionsContributed is set to 2. Hence,
    // contributions to 2 partitions should appear in the result.
    assertThat(dpAggregates.data.toMap().values.map { it.count }).containsExactly(1.0, 1.0, 0.0)
    assertThat(dpAggregates.data.toMap().values.map { it.sum }).containsExactly(10.0, 10.0, 0.0)
    assertThat(dpAggregates.data.toMap().values.map { it.privacyIdCount })
      .containsExactly(1.0, 1.0, 0.0)
  }
}
