package com.google.privacy.differentialprivacy.pipelinedp4j.api

import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.DpAggregates
import com.google.privacy.differentialprivacy.pipelinedp4j.spark.SparkTable
import org.apache.spark.api.java.function.MapFunction
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders
import scala.Tuple2

/**
 * A differentially-private query to run on Spark.
 *
 * @param T the type of the elements in the collection.
 */
class SparkQuery<T>
internal constructor(
    data: PipelineDpCollection<T>,
    privacyIdExtractor: (T) -> String,
    groupKeyExtractor: (T) -> String,
    maxGroupsContributed: Int,
    maxContributionsPerGroup: Int,
    publicKeys: PipelineDpCollection<String>?,
    aggregations: List<AggregationSpec<T>>,
) :
    Query<T, Dataset<Tuple2<String, Map<String, Double>>>>(
        data,
        privacyIdExtractor,
        groupKeyExtractor,
        maxGroupsContributed,
        maxContributionsPerGroup,
        publicKeys,
        aggregations,
    ) {
    /**
     * Runs the query with the given total budget and noise kind.
     *
     * @param budget the budget to use for the query.
     * @param noiseKind the noise kind to use for the query.
     * @return the result of the query.
     */
    override fun run(
        budget: TotalBudget,
        noiseKind: NoiseKind,
    ): Dataset<Tuple2<String, Map<String, Double>>> {
        val result = (runWithDpEngine(budget, noiseKind) as SparkTable<String, DpAggregates>).data
        val outputColumnNamesWithMetricTypes = aggregations.outputColumnNamesWithMetricTypes()
        val encoder = Encoders.tuple(Encoders.STRING(), Encoders.kryo(Map::class.java) as org.apache.spark.sql.Encoder<Map<String, Double>>)
        val mapToResultFn = { kv: Tuple2<String, DpAggregates> ->
            val key = kv._1
            val dpAggregates = kv._2

            val aggregationsMap =
                buildMap<String, Double> {
                    for ((outputColumnName, metricType) in outputColumnNamesWithMetricTypes) {
                        when (metricType) {
                            MetricType.PRIVACY_ID_COUNT -> put(outputColumnName, dpAggregates.privacyIdCount)
                            MetricType.COUNT -> put(outputColumnName, dpAggregates.count)
                            MetricType.SUM -> put(outputColumnName, dpAggregates.sum)
                            MetricType.MEAN -> put(outputColumnName, dpAggregates.mean)
                            MetricType.VARIANCE -> put(outputColumnName, dpAggregates.variance)
                            is MetricType.QUANTILES -> {
                                for ((rank, value) in metricType.ranks.zip(dpAggregates.quantilesList)) {
                                    put("${outputColumnName}_${rank}", value)
                                }
                            }
                        }
                    }
                }

            Tuple2(key, aggregationsMap)
        }
        return result.map(MapFunction { kv: Tuple2<String, DpAggregates> -> mapToResultFn(kv) }, encoder)
    }
}