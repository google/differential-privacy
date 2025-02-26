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

package com.google.privacy.differentialprivacy.pipelinedp4j.examples;

import static java.lang.Math.round;
import static java.util.stream.Collectors.toCollection;

import com.google.privacy.differentialprivacy.pipelinedp4j.api.Bounds;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.ContributionBoundingLevel;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.ContributionBounds;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.GroupsType;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.NoiseKind;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.QueryPerGroupResult;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.SparkQueryBuilder;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.TotalBudget;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.ValueAggregationsBuilder;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.stream.IntStream;
import kotlin.jvm.functions.Function1;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/**
 * An end-to-end example how to compute DP metrics on a Netflix dataset using the library on Spark.
 *
 * <p>See README for details including how to run the example.
 */
@Command(
    name = "SparkExample",
    version = {"SparkExample 1.0"},
    mixinStandardHelpOptions = true)
public class SparkExample implements Runnable {
  @Option(
      names = "--usePublicGroups",
      description =
          "If true we will assume in the example that movie ids are publicly known and are from "
              + "4500 to 4509"
              + ". Default is false, i.e. we will choose movie ids in a differentially"
              + " private way.",
      defaultValue = "false")
  private boolean usePublicGroups = false;

  @Option(
      names = "--inputFilePath",
      description =
          "Input file. For using as input file you can download data from"
              + " https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data. Use only part of"
              + " it to speed up the calculations.",
      required = true)
  private String inputFilePath;

  @Option(
      names = "--outputFolder",
      description = "Folder where output files will be written.",
      required = true)
  private String outputFolder;

  public static void main(String[] args) {
    int exitCode = new CommandLine(new SparkExample()).execute(args);
    System.exit(exitCode);
  }

  @Override
  public void run() {
    System.out.println("Starting calculations...");
    SparkSession spark = initSpark();
    // Read the input data, these are movie views that contain movie id, user id and rating.
    Dataset<MovieView> data = readData(spark);

    // Define the query
    var groupsType =
        usePublicGroups
            ? GroupsType.PublicGroups.create(publiclyKnownMovieIds(spark))
            : new GroupsType.PrivateGroups();
    var query =
        SparkQueryBuilder.from(
                data,
                /* privacyUnitExtractor= */ MovieView::getUserId,
                new ContributionBoundingLevel.DATASET_LEVEL(
                    /* maxGroupsContributed= */ 3, /* maxContributionsPerGroup= */ 1))
            .groupBy(/* groupKeyExtractor= */ MovieView::getMovieId, groupsType)
            .countDistinctPrivacyUnits(/* outputColumnName= */ "numberOfViewers")
            .count(/* outputColumnName= */ "numberOfViews")
            .aggregateValue(
                /* valueExtractor= */ new RatingExtractor(),
                /* valueAggregations= */ new ValueAggregationsBuilder()
                    .mean(/* outputColumnName= */ "averageOfRatings"),
                /* contributionBounds= */ new ContributionBounds(
                    /* totalValueBounds= */ null,
                    /* valueBounds= */ new Bounds(/* minValue= */ 1.0, /* maxValue= */ 5.0)))
            .build(new TotalBudget(/* epsilon= */ 1.1, /* delta= */ 1e-10), NoiseKind.LAPLACE);
    // Run the query with DP parameters.
    Dataset<QueryPerGroupResult<String>> result = query.run();

    // Convert the result to better representation, i.e. to MovieMetrics.
    Encoder<MovieMetrics> movieMetricsEncoder = Encoders.kryo(MovieMetrics.class);
    MapFunction<QueryPerGroupResult<String>, MovieMetrics> mapToMovieMetricsFn =
        perGroupResult -> {
          String movieId = perGroupResult.getGroupKey();
          long numberOfViewers =
              round(perGroupResult.getAggregationResults().get("numberOfViewers"));
          long numberOfViews = round(perGroupResult.getAggregationResults().get("numberOfViews"));
          double averageOfRatings = perGroupResult.getAggregationResults().get("averageOfRatings");
          return new MovieMetrics(movieId, numberOfViewers, numberOfViews, averageOfRatings);
        };
    // We now have our anonymized metrics of movie views.
    Dataset<MovieMetrics> anonymizedMovieMetrics =
        result.map(mapToMovieMetricsFn, movieMetricsEncoder);

    // Save the result to a file.
    writeOutput(anonymizedMovieMetrics);

    // Stop spark session
    spark.stop();
    System.out.println("Finished calculations.");
  }

  /**
   * Static extractor for rating extraction.
   *
   * <p>Rating extractor must be serializable. In Java, we can't use lambdas, method references or
   * anonymous classes because they capture `this` and therefore are not serializable.
   */
  private static class RatingExtractor implements Function1<MovieView, Double>, Serializable {
    @Override
    public Double invoke(MovieView movieView) {
      return movieView.getRating();
    }
  }

  private static SparkSession initSpark() {
    String sparkMasterEnv = System.getenv("SPARK_MASTER");
    return SparkSession.builder()
        .appName("PipelineDP4j Spark Example")
        .master(sparkMasterEnv != null && !sparkMasterEnv.isEmpty() ? sparkMasterEnv : "local[*]")
        .getOrCreate();
  }

  private Dataset<MovieView> readData(SparkSession spark) {
    Dataset<Row> inputDataFrame = spark.read().option("header", "false").csv(inputFilePath);
    MapFunction<Row, MovieView> mapToMovieView =
        row ->
            new MovieView(row.getString(1), row.getString(0), Double.valueOf((String) row.get(2)));
    return inputDataFrame.map(mapToMovieView, Encoders.kryo(MovieView.class));
  }

  /**
   * Movie ids (which are group keys for this dataset) are integers from 1 to ~17000. Set public
   * groups 4500-4509.
   */
  private static Dataset<String> publiclyKnownMovieIds(SparkSession spark) {
    ArrayList<String> publicGroupsAsJavaList =
        IntStream.rangeClosed(4500, 4509)
            .mapToObj(Integer::toString)
            .collect(toCollection(ArrayList::new));
    return spark.createDataset(publicGroupsAsJavaList, Encoders.STRING());
  }

  private void writeOutput(Dataset<MovieMetrics> result) {
    Dataset<String> lines =
        result.map((MapFunction<MovieMetrics, String>) MovieMetrics::toString, Encoders.STRING());
    lines.write().mode(SaveMode.Overwrite).text(outputFolder);
  }
}
