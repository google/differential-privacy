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

import static java.util.stream.Collectors.toCollection;
import static org.apache.spark.sql.functions.col;

import com.google.privacy.differentialprivacy.pipelinedp4j.api.Bounds;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.ColumnNames;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.ContributionBoundingLevel;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.ContributionBounds;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.GroupsType;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.NoiseKind;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.SparkDataFrameQueryBuilder;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.TotalBudget;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.ValueAggregationsBuilder;
import java.util.ArrayList;
import java.util.stream.IntStream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/**
 * An end-to-end example how to compute DP metrics on a Netflix dataset using the library on Spark
 * with DataFrame API.
 *
 * <p>See README for details including how to run the example.
 */
@Command(
    name = "SparkDataFrameExample",
    version = {"SparkDataFrameExample 1.0"},
    mixinStandardHelpOptions = true)
public class SparkDataFrameExample implements Runnable {
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
    int exitCode = new CommandLine(new SparkDataFrameExample()).execute(args);
    System.exit(exitCode);
  }

  @Override
  public void run() {
    System.out.println("Starting calculations...");
    SparkSession spark = initSpark();
    // Read the input data, these are movie views that contain movie id, user id and rating.
    Dataset<Row> data = readData(spark);

    // Define the query
    var groupsType =
        usePublicGroups
            ? GroupsType.PublicGroups.createForDataFrame(publiclyKnownMovieIds(spark))
            : new GroupsType.PrivateGroups();
    var query =
        SparkDataFrameQueryBuilder.from(
                data,
                /* privacyUnitColumnNames= */ new ColumnNames("userId"),
                new ContributionBoundingLevel.DATASET_LEVEL(
                    /* maxGroupsContributed= */ 3, /* maxContributionsPerGroup= */ 1))
            .groupBy(/* groupKeyColumnNames= */ new ColumnNames("movieId"), groupsType)
            .countDistinctPrivacyUnits(/* outputColumnName= */ "numberOfViewers")
            .count(/* outputColumnName= */ "numberOfViews")
            .aggregateValue(
                /* valueColumnName= */ "rating",
                /* valueAggregations= */ new ValueAggregationsBuilder()
                    .mean(/* outputColumnName= */ "averageOfRatings"),
                /* contributionBounds= */ new ContributionBounds(
                    /* totalValueBounds= */ null,
                    /* valueBounds= */ new Bounds(/* minValue= */ 1.0, /* maxValue= */ 5.0)))
            .build(new TotalBudget(/* epsilon= */ 1.1, /* delta= */ 1e-10), NoiseKind.LAPLACE);
    // Run the query with DP parameters.
    Dataset<Row> anonymizedMovieMetrics = query.run();

    // Save the result to a file.
    writeOutput(anonymizedMovieMetrics);

    // Stop spark session
    spark.stop();
    System.out.println("Finished calculations.");
  }

  private static SparkSession initSpark() {
    String sparkMasterEnv = System.getenv("SPARK_MASTER");
    return SparkSession.builder()
        .appName("PipelineDP4j Spark DataFrame Example")
        .master(sparkMasterEnv != null && !sparkMasterEnv.isEmpty() ? sparkMasterEnv : "local[*]")
        .getOrCreate();
  }

  private Dataset<Row> readData(SparkSession spark) {
    return spark
        .read()
        .option("header", "false")
        .csv(inputFilePath)
        .selectExpr("_c0 as movieId", "_c1 as userId", "_c2 as rating")
        .withColumn("rating", col("rating").cast("double"));
  }

  /**
   * Movie ids (which are group keys for this dataset) are integers from 1 to ~17000. Set public
   * groups 4500-4509.
   */
  private static Dataset<Row> publiclyKnownMovieIds(SparkSession spark) {
    ArrayList<String> publicGroupsAsJavaList =
        IntStream.rangeClosed(4500, 4509)
            .mapToObj(Integer::toString)
            .collect(toCollection(ArrayList::new));
    return spark.createDataset(publicGroupsAsJavaList, Encoders.STRING()).toDF("movieId");
  }

  private void writeOutput(Dataset<Row> result) {
    result
        .write()
        .format("csv")
        .option("header", "true")
        .option("delimiter", ",")
        .mode(SaveMode.Overwrite)
        .save(outputFolder);
  }
}
