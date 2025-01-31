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

/**
 * An end-to-end example how to compute DP metrics on a Netflix dataset using the library.
 *
 * <p>See README for details including how to run the example.
 */
package com.google.privacy.differentialprivacy.pipelinedp4j.examples;

import static java.lang.Math.round;
import static java.util.stream.Collectors.toCollection;

import com.google.privacy.differentialprivacy.pipelinedp4j.api.BeamQueryBuilder;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.Bounds;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.ContributionBoundingLevel;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.ContributionBounds;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.GroupsType;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.NoiseKind;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.QueryPerGroupResult;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.TotalBudget;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.ValueAggregationsBuilder;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.stream.IntStream;
import kotlin.jvm.functions.Function1;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.coders.StringUtf8Coder;
import org.apache.beam.sdk.extensions.avro.coders.AvroCoder;
import org.apache.beam.sdk.io.TextIO;
import org.apache.beam.sdk.options.Default;
import org.apache.beam.sdk.options.Description;
import org.apache.beam.sdk.options.PipelineOptions;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.options.Validation.Required;
import org.apache.beam.sdk.transforms.Create;
import org.apache.beam.sdk.transforms.MapElements;
import org.apache.beam.sdk.transforms.SerializableFunction;
import org.apache.beam.sdk.values.PCollection;

/**
 * An end-to-end example how to compute DP metrics on a Netflix dataset using the library on Beam.
 *
 * <p>See README for details including how to run the example.
 */
public final class BeamExample {
  /**
   * Options supported by {@link BeamExample}.
   *
   * <p>Inherits standard configuration options.
   */
  public interface BeamExampleOptions extends PipelineOptions {
    @Description(
        "If true we will assume in the example that movie ids are publicly known and are from "
            + "4500 to 4509"
            + ". Default is false, i.e. we will choose movie ids in a differentially"
            + " private way.")
    @Default.Boolean(false)
    boolean getUsePublicGroups();

    void setUsePublicGroups(boolean usePublicGroups);

    @Description(
        "Input file. For using as input file you can download data from"
            + " https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data. Use only part of"
            + " it to speed up the calculations.")
    @Required
    String getInputFilePath();

    void setInputFilePath(String value);

    /** Set this required option to specify where to write the output. */
    @Description("Output file.")
    @Required
    String getOutputFilePath();

    void setOutputFilePath(String value);
  }

  public static void main(String[] args) {
    BeamExampleOptions options =
        PipelineOptionsFactory.fromArgs(args).withValidation().as(BeamExampleOptions.class);

    runBeamExample(options);
  }

  static void runBeamExample(BeamExampleOptions options) {
    System.out.println("Starting calculations...");

    var pipeline = Pipeline.create(options);
    // Read the input data, these are movie views that contain movie id, user id and rating.
    PCollection<MovieView> data = readData(pipeline, options.getInputFilePath());

    // Define the query
    var groupsType =
        options.getUsePublicGroups()
            ? GroupsType.PublicGroups.create(publiclyKnownMovieIds(pipeline))
            : new GroupsType.PrivateGroups();
    var query =
        BeamQueryBuilder.from(
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
    PCollection<QueryPerGroupResult<String>> result = query.run();

    // Convert the result to better representation, i.e. to MovieMetrics.
    var movieMetricsCoder = AvroCoder.of(MovieMetrics.class);
    SerializableFunction<QueryPerGroupResult<String>, MovieMetrics> mapToMovieMetricsFn =
        perGroupResult -> {
          String movieId = perGroupResult.getGroupKey();
          long numberOfViewers =
              round(perGroupResult.getAggregationResults().get("numberOfViewers"));
          long numberOfViews = round(perGroupResult.getAggregationResults().get("numberOfViews"));
          double averageOfRatings = perGroupResult.getAggregationResults().get("averageOfRatings");
          return new MovieMetrics(movieId, numberOfViewers, numberOfViews, averageOfRatings);
        };
    // We now have our anonymized metrics of movie views.
    PCollection<MovieMetrics> anonymizedMovieMetrics =
        result
            .apply(
                "Map query result to MovieMetrics",
                MapElements.into(movieMetricsCoder.getEncodedTypeDescriptor())
                    .via(mapToMovieMetricsFn))
            .setCoder(movieMetricsCoder);

    // Save the result to a file.
    writeOutput(anonymizedMovieMetrics, options.getOutputFilePath());

    // Run the scheduled calculations in the pipeline.
    pipeline.run().waitUntilFinish();
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

  private static PCollection<MovieView> readData(Pipeline pipeline, String inputFilePath) {
    PCollection<String> inputPCollection =
        pipeline.apply("Read input", TextIO.read().from(inputFilePath));
    var coder = AvroCoder.of(MovieView.class);
    SerializableFunction<String, MovieView> parseFunction = MovieView::parseView;
    return inputPCollection
        .apply("Parse input", MapElements.into(coder.getEncodedTypeDescriptor()).via(parseFunction))
        .setCoder(coder);
  }

  /**
   * Movie ids (which are group keys for this dataset) are integers from 1 to ~17000. Set public
   * groups to a subset of them.
   */
  private static PCollection<String> publiclyKnownMovieIds(Pipeline pipeline) {
    var publicGroupsAsJavaList =
        IntStream.rangeClosed(
                4500, 4509
                )
            .mapToObj(Integer::toString)
            .collect(toCollection(ArrayList::new));
    return pipeline.apply("Create public groups", Create.of(publicGroupsAsJavaList));
  }

  private static void writeOutput(PCollection<MovieMetrics> result, String outputFilePath) {
    SerializableFunction<MovieMetrics, String> toStringFunction = MovieMetrics::toString;
    var lines =
        result.apply(
            "Map MovieMetrics to string",
            MapElements.into(StringUtf8Coder.of().getEncodedTypeDescriptor())
                .via(toStringFunction));
    lines.apply("Write output to file", TextIO.write().withoutSharding().to(outputFilePath));
  }

  private BeamExample() {}
}
