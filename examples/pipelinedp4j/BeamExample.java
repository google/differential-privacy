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

import com.google.privacy.differentialprivacy.pipelinedp4j.api.NoiseKind;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.QueryBuilder;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.QueryPerGroupResult;
import com.google.privacy.differentialprivacy.pipelinedp4j.api.TotalBudget;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.stream.IntStream;
import kotlin.jvm.functions.Function1;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.coders.StringUtf8Coder;
import org.apache.beam.sdk.extensions.avro.coders.AvroCoder;
import org.apache.beam.sdk.io.TextIO;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.transforms.Create;
import org.apache.beam.sdk.transforms.MapElements;
import org.apache.beam.sdk.transforms.SerializableFunction;
import org.apache.beam.sdk.values.PCollection;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/**
 * An end-to-end example how to compute DP metrics on a Netflix dataset using the library on Beam.
 *
 * <p>See README for details including how to run the example.
 */
@Command(
    name = "BeamExample",
    version = {"BeamExample 1.0"},
    mixinStandardHelpOptions = true)
public class BeamExample implements Runnable {
  @Option(
      names = "--use-public-groups",
      description =
          "If true we will assume in the example that movie ids are publicly known and are from "
              + "4500 to 4509"
              + ". Default is false, i.e. we will choose movie ids in a differentially"
              + " private way.",
      defaultValue = "false")
  private boolean usePublicGroups = false;

  @Option(
      names = "--local-input-file-path",
      description =
          "Input file. For using as input file you can download data from"
              + " https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data. Use only part of"
              + " it to speed up the calculations.",
      required = true)
  private String localInputFilePath;

  @Option(
      names = "--local-output-file-path",
      description = "Output file.",
      defaultValue = "/tmp/anonymized_output.txt")
  private String localOutputFilePath;

  public static void main(String[] args) {
    int exitCode = new CommandLine(new BeamExample()).execute(args);
    System.exit(exitCode);
  }

  @Override
  public void run() {
    System.out.println("Starting calculations...");

    var pipeline = initBeam();
    // Read the input data, these are movie views that contain movie id, user id and rating.
    PCollection<MovieView> data = readData(pipeline);

    // Define the query
    var query =
        QueryBuilder.from(data, /* privacyIdExtractor= */ new UserIdExtractor())
            .groupBy(
                /* groupKeyExtractor= */ new MovieIdExtractor(),
                /* maxGroupsContributed= */ 3,
                /* maxContributionsPerGroup= */ 1,
                usePublicGroups ? publiclyKnownMovieIds(pipeline) : null)
            .countDistinctPrivacyUnits("numberOfViewers")
            .count(/* outputColumnName= */ "numberOfViews")
            .mean(
                new RatingExtractor(),
                /* minValue= */ 1.0,
                /* maxValue= */ 5.0,
                /* outputColumnName= */ "averageOfRatings",
                /* budget= */ null)
            .build();
    // Run the query with DP parameters.
    PCollection<QueryPerGroupResult> result =
        query.run(new TotalBudget(/* epsilon= */ 1.1, /* delta= */ 1e-10), NoiseKind.LAPLACE);

    // Convert the result to better representation, i.e. to MovieMetrics.
    var movieMetricsCoder = AvroCoder.of(MovieMetrics.class);
    SerializableFunction<QueryPerGroupResult, MovieMetrics> mapToMovieMetricsFn =
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
    writeOutput(anonymizedMovieMetrics);

    // Run the scheduled calculations in the pipeline.
    pipeline.run().waitUntilFinish();
    System.out.println("Finished calculations.");
  }

  // Data extractors. They always have to implement Function1 and Serializable interfaces. If it
  // doesn't implement Serializable interface, it will fail on Beam. If it doesn't implement
  // Function1, it will at compile time due to types mismatch. Do not use lambdas for data
  // extractors as they won't be serializable.
  static class UserIdExtractor implements Function1<MovieView, String>, Serializable {
    @Override
    public String invoke(MovieView movieView) {
      return movieView.getUserId();
    }
  }

  static class MovieIdExtractor implements Function1<MovieView, String>, Serializable {
    @Override
    public String invoke(MovieView movieView) {
      return movieView.getMovieId();
    }
  }

  static class RatingExtractor implements Function1<MovieView, Double>, Serializable {
    @Override
    public Double invoke(MovieView movieView) {
      return movieView.getRating();
    }
  }

  private static Pipeline initBeam() {
    var options = PipelineOptionsFactory.create();
    return Pipeline.create(options);
  }

  private PCollection<MovieView> readData(Pipeline pipeline) {
    PCollection<String> inputPCollection =
        pipeline.apply("Read input", TextIO.read().from(localInputFilePath));
    var coder = AvroCoder.of(MovieView.class);
    SerializableFunction<String, MovieView> parseFunction = MovieView::parseView;
    return inputPCollection
        .apply("Parse input", MapElements.into(coder.getEncodedTypeDescriptor()).via(parseFunction))
        .setCoder(coder);
  }

  /**
   * Movie ids (which are group keys for this dataset) are integers from 1 to ~17000. Set public
   * groups 1-10.
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

  private void writeOutput(PCollection<MovieMetrics> result) {
    SerializableFunction<MovieMetrics, String> toStringFunction = MovieMetrics::toString;
    var lines =
        result.apply(
            "Map MovieMetrics to string",
            MapElements.into(StringUtf8Coder.of().getEncodedTypeDescriptor())
                .via(toStringFunction));
    lines.apply("Write output to file", TextIO.write().withoutSharding().to(localOutputFilePath));
  }
}
