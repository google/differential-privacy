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

/**
 * Metrics for a single movie in the Netflix dataset.
 *
 * <p>It is the result of the DP metrics query.
 */
public final class MovieMetrics {
  private final String movieId;

  private final long numberOfViewers;
  private final long numberOfViews;

  private final double averageOfRatings;

  MovieMetrics(String movieId, long numberOfViewers, long numberOfViews, double averageOfRatings) {
    this.movieId = movieId;
    this.numberOfViewers = numberOfViewers;
    this.numberOfViews = numberOfViews;
    this.averageOfRatings = averageOfRatings;
  }

  // 0-arg constructor is necessary for serialization to work.
  private MovieMetrics() {
    this("", 0, 0, 0.0);
  }

  @Override
  public String toString() {
    return String.format(
        "movieId=%s, numberOfViewers=%s, numberOfViews=%s, averageOfRatings=%s",
        movieId, numberOfViewers, numberOfViews, averageOfRatings);
  }
}
