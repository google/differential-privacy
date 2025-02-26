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

import com.google.common.base.Splitter;
import java.util.List;

/** Represents a single movie view from the Netflix dataset. */
public final class MovieView {
  private final String userId;
  private final String movieId;
  private final Double rating;

  MovieView(String userId, String movieId, Double rating) {
    this.userId = userId;
    this.movieId = movieId;
    this.rating = rating;
  }

  // 0-arg constructor is necessary for serialization to work.
  private MovieView() {
    this("", "", 0.0);
  }

  String getUserId() {
    return userId;
  }

  String getMovieId() {
    return movieId;
  }

  Double getRating() {
    return rating;
  }

  static MovieView parseView(String s) {
    List<String> spl = Splitter.on(',').splitToList(s);
    return new MovieView(spl.get(1), spl.get(0), Double.parseDouble(spl.get(2)));
  }
}
