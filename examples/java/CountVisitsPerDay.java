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

package com.google.privacy.differentialprivacy.example;

import com.google.privacy.differentialprivacy.Count;
import java.time.DayOfWeek;
import java.util.Arrays;
import java.util.EnumMap;

/**
 * Reads weekly visits from {@link InputFilePath#WEEK_STATISTICS}. Calculates non-private
 * and private counts of visits per week day and prints them to {@link #NON_PRIVATE_OUTPUT} and
 * {@link #PRIVATE_OUTPUT} correspondingly. Assumes that a visitor may enter the restaurant once per
 * day multiple times per week.
 */
public class CountVisitsPerDay {
  private static final String NON_PRIVATE_OUTPUT = "non_private_counts_per_day.csv";
  private static final String PRIVATE_OUTPUT = "private_counts_per_day.csv";

  private static final double LN_3 = Math.log(3);

  /**
   * Number of weekly visits for a visitor is limited to 3. All exceeding visits will be discarded.
   */
  private static final int MAX_VISITS_PER_WEEK = 3;

  private CountVisitsPerDay() {}

  /**
   * Reads statistics for a week, calculates raw and anonymized counts of visits per day,
   * and writes the results.
   * {@see the Javadoc of {@link CountVisitsPerDay} for more details}.
   */
  public static void run() {
    VisitsForWeek visitsForWeek = IOUtils.readWeeklyVisits(InputFilePath.WEEK_STATISTICS);

    EnumMap<DayOfWeek, Integer> nonPrivateCounts = getNonPrivateCounts(visitsForWeek);
    EnumMap<DayOfWeek, Integer> privateCounts = getPrivateCounts(visitsForWeek);

    IOUtils.writeCountsPerDayOfWeek(nonPrivateCounts, NON_PRIVATE_OUTPUT);
    IOUtils.writeCountsPerDayOfWeek(privateCounts, PRIVATE_OUTPUT);
  }

  /** Returns total raw count of visits for each day of the week. */
  private static EnumMap<DayOfWeek, Integer> getNonPrivateCounts(VisitsForWeek visits) {
    EnumMap<DayOfWeek, Integer> countsPerDay = new EnumMap<>(DayOfWeek.class);
    Arrays.stream(DayOfWeek.values()).forEach(d ->
        countsPerDay.put(d, visits.getVisitsForDay(d).size()));
    return countsPerDay;
  }

  /** Returns total anonymized count of visits for each day of the week. */
  private static EnumMap<DayOfWeek, Integer> getPrivateCounts(VisitsForWeek visits) {
    EnumMap<DayOfWeek, Integer> privateCountsPerDay = new EnumMap<>(DayOfWeek.class);

    // Pre-process the data set: limit the number of visits to MAX_VISITS_PER_WEEK
    // per visitorId.
    VisitsForWeek boundedVisits =
        ContributionBoundingUtils.boundVisits(visits, MAX_VISITS_PER_WEEK);

    Arrays.stream(DayOfWeek.values()).forEach(d -> {
      Count dpCount =
          Count.builder()
              // We use epsilon = log(3) in the example as suggested in
              // [Cynthia Dwork and Aaron Roth (2014),
              // "The Algorithmic Foundations of Differential Privacy"].
              .epsilon(LN_3)
              // The data was pre-processed so that
              // each user may visit the restaurant up to MAX_VISITS_PER_WEEK times per week.
              // Hence, each user may contribute to up to MAX_VISITS_PER_WEEK daily counts.
              // Note: while the library accepts this limit as a configurable parameter,
              // it doesn't pre-process the data to ensure this limit is respected.
              // It is responsibility of the caller to ensure the data passed to the library
              // is capped for getting the correct privacy guarantee.
              .maxPartitionsContributed(MAX_VISITS_PER_WEEK)
              .build();
      dpCount.incrementBy(boundedVisits.getVisitsForDay(d).size());
      privateCountsPerDay.put(d, (int) dpCount.computeResult());
    });

    return privateCountsPerDay;
  }
}
