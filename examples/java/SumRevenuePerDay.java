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

import com.google.privacy.differentialprivacy.BoundedSum;
import java.time.DayOfWeek;
import java.util.EnumMap;

/**
 * Reads weekly visits from {@link InputFilePath#WEEK_STATISTICS}, calculates the
 * raw and anonymized amount of money spent by the visitors per week day
 * and prints them to {@link #NON_PRIVATE_OUTPUT} and {@link #PRIVATE_OUTPUT} correspondingly.
 * Assumes that a visitor may enter the restaurant once per day
 * multiple times per week.
 */
public class SumRevenuePerDay {
  private static final String NON_PRIVATE_OUTPUT = "non_private_sums_per_day.csv";
  private static final String PRIVATE_OUTPUT = "private_sums_per_day.csv";

  private static final double LN_3 = Math.log(3);

  /**
   * Number of visit days contributed by a single visitor will be limited to 4. All exceeding
   * visits will be discarded.
   */
  private static final int MAX_CONTRIBUTED_DAYS = 4;
  /** Minimum amount of money we expect a visitor to spend on a single visit. */
  private static final int MIN_EUROS_SPENT = 0;
  /** Maximum amount of money we expect a visitor to spend on a single visit. */
  private static final int MAX_EUROS_SPENT = 50;

  private SumRevenuePerDay() { }

  /**
   * Reads statistics for a week, calculates raw and anonymized sums of money spent by visits
   * per day, and writes the results.
   * {@see the Javadoc of {@link SumRevenuePerDay} for more details}.
   */
  public static void run() {
    VisitsForWeek visitsForWeek = IOUtils.readWeeklyVisits(InputFilePath.WEEK_STATISTICS);

    EnumMap<DayOfWeek, Integer> nonPrivateSums = getNonPrivateSums(visitsForWeek);
    EnumMap<DayOfWeek, Integer> privateSums = getPrivateSums(visitsForWeek);

    IOUtils.writeCountsPerDayOfWeek(nonPrivateSums, NON_PRIVATE_OUTPUT);
    IOUtils.writeCountsPerDayOfWeek(privateSums, PRIVATE_OUTPUT);
  }

  /** Returns total raw revenue for each day of the week. */
  static EnumMap<DayOfWeek, Integer> getNonPrivateSums(VisitsForWeek visits) {
    EnumMap<DayOfWeek, Integer> sumsPerDay = new EnumMap<>(DayOfWeek.class);
    for (DayOfWeek d : DayOfWeek.values()) {
      int sum = 0;
      for (Visit v : visits.getVisitsForDay(d)) {
        sum += v.eurosSpent();
      }
      sumsPerDay.put(d, sum);
    }
    return sumsPerDay;
  }

  /** Returns total anonymized revenue for each day of the week. */
  private static EnumMap<DayOfWeek, Integer> getPrivateSums(VisitsForWeek visits) {
    EnumMap<DayOfWeek, Integer> privateSumsPerDay = new EnumMap<>(DayOfWeek.class);

    // Pre-process the data set: limit the number of visits to MAX_VISITS_PER_WEEK
    // per visitorId.
    VisitsForWeek boundedVisits =
        ContributionBoundingUtils.boundContributedDays(visits, MAX_CONTRIBUTED_DAYS);

    for (DayOfWeek d : DayOfWeek.values()) {
      BoundedSum dpSum =
          BoundedSum.builder()
              .epsilon(LN_3)
              // The data was pre-processed so that each visitor may visit the restaurant up to
              // MAX_VISIT_DAYS days per week.
              // Hence, each user may contribute to up to MAX_VISIT_DAYS daily counts.
              // Note: while the library accepts this limit as a configurable parameter,
              // it doesn't pre-process the data to ensure this limit is respected.
              // It is responsibility of the caller to ensure the data passed to the library
              // is capped for getting the correct privacy guarantee.
              .maxPartitionsContributed(MAX_CONTRIBUTED_DAYS)
              // No need to pre-process the data: BoundedSum will clamp the input values.
              .lower(MIN_EUROS_SPENT)
              .upper(MAX_EUROS_SPENT)
              .build();

      for (Visit v : boundedVisits.getVisitsForDay(d)) {
        dpSum.addEntry(v.eurosSpent());
      }

      privateSumsPerDay.put(d, (int) dpSum.computeResult());
    }

    return privateSumsPerDay;
  }
}
