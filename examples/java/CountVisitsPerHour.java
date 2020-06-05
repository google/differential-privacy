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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.privacy.differentialprivacy.example.RestaurantSchedule.CLOSING_HOUR;
import static com.google.privacy.differentialprivacy.example.RestaurantSchedule.OPENING_HOUR;
import static com.google.privacy.differentialprivacy.example.RestaurantSchedule.VALID_HOURS;
import static java.util.stream.Collectors.toMap;

import com.google.common.collect.ImmutableSortedMap;
import com.google.privacy.differentialprivacy.Count;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * Loads {@link InputFilePath#DAY_STATISTICS} file, calculates non-anonymized and
 * anonymized counts of visitors entering a restaurant every hour and prints result
 * to {@link #NON_PRIVATE_OUTPUT} and {@link #PRIVATE_OUTPUT}.
 * Uses {@link Count}.
 */
public class CountVisitsPerHour {
  private static final String NON_PRIVATE_OUTPUT = "non_private_counts_per_hour.csv";
  private static final String PRIVATE_OUTPUT = "private_counts_per_hour.csv";

  private static final double LN_3 = Math.log(3);

  private CountVisitsPerHour() {}

  public static void run() {
    Collection<Visit> dailyVisits = IOUtils.readDailyVisits(InputFilePath.DAY_STATISTICS);

    Map<Integer, Integer> nonPrivateCounts = getNonPrivateCounts(dailyVisits);
    Map<Integer, Integer> privateCounts = getPrivateCounts(dailyVisits);

    IOUtils.writeCountsPerHourOfDay(nonPrivateCounts, NON_PRIVATE_OUTPUT);
    IOUtils.writeCountsPerHourOfDay(privateCounts, PRIVATE_OUTPUT);
  }

  /**
   * Calculates raw count of {@link Visit}s per hour of day.
   *
   * <p>{@return} a {@link ImmutableSortedMap} that maps an hour to a raw count of visits for the
   * hours between {@link RestaurantSchedule#OPENING_HOUR} and {@link
   * RestaurantSchedule#CLOSING_HOUR}.
   */
  private static ImmutableSortedMap<Integer, Integer> getNonPrivateCounts(
      Collection<Visit> visits) {
    Map<Integer, Integer> counts = new TreeMap<>();

    for (int h = OPENING_HOUR; h <= CLOSING_HOUR; h++) {
      counts.put(h, 0);
    }

    visits.forEach(v -> {
      int hour = v.entryTime().getHour();
      // Validate that the visit happened during one of valid hours.
      checkArgument(VALID_HOURS.contains(hour));
      int newCount = counts.get(hour) + 1;
      counts.put(hour, newCount);
    });

    return ImmutableSortedMap.copyOf(counts);
  }

  /**
   * Calculates anonymized ("private") count of {@link Visit}s per hour of day.
   *
   * {@return} a {@link ImmutableSortedMap} that maps an hour to anonymized count of visits.
   */
  private static ImmutableSortedMap<Integer, Integer> getPrivateCounts(Collection<Visit> visits) {
    // Construct DP Count objects which will be used to calculate DP counts
    // one Count is created for every work hour.
    Map<Integer, Count> dpCounts = new HashMap<>();
    for (int i = OPENING_HOUR; i <= CLOSING_HOUR; i++) {
      Count dpCount = Count.builder()
          .epsilon(LN_3)
          .maxPartitionsContributed(1)
          .build();
      dpCounts.put(i, dpCount);
    }

    // Go through all visits and update Counts at the corresponding hours.
    visits.forEach(v -> dpCounts.get(v.entryTime().getHour()).increment());

    // Trigger DP logic to produce anonymized counts of visits.
    return ImmutableSortedMap.copyOf(
        dpCounts.entrySet().stream()
        .collect(
            toMap(Map.Entry::getKey, e -> (int) e.getValue().computeResult())));
  }
}
