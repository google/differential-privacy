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

import static java.nio.charset.StandardCharsets.UTF_8;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.Resources;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.DayOfWeek;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.util.EnumMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Reads visitors' data and prints statistics. */
class IOUtils {

  private static final String CSV_ITEM_SEPARATOR = ",";
  private static final DateTimeFormatter TIME_FORMATTER =
      new DateTimeFormatterBuilder()
          // case insensitive
          .parseCaseInsensitive()
          // pattern
          .appendPattern("h:mm:ss a")
          // set Locale that uses "AM" and "PM"
          .toFormatter(Locale.ENGLISH);
  private static final String CSV_HOUR_COUNT_WRITE_TEMPLATE = "%d,%d\n";
  private static final String CSV_DAY_COUNT_WRITE_TEMPLATE = "%s,%d\n";

  private IOUtils() {}

  /**
   * Reads daily visitors' data.
   * {@see #convertCsvLineWithoutDayToList} for details on the format.
   */
  static ImmutableSet<Visit> readDailyVisits(String file) {
    try {
      List<String> visitsAsText =
          Resources.readLines(Resources.getResource(file), UTF_8);

      return visitsAsText.stream()
          .skip(1)
          .map(IOUtils::convertLineToVisit)
          .collect(toImmutableSet());
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Converts a line of format "visitorId,entryTime, minutesSpent, moneySpent, day" to
   * {@link Visit}.
   */
  private static Visit convertLineToVisit(String visitAsText) {
    Iterator<String> splitVisit = Splitter.on(CSV_ITEM_SEPARATOR).split(visitAsText).iterator();
    // element 0
    String visitorId = splitVisit.next();
    // element 1
    LocalTime timeEntered = LocalTime.parse(splitVisit.next(), TIME_FORMATTER);
    // element 2
    int timeSpent = Integer.parseInt(splitVisit.next());
    // element 3
    int moneySpent = Integer.parseInt(splitVisit.next());
    // element 4
    DayOfWeek day = DayOfWeek.of(Integer.parseInt(splitVisit.next()));

    return Visit.create(visitorId, timeEntered, timeSpent, moneySpent, day);
  }

  /**
   * Reads daily visitors' data. Assumes that the input file is a .csv file of format "visitorId,
   * entryTime, minutesSpent, moneySpent, day".
   */
  static VisitsForWeek readWeeklyVisits(String file) {
    VisitsForWeek result = new VisitsForWeek();

    try {
      List<String> visitsAsText =
          Resources.readLines(Resources.getResource(file), UTF_8);
      visitsAsText.stream()
          .skip(1)
          .forEach(v -> {
            Visit visit = convertLineToVisit(v);
            result.addVisit(visit);
          });

    } catch (IOException e) {
      throw new IllegalStateException(e);
    }

    return result;
  }

  static void writeCountsPerHourOfDay(Map<Integer, Integer> counts, String file) {
    try (PrintWriter pw = new PrintWriter(new File(file), UTF_8.name())) {
      counts.forEach((
          hour, count) -> pw.write(String.format(CSV_HOUR_COUNT_WRITE_TEMPLATE, hour, count)));
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  static void writeCountsPerDayOfWeek(EnumMap<DayOfWeek, Integer> counts, String file) {
    try (PrintWriter pw = new PrintWriter(new File(file), UTF_8.name())) {
      counts.forEach(
          (day, count) -> pw.write(String.format(CSV_DAY_COUNT_WRITE_TEMPLATE, day.name(), count)));
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }
}
