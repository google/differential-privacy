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

import java.time.DayOfWeek;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Static utils that bound contributions on the input data.
 */
public class ContributionBoundingUtils {

  private ContributionBoundingUtils() { }

  /**
   * @return {@link VisitsForWeek} containing the restaurant visits limited to
   * {@code maxVisitsPerWeek} per {@link Visit#visitorId}.
   */
  static VisitsForWeek boundVisits(VisitsForWeek visits, int maxVisitsPerWeek) {
    Map<String, Integer> visitorIdToVisitsCount = new HashMap<>();
    List<Visit> allVisits = new ArrayList<>();
    Map<Visit, DayOfWeek> visitToDay = new HashMap<>();
    VisitsForWeek boundedVisits = new VisitsForWeek();

    // Add all visits to a list in order to shuffle them.
    for (DayOfWeek d : DayOfWeek.values()) {
      Collection<Visit> visitsForDay = visits.getVisitsForDay(d);
      allVisits.addAll(visitsForDay);
      visitsForDay.forEach(v -> visitToDay.put(v, d));
    }
    Collections.shuffle(allVisits);

    // Go through the unordered collection of visits. Add up to MAX_VISITS_PER_WEEK per visitorId to
    // the final result.
    for (Visit v : allVisits) {
          String visitorId = v.visitorId();
          Integer visitsCount = visitorIdToVisitsCount.get(visitorId);
          if (visitsCount == null) {
            visitorIdToVisitsCount.put(visitorId, 1);
            boundedVisits.addVisit(visitToDay.get(v), v);
          } else if (visitsCount < maxVisitsPerWeek) {
            visitorIdToVisitsCount.put(visitorId, visitsCount + 1);
            boundedVisits.addVisit(visitToDay.get(v), v);
          } // Otherwise, ignore the visit.
    };

    return boundedVisits;
  }
}
