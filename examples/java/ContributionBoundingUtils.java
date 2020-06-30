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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Static utils that bound contributions on the input data.
 */
public class ContributionBoundingUtils {

  private ContributionBoundingUtils() { }

  /**
   * @return {@link VisitsForWeek} containing the restaurant visits where the number of days
   * contributed by a single visitor is limited to {@code maxContributedDays}.
   *
   * TODO: Generalize the logic to be used for different partition keys.
   */
  static VisitsForWeek boundContributedDays(VisitsForWeek visits, int maxContributedDays) {
    Map<String, Set<DayOfWeek>> boundedVisitorDays = new HashMap<>();
    List<Visit> allVisits = new ArrayList<>();
    VisitsForWeek boundedVisits = new VisitsForWeek();

    // Add all visits to a list in order to shuffle them.
    for (DayOfWeek d : DayOfWeek.values()) {
      Collection<Visit> visitsForDay = visits.getVisitsForDay(d);
      allVisits.addAll(visitsForDay);
    }
    Collections.shuffle(allVisits);

    // For each visitorId, copy their visits for at most maxContributedDays days to the result.
    for (Visit visit : allVisits) {
      String visitorId = visit.visitorId();
      DayOfWeek visitDay = visit.day();
      if (boundedVisitorDays.containsKey(visitorId)) {
        Set<DayOfWeek> visitorDays = boundedVisitorDays.get(visitorId);
        if (visitorDays.contains(visitDay)) {
          boundedVisits.addVisit(visit);
        } else if (visitorDays.size() < maxContributedDays) {
          visitorDays.add(visitDay);
          boundedVisits.addVisit(visit);
        }
      } else {
        Set<DayOfWeek> visitorDays = new HashSet<>();
        boundedVisitorDays.put(visitorId, visitorDays);
        visitorDays.add(visitDay);
        boundedVisits.addVisit(visit);
      }
    }

    return boundedVisits;
  }
}
