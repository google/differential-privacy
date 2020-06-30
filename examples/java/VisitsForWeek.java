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
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumMap;
import java.util.HashSet;

/**
 * Stores {@link Visit}s for each {@link DayOfWeek}.
 */
class VisitsForWeek {
  private final EnumMap<DayOfWeek, Collection<Visit>> visits;

  VisitsForWeek() {
    visits = new EnumMap<>(DayOfWeek.class);
    Arrays.stream(DayOfWeek.values()).forEach(d -> visits.put(d, new HashSet<>()));
  }

  /**
   * Adds the given {@link Visit}.
   */
  void addVisit(Visit visit) {
    visits.get(visit.day()).add(visit);
  }

  Collection<Visit> getVisitsForDay(DayOfWeek day) {
    return visits.get(day);
  }
}
