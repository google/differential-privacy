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

import com.google.auto.value.AutoValue;
import java.time.DayOfWeek;
import java.time.LocalTime;

/** Stores data about single visit of a user to the restaurant. */
@AutoValue
abstract class Visit {

  static Visit create(
      String visitorId, LocalTime entryTime, int minutesSpent, int eurosSpent, DayOfWeek day) {
    return new AutoValue_Visit(visitorId, entryTime, minutesSpent, eurosSpent, day);
  }

  abstract String visitorId();

  abstract LocalTime entryTime();

  abstract int minutesSpent();

  abstract int eurosSpent();

  abstract DayOfWeek day();
}
