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

import com.google.common.collect.Range;

public class RestaurantSchedule {
  /** An hour when visitors start entering the restaurant. */
  static final int OPENING_HOUR = 9;
  /** An hour when visitors stop entering the restaurant. */
  static final int CLOSING_HOUR = 20;
  /**For how many hours visitors can enter the restaurant. */
  static final int NUM_OF_WORK_HOURS = CLOSING_HOUR - OPENING_HOUR + 1;
  /** Range of valid work hours when a visitor can enter the restaurant. */
  static final Range<Integer> VALID_HOURS = Range.closed(OPENING_HOUR, CLOSING_HOUR);

  private RestaurantSchedule() {}
}
