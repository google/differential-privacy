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

package codelab

import (
	"github.com/google/differential-privacy/privacy-on-beam/pbeam"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/transforms/stats"
)

func init() {
	beam.RegisterFunction(extractVisitHourAndTimeSpentFn)
}

// MeanTimeSpent calculates and returns the average time spent by visitors
// who entered the restaurant for each hour. This produces a non-anonymized,
// non-private count. Use PrivateMeanTimeSpent for computing this in an
// anonymized way.
func MeanTimeSpent(s beam.Scope, col beam.PCollection) beam.PCollection {
	s = s.Scope("MeanTimeSpent")
	hourToTimeSpent := beam.ParDo(s, extractVisitHourAndTimeSpentFn, col)
	meanTimeSpent := stats.MeanPerKey(s, hourToTimeSpent)
	return meanTimeSpent
}

func extractVisitHourAndTimeSpentFn(v Visit) (int, int) {
	return v.TimeEntered.Hour(), v.MinutesSpent
}

// PrivateMeanTimeSpent calculates and returns the average time spent by visitors
// who entered the restaurant for each hour in a differentially private way.
func PrivateMeanTimeSpent(s beam.Scope, col beam.PCollection) beam.PCollection {
	s = s.Scope("PrivateMeanTimeSpent")
	// Create a Privacy Spec and convert col into a PrivatePCollection
	spec := pbeam.NewPrivacySpec(epsilon, delta)
	pCol := pbeam.MakePrivateFromStruct(s, col, spec, "VisitorID")

	hourToTimeSpent := pbeam.ParDo(s, extractVisitHourAndTimeSpentFn, pCol)
	meanTimeSpent := pbeam.MeanPerKey(s, hourToTimeSpent, pbeam.MeanParams{
		MaxPartitionsContributed:     1,  // Visitors can visit the restaurant once (one hour) a day
		MaxContributionsPerPartition: 1,  // Visitors can visit the restaurant once within an hour
		MinValue:                     0,  // Minimum time spent per user (in mins)
		MaxValue:                     60, // Maximum time spent per user (in mins)
	})
	return meanTimeSpent
}
