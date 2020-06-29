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
)

// ComputeCountMeanSum computes the three aggregations (count, mean and sum) we
// compute separately in the other files in a differentially private way.
// This pipeline uses a single PrivacySpec for all the aggregations, meaning
// that they share the same privacy budget.
func ComputeCountMeanSum(s beam.Scope, col beam.PCollection) (visitsPerHour, meanTimeSpent, revenues beam.PCollection) {
	s = s.Scope("ComputeCountMeanSum")
	// Create a Privacy Spec and convert col into a PrivatePCollection
	spec := pbeam.NewPrivacySpec(epsilon, delta) // Shared by count, mean and sum.
	pCol := pbeam.MakePrivateFromStruct(s, col, spec, "VisitorID")

	visitHours := pbeam.ParDo(s, extractVisitHour, pCol)
	visitsPerHour = pbeam.Count(s, visitHours, pbeam.CountParams{
		Epsilon:                  epsilon / 3,
		Delta:                    delta / 3,
		MaxPartitionsContributed: 1, // Visitors can visit the restaurant once (one hour) a day
		MaxValue:                 1, // Visitors can visit the restaurant once within an hour
	})

	hourToTimeSpent := pbeam.ParDo(s, extractVisitHourAndTimeSpentFn, pCol)
	meanTimeSpent = pbeam.MeanPerKey(s, hourToTimeSpent, pbeam.MeanParams{
		Epsilon:                      epsilon / 3,
		Delta:                        delta / 3,
		MaxPartitionsContributed:     1,  // Visitors can visit the restaurant once (one hour) a day
		MaxContributionsPerPartition: 1,  // Visitors can visit the restaurant once within an hour
		MinValue:                     0,  // Minimum time spent per user (in mins)
		MaxValue:                     60, // Maximum time spent per user (in mins)
	})

	hourToMoneySpent := pbeam.ParDo(s, extractVisitHourAndTimeSpentFn, pCol)
	revenues = pbeam.SumPerKey(s, hourToMoneySpent, pbeam.SumParams{
		Epsilon:                  epsilon / 3,
		Delta:                    delta / 3,
		MaxPartitionsContributed: 1,  // Visitors can visit the restaurant once (one hour) a day
		MinValue:                 0,  // Minimum money spent per user (in dollars)
		MaxValue:                 40, // Maximum money spent per user (in dollars)
	})

	return visitsPerHour, meanTimeSpent, revenues
}
