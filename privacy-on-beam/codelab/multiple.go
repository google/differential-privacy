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
	log "github.com/golang/glog"
	"github.com/google/differential-privacy/privacy-on-beam/v3/pbeam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
)

// ComputeCountMeanSum computes the three aggregations (count, mean and sum) we
// compute separately in the other files in a differentially private way.
// This pipeline uses a single PrivacySpec for all the aggregations, meaning
// that they share the same privacy budget.
func ComputeCountMeanSum(s beam.Scope, col beam.PCollection) (visitsPerHour, meanTimeSpent, revenues beam.PCollection) {
	s = s.Scope("ComputeCountMeanSum")
	// Create a Privacy Spec and convert col into a PrivatePCollection.
	spec, err := pbeam.NewPrivacySpec(pbeam.PrivacySpecParams{AggregationEpsilon: epsilon}) // Shared by count, mean and sum.
	if err != nil {
		log.Fatalf("Couldn't create a PrivacySpec: %v", err)
	}
	pCol := pbeam.MakePrivateFromStruct(s, col, spec, "VisitorID")

	// Create a PCollection of output partitions, i.e. restaurant's work hours (from 9 am till 9pm (exclusive)).
	hours := beam.CreateList(s, [12]int{9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20})

	visitHours := pbeam.ParDo(s, extractVisitHourFn, pCol)
	visitsPerHour = pbeam.Count(s, visitHours, pbeam.CountParams{
		AggregationEpsilon:       epsilon / 3,
		MaxPartitionsContributed: 1,     // Visitors can visit the restaurant once (one hour) a day
		MaxValue:                 1,     // Visitors can visit the restaurant once within an hour
		PublicPartitions:         hours, // Visitors only visit during work hours
	})

	hourToTimeSpent := pbeam.ParDo(s, extractVisitHourAndTimeSpentFn, pCol)
	meanTimeSpent = pbeam.MeanPerKey(s, hourToTimeSpent, pbeam.MeanParams{
		AggregationEpsilon:           epsilon / 3,
		MaxPartitionsContributed:     1,     // Visitors can visit the restaurant once (one hour) a day
		MaxContributionsPerPartition: 1,     // Visitors can visit the restaurant once within an hour
		MinValue:                     0,     // Minimum time spent per user (in mins)
		MaxValue:                     60,    // Maximum time spent per user (in mins)
		PublicPartitions:             hours, // Visitors only visit during work hours
	})

	hourToMoneySpent := pbeam.ParDo(s, extractVisitHourAndMoneySpentFn, pCol)
	revenues = pbeam.SumPerKey(s, hourToMoneySpent, pbeam.SumParams{
		AggregationEpsilon:       epsilon / 3,
		MaxPartitionsContributed: 1,     // Visitors can visit the restaurant once (one hour) a day
		MinValue:                 0,     // Minimum money spent per user (in euros)
		MaxValue:                 40,    // Maximum money spent per user (in euros)
		PublicPartitions:         hours, // Visitors only visit during work hours
	})

	return visitsPerHour, meanTimeSpent, revenues
}
