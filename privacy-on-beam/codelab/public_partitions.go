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

// PrivateCountVisitsPerHourWithPublicPartitions counts and returns the number of visits to a
// restaurant for each hour in a differentially private way. It uses the "public partitions"
// feature to disable partition selection/thresholding.
func PrivateCountVisitsPerHourWithPublicPartitions(s beam.Scope, col beam.PCollection) beam.PCollection {
	s = s.Scope("PrivateCountVisitsPerHourWithPublicPartitions")
	// Create a Privacy Spec and convert col into a PrivatePCollection.
	spec, err := pbeam.NewPrivacySpec(pbeam.PrivacySpecParams{AggregationEpsilon: epsilon})
	if err != nil {
		log.Fatalf("Couldn't create a PrivacySpec: %v", err)
	}
	pCol := pbeam.MakePrivateFromStruct(s, col, spec, "VisitorID")

	// Create a PCollection of output partitions, i.e. restaurant's work hours (from 9 am till 9pm (exclusive)).
	hours := beam.CreateList(s, [12]int{9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20})

	visitHours := pbeam.ParDo(s, extractVisitHourFn, pCol)
	visitsPerHour := pbeam.Count(s, visitHours, pbeam.CountParams{
		MaxPartitionsContributed: 1,     // Visitors can visit the restaurant once (one hour) a day
		MaxValue:                 1,     // Visitors can visit the restaurant once within an hour
		PublicPartitions:         hours, // Visitors only visit during work hours
	})
	return visitsPerHour
}
