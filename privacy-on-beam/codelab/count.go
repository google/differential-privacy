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

// Package codelab contains example pipelines for computing various aggregations using Privacy on Beam.
package codelab

import (
	"math"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/privacy-on-beam/v3/pbeam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/register"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/stats"
)

// Constants used throughtout the codelab
var epsilon = math.Log(3)

const delta = 1e-5

func init() {
	register.Function1x1[Visit, int](extractVisitHourFn)
}

// CountVisitsPerHour counts and returns the number of visits to a restaurant for each hour.
// This produces a non-anonymized, non-private count. Use PrivateCountVisitsPerHour
// for computing this in an anonymized way.
func CountVisitsPerHour(s beam.Scope, col beam.PCollection) beam.PCollection {
	s = s.Scope("CountVisitsPerHour")
	visitHours := beam.ParDo(s, extractVisitHourFn, col)
	visitsPerHour := stats.Count(s, visitHours)
	return visitsPerHour
}

func extractVisitHourFn(v Visit) int {
	return v.TimeEntered.Hour()
}

// PrivateCountVisitsPerHour counts and returns the number of visits to a restaurant for each hour
// in a differentially private way.
func PrivateCountVisitsPerHour(s beam.Scope, col beam.PCollection) beam.PCollection {
	s = s.Scope("PrivateCountVisitsPerHour")
	// Create a Privacy Spec and convert col into a PrivatePCollection
	spec, err := pbeam.NewPrivacySpec(pbeam.PrivacySpecParams{
		AggregationEpsilon:        epsilon / 2,
		PartitionSelectionEpsilon: epsilon / 2,
		PartitionSelectionDelta:   delta,
	})
	if err != nil {
		log.Fatalf("Couldn't create a PrivacySpec: %v", err)
	}
	pCol := pbeam.MakePrivateFromStruct(s, col, spec, "VisitorID")

	visitHours := pbeam.ParDo(s, extractVisitHourFn, pCol)
	visitsPerHour := pbeam.Count(s, visitHours, pbeam.CountParams{
		MaxPartitionsContributed: 1, // Visitors can visit the restaurant once (one hour) a day
		MaxValue:                 1, // Visitors can visit the restaurant once within an hour
	})
	return visitsPerHour
}
