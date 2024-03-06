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
	"github.com/apache/beam/sdks/v2/go/pkg/beam/register"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/stats"
)

func init() {
	register.Function1x2[Visit, int, int](extractVisitHourAndMoneySpentFn)
}

// RevenuePerHour calculates and returns the total money spent by visitors
// who entered the restaurant for each hour. This DOES NOT produce an anonymized output.
// Use PrivateRevenuePerHour for computing this in an anonymized way.
func RevenuePerHour(s beam.Scope, col beam.PCollection) beam.PCollection {
	s = s.Scope("RevenuePerHour")
	hourToMoneySpent := beam.ParDo(s, extractVisitHourAndMoneySpentFn, col)
	revenues := stats.SumPerKey(s, hourToMoneySpent)
	return revenues
}

func extractVisitHourAndMoneySpentFn(v Visit) (int, int) {
	return v.TimeEntered.Hour(), v.MoneySpent
}

// PrivateRevenuePerHour calculates and returns the total money spent by visitors
// who entered the restaurant for each hour in a differentially private way.
func PrivateRevenuePerHour(s beam.Scope, col beam.PCollection) beam.PCollection {
	s = s.Scope("PrivateRevenuePerHour")
	// Create a Privacy Spec and convert col into a PrivatePCollection.
	spec, err := pbeam.NewPrivacySpec(pbeam.PrivacySpecParams{AggregationEpsilon: epsilon})
	if err != nil {
		log.Fatalf("Couldn't create a PrivacySpec: %v", err)
	}
	pCol := pbeam.MakePrivateFromStruct(s, col, spec, "VisitorID")

	// Create a PCollection of output partitions, i.e. restaurant's work hours (from 9 am till 9pm (exclusive)).
	hours := beam.CreateList(s, [12]int{9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20})

	hourToMoneySpent := pbeam.ParDo(s, extractVisitHourAndMoneySpentFn, pCol)
	revenues := pbeam.SumPerKey(s, hourToMoneySpent, pbeam.SumParams{
		MaxPartitionsContributed: 1,     // Visitors can visit the restaurant once (one hour) a day
		MinValue:                 0,     // Minimum money spent per user (in euros)
		MaxValue:                 40,    // Maximum money spent per user (in euros)
		PublicPartitions:         hours, // Visitors only visit during work hours
	})
	return revenues
}
