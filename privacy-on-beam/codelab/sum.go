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
	beam.RegisterFunction(extractVisitHourAndMoneySpent)
}

// RevenuePerHour calculates and returns the total money spent by visitors
// who entered the restaurant for each hour. This DOES NOT produce an anonymized output.
// Use PrivateRevenuePerHour for computing this in an anonymized way.
func RevenuePerHour(s beam.Scope, col beam.PCollection) beam.PCollection {
	s = s.Scope("RevenuePerHour")
	hourToMoneySpent := beam.ParDo(s, extractVisitHourAndMoneySpent, col)
	revenues := stats.SumPerKey(s, hourToMoneySpent)
	return revenues
}

func extractVisitHourAndMoneySpent(v Visit) (int, int) {
	return v.TimeEntered.Hour(), v.MoneySpent
}

// PrivateRevenuePerHour calculates and returns the total money spent by visitors
// who entered the restaurant for each hour in a differentially private way.
func PrivateRevenuePerHour(s beam.Scope, col beam.PCollection) beam.PCollection {
	s = s.Scope("PrivateRevenuePerHour")
	// Create a Privacy Spec and convert col into a PrivatePCollection
	spec := pbeam.NewPrivacySpec(epsilon, delta)
	pCol := pbeam.MakePrivateFromStruct(s, col, spec, "VisitorID")

	hourToMoneySpent := pbeam.ParDo(s, extractVisitHourAndTimeSpentFn, pCol)
	revenues := pbeam.SumPerKey(s, hourToMoneySpent, pbeam.SumParams{
		MaxPartitionsContributed: 1,  // Visitors can visit the restaurant once (one hour) a day
		MinValue:                 0,  // Minimum money spent per user (in dollars)
		MaxValue:                 40, // Maximum money spent per user (in dollars)
	})
	return revenues
}
