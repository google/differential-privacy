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

// Package examples contains different usage examples of the DP library and related utilities.
package examples

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/google/differential-privacy/go/v3/dpagg"
	"github.com/google/differential-privacy/go/v3/noise"
)

const (
	// openingHour is the hour when visitors start entering the restaurant.
	openingHour = 9
	// closingHour is the hour when visitors stop entering the restaurant.
	closingHour = 20
	// Number of weekly visits for a visitor is limited to 3. All exceeding visits will be discarded.
	maxThreeVisitsPerWeek = 3
	// Number of weekly visits for a visitor is limited to 4. All exceeding visits will be discarded.
	maxFourVisitsPerWeek = 4
	// Minimum amount of money we expect a visitor to spend on a single visit.
	minEurosSpent = 0
	// Maximum amount of money we expect a visitor to spend on a single visit.
	maxEurosSpent = 50
)

var (
	ln3 = math.Log(3)
	// 1 - Monday, 2 - Tuesday, 3 - Wednesday, 4 - Thursday, 5 - Friday, 6 - Saturday, 7 - Sunday
	weekDays = []int64{1, 2, 3, 4, 5, 6, 7}
)

// Scenario is an interface for codelab scenarios.
type Scenario interface {
	getNonPrivateResults(visits []Visit) map[int64]int64
	getPrivateResults(visits []Visit) (map[int64]int64, error)
}

// RunScenario runs scenario sc on the input file and writes private and non-private results to the output files.
func RunScenario(sc Scenario, inputFile, nonPrivateResultsOutputFile, privateResultsOutputFile string) error {
	visits, err := readVisitsFromCSV(inputFile)
	if err != nil {
		return err
	}

	nonPrResults := sc.getNonPrivateResults(visits)
	prResults, err := sc.getPrivateResults(visits)
	if err != nil {
		return err
	}

	err = writeResultsToCSV(nonPrResults, nonPrivateResultsOutputFile)
	if err != nil {
		return err
	}
	return writeResultsToCSV(prResults, privateResultsOutputFile)
}

// CountVisitsPerHourScenario loads the input file, calculates non-anonymized and
// anonymized counts of visitors entering a restaurant every hour, and prints results
// to nonPrivateResultsOutputFile and privateResultsOutputFile.
// Uses dpagg.Count for calculating anonymized counts.
type CountVisitsPerHourScenario struct{}

// Calculates the raw count of the given visits per hour of day.
// Returns the map that maps an hour to a raw count of visits for the hours between
// OpeningHour and ClosingHour.
func (sc *CountVisitsPerHourScenario) getNonPrivateResults(dayVisits []Visit) map[int64]int64 {
	counts := make(map[int64]int64)
	for _, visit := range dayVisits {
		h := visit.VisitTime.Hour()
		counts[int64(h)]++
	}
	return counts
}

// Calculates the anonymized (i.e., "private") counts of the given visits per hour of day.
// Returns the map that maps an hour to an anonymized count of visits for the hours between
// OpeningHour and ClosingHour.
func (sc *CountVisitsPerHourScenario) getPrivateResults(dayVisits []Visit) (map[int64]int64, error) {
	hourToDpCount := make(map[int64]*dpagg.Count)

	var err error
	for h := int64(openingHour); h <= closingHour; h++ {
		// Construct dpagg.Count objects which will be used to calculate DP counts.
		// One dpagg.Count is created for every work hour.
		hourToDpCount[h], err = dpagg.NewCount(&dpagg.CountOptions{
			Epsilon:                  ln3,
			MaxPartitionsContributed: 1,
			Noise:                    noise.Laplace(),
		})
		if err != nil {
			return nil, fmt.Errorf("couldn't initialize count for hour %d: %w", h, err)
		}
	}

	for _, visit := range dayVisits {
		h := visit.VisitTime.Hour()
		hourToDpCount[int64(h)].Increment()
	}

	privateCounts := make(map[int64]int64)
	for h, dpCount := range hourToDpCount {
		privateCounts[h], err = dpCount.Result()
		if err != nil {
			return nil, fmt.Errorf("couldn't compute dp count: %w", err)
		}
	}

	return privateCounts, nil
}

// CountVisitsPerDayScenario loads the input file, calculates non-anonymized and
// anonymized counts of visits per week day, and prints results to
// nonPrivateResultsOutputFile and privateResultsOutputFile.
// Assumes that a visitor may enter the restaurant only once per day,
// but may enter multiple days per week.
// Uses dpagg.Count for calculating anonymized counts.
type CountVisitsPerDayScenario struct{}

// Calculates the raw count of the given number of visits per day.
// Returns the map that maps a day to a raw count of visits in this day.
func (sc *CountVisitsPerDayScenario) getNonPrivateResults(weekVisits []Visit) map[int64]int64 {
	counts := make(map[int64]int64)
	for _, visit := range weekVisits {
		counts[int64(visit.Day)]++
	}
	return counts
}

// Calculates the anonymized (i.e., "private") count of the given visits per day.
// Returns the map that maps a day to an anonymized count of visits in this day.
func (sc *CountVisitsPerDayScenario) getPrivateResults(weekVisits []Visit) (map[int64]int64, error) {
	dayToDpCount := make(map[int64]*dpagg.Count)

	var err error
	for _, day := range weekDays {
		// Construct dpagg.Count objects which will be used to calculate the DP counts.
		// One dpagg.Count is created for every work hour.
		dayToDpCount[day], err = dpagg.NewCount(&dpagg.CountOptions{
			Epsilon: ln3,
			// The data was pre-processed so that
			// each visitor may visit the restaurant up to maxThreeVisitsPerWeek times per week.
			// Hence, each visitor may contribute to up to maxThreeVisitsPerWeek daily visit counts.
			// Note: while the library accepts this limit as a configurable parameter,
			// it doesn't pre-process the data to ensure this limit is respected.
			// It is responsibility of the caller to ensure the data passed to the library
			// is capped for getting the correct privacy guarantee.
			// TODO: Clarify the note above.
			MaxPartitionsContributed: maxThreeVisitsPerWeek,
			Noise:                    noise.Laplace(),
		})
		if err != nil {
			return nil, fmt.Errorf("couldn't initialize count for day %d: %w", day, err)
		}
	}

	// Pre-process the data set by limiting the number of visits to maxThreeVisitsPerWeek
	// per VisitorId.
	boundedVisits := boundVisits(weekVisits, maxThreeVisitsPerWeek)
	for _, visit := range boundedVisits {
		dayToDpCount[int64(visit.Day)].Increment()
	}

	privateCounts := make(map[int64]int64)
	for day, dpCount := range dayToDpCount {
		privateCounts[day], err = dpCount.Result()
		if err != nil {
			return nil, fmt.Errorf("couldn't compute dp count: %w", err)
		}
	}

	return privateCounts, nil
}

// SumRevenuePerDayScenario loads the input file, calculates non-anonymized and
// anonymized amount of money spent by the visitors per weekday, and prints results
// to nonPrivateResultsOutputFile and privateResultsOutputFile.
// Assumes that a visitor may enter the restaurant at most once per day,
// but may enter multiple days per week.
// Uses dpagg.BoundedSumInt64 for calculating anonymized sums.
type SumRevenuePerDayScenario struct {
	Scenario
}

// Calculates the total raw revenue for each day of the week.
// Returns the map that maps a day to a raw revenue for this day.
func (sc *SumRevenuePerDayScenario) getNonPrivateResults(visits []Visit) map[int64]int64 {
	sums := make(map[int64]int64)
	for _, visit := range visits {
		sums[int64(visit.Day)] += visit.EurosSpent
	}
	return sums
}

// Calculates the  anonymized revenue for each day of the week.
// Returns the map that maps a day to an anonymized revenue for this day.
func (sc *SumRevenuePerDayScenario) getPrivateResults(visits []Visit) (map[int64]int64, error) {
	dayToBoundedSum := make(map[int64]*dpagg.BoundedSumInt64)

	var err error
	for _, day := range weekDays {
		// Construct dpagg.BoundedSumInt64 objects that will be used to calculate DP sums.
		// One dpagg.BoundedSumInt64 is created for every day.
		dayToBoundedSum[day], err = dpagg.NewBoundedSumInt64(&dpagg.BoundedSumInt64Options{
			Epsilon: ln3,
			// The data was pre-processed so that
			// each visitor may visit the restaurant up to maxFourVisitsPerWeek times per week.
			// Hence, each privacy unit may contribute to up to maxFourVisitsPerWeek daily counts.
			// Note: while the library accepts this limit as a configurable parameter,
			// it doesn't pre-process the data to ensure this limit is respected.
			// It is responsibility of the caller to ensure the data passed to the library
			// is capped for getting the correct privacy guarantee.
			// TODO: Clarify the note above.
			MaxPartitionsContributed: maxFourVisitsPerWeek,
			// No need to pre-process the data: BoundedSumInt64 will clamp the input values.
			Lower: minEurosSpent,
			Upper: maxEurosSpent,
			Noise: noise.Laplace(),
		})
		if err != nil {
			return nil, fmt.Errorf("couldn't initialize sum for day %d: %w", day, err)
		}
	}

	// Pre-process the data set by limiting the number of visits to maxFourVisitsPerWeek
	// per VisitorID.
	boundedVisits := boundVisits(visits, maxFourVisitsPerWeek)
	d := make(map[int]int64)

	for _, v := range boundedVisits {
		d[v.Day]++
	}
	for _, visit := range boundedVisits {
		dayToBoundedSum[int64(visit.Day)].Add(visit.EurosSpent)
	}

	privateSums := make(map[int64]int64)
	for day, boundedSum := range dayToBoundedSum {
		privateSums[day], err = boundedSum.Result()
		if err != nil {
			return nil, fmt.Errorf("couldn't compute dp sum: %w", err)
		}
	}

	return privateSums, nil
}

// CountVisitsPerCertainDurationScenario loads the input file and calculates the non-anonymized
// and anonymized count of visits per certain duration during the week.
// Assumes that a visitor may enter the restaurant at most once per day,
// but may enter multiple times per week.
// Uses dpagg.BoundedSumInt64 for calculating anonymized counts because the visitor can
// contribute to the same duration multiple times and therefore dpagg.Count can't be used
// for this.
type CountVisitsPerCertainDurationScenario struct{}

// Calculates the total count of visits for each exixting visit duration during the week.
// Returns the map that maps a duration to a raw count of visits for this duration.
func (sc *CountVisitsPerCertainDurationScenario) getNonPrivateResults(weekVisits []Visit) map[int64]int64 {
	counts := make(map[int64]int64)
	for _, visit := range weekVisits {
		counts[roundMinutes(visit.MinutesSpent, 10)]++
	}
	return counts
}

// Calculates the total anonymized count of visits for each existing visit duration during the week and
// eliminate the durations of time whereby too few visitors spent time at the restaurant.
// Returns the map that maps a duration to an anonymized count of visits for this duration.
func (sc *CountVisitsPerCertainDurationScenario) getPrivateResults(weekVisits []Visit) (map[int64]int64, error) {
	// Pre-process the data set by limiting the number of visits to maxThreeVisitsPerWeek
	// per VisitorID.
	boundedVisits := boundVisits(weekVisits, maxThreeVisitsPerWeek)

	durationToSelectPartition := make(map[int64]*dpagg.PreAggSelectPartition)
	durationToBoundedSum := make(map[int64]*dpagg.BoundedSumInt64)

	// Pre-aggregate data, since a visitor can contribute multiple times to a single partition.
	visitorIDToDurationToVisitCount := make(map[int64]map[int64]int64)
	for _, visit := range boundedVisits {
		durationToVisitCount, ok := visitorIDToDurationToVisitCount[visit.VisitorID]
		if !ok {
			durationToVisitCount = make(map[int64]int64)
			visitorIDToDurationToVisitCount[visit.VisitorID] = durationToVisitCount
		}

		duration := roundMinutes(visit.MinutesSpent, 10)
		durationToVisitCount[duration]++

		var err error
		_, ok = durationToSelectPartition[duration]
		if !ok {
			// Recall that each possible visit duration may be represented by a
			// partition. Here, we construct DP PreAggSelectPartition objects for each
			// such duration. PreAggSelectPartition decides whether there are enough
			// visitors in a partition to warrant keeping it, or if there were too few
			// visitors who stayed for that duration, meaning the partition should be
			// dropped.
			//
			// We use epsilon = log(3) / 2 in this example,
			// because we must split epsilon between all the functions that apply differential privacy,
			// which, in this case, is 2 functions: BoundedSumInt64 and PreAggSelectPartition.
			durationToSelectPartition[duration], err = dpagg.NewPreAggSelectPartition(&dpagg.PreAggSelectPartitionOptions{
				Epsilon:                  ln3 / 2,
				Delta:                    0.02,
				MaxPartitionsContributed: maxThreeVisitsPerWeek,
			})
			if err != nil {
				return nil, fmt.Errorf("couldn't initialize PreAggSelectPartition for duration %d: %w", duration, err)
			}

			// Construct dpagg.BoundedSumInt64 objects which will be used to calculate DP
			// counts with multiple contributions from a single privacy unit (visitor).
			// One dpagg.BoundedSumInt64 is created for every duration.
			// We use epsilon = log(3) / 2 in this example,
			// because we must split epsilon between all the functions that apply differential privacy,
			// which, in this case, is 2 functions: BoundedSumInt64 and PreAggSelectPartition.
			durationToBoundedSum[duration], err = dpagg.NewBoundedSumInt64(&dpagg.BoundedSumInt64Options{
				Epsilon:                  ln3 / 2,
				MaxPartitionsContributed: maxThreeVisitsPerWeek,
				Lower:                    0,
				Upper:                    maxThreeVisitsPerWeek,
				Noise:                    noise.Laplace(),
			})
			if err != nil {
				return nil, fmt.Errorf("couldn't initialize sum for duration %d: %w", duration, err)
			}
		}
	}

	for _, visits := range visitorIDToDurationToVisitCount {
		for duration, totalVisits := range visits {
			durationToBoundedSum[duration].Add(totalVisits)
			// Count distinct visitors for each duration.
			durationToSelectPartition[duration].Increment()
		}
	}

	privateSums := make(map[int64]int64)
	for duration, boundedSum := range durationToBoundedSum {
		// Pre-aggregation partition selection.
		// If there are enough visitors within this duration,
		// then it will appear in the result statistics table.
		// Otherwise, the duration's partition is simply dropped and excluded from the result.
		shouldKeepPartition, err := durationToSelectPartition[duration].ShouldKeepPartition()
		if err != nil {
			return nil, fmt.Errorf("couldn't compute shouldKeepPartition: %w", err)
		}
		if shouldKeepPartition {
			privateSums[duration], err = boundedSum.Result()
			if err != nil {
				return nil, fmt.Errorf("couldn't compute dp sum: %w", err)
			}
		}
	}

	return privateSums, nil
}

// Round up to the next 10-minute mark based on a divider
// e.g divider = 10 then
// for minutes = 1 => 10
// for minutes = 10 => 10
// for minutes = 16 => 20
// for minutes = 20 => 20
// etc
func roundMinutes(minutes int64, divider int64) int64 {
	quotient := minutes / divider
	if minutes%divider > 0 {
		quotient++
	}

	return quotient * divider
}

func boundVisits(initialVisits []Visit, maxVisitsPerWeek int64) []Visit {
	boundedVisits := make([]Visit, 0)

	// Shuffle all visits.
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(initialVisits), func(i, j int) { initialVisits[i], initialVisits[j] = initialVisits[j], initialVisits[i] })

	// Go through the unordered collection of visits, but only add, at most,
	// maxVisitsPerWeek visits per VisitorID to the final result, and
	// discard all additional visits.
	visitorIDToVisitCount := make(map[int64]int64)
	for _, visit := range initialVisits {
		val, ok := visitorIDToVisitCount[visit.VisitorID]
		if !ok {
			val = 1
		} else {
			val++
		}

		if val <= maxVisitsPerWeek {
			visitorIDToVisitCount[visit.VisitorID] = val
			boundedVisits = append(boundedVisits, visit)
		}
	}

	return boundedVisits
}
