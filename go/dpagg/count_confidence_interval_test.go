//
// Copyright 2021 Google LLC
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

package dpagg

import (
	"testing"

	"github.com/google/differential-privacy/go/v3/noise"
)

func getCount(t *testing.T, n noise.Noise) *Count {
	t.Helper()
	delta := arbitraryDelta
	if n == noise.Laplace() {
		delta = 0.0
	}
	c, err := NewCount(&CountOptions{
		Epsilon:                  arbitraryEpsilon,
		Delta:                    delta,
		Noise:                    n,
		MaxPartitionsContributed: arbitraryMaxPartitionsContributed})
	if err != nil {
		t.Fatalf("Couldn't get count with noise=%v: %v", n, err)
	}
	return c
}

// Tests that Count.ComputeConfidenceInterval() does not return negative intervals.
func TestCountComputeConfidenceInterval_ClampsNegativeSubinterval(t *testing.T) {
	for i := 0; i < 1000; i++ {
		count := getCount(t, noise.Gaussian())
		_, err := count.Result()
		if err != nil {
			t.Fatalf("Couldn't compute dp result: %v", err)
		}

		// Using a large alpha to get a small confidence interval. This increases the chance of both
		// the lower and the upper bound being clamped.
		confInt, err := count.ComputeConfidenceInterval(0.99)
		if err != nil {
			t.Fatalf("Couldn't compute confidence interval: %v", err)
		}
		if confInt.LowerBound < 0 || confInt.UpperBound < 0 {
			t.Errorf("Confidence interval=%+v should have non-negative lower and upper bounds", confInt)
		}

		// Using a small alpha to get a large confidence interval. This increases the chance of only
		// the the upper bound being clamped.
		confInt, err = count.ComputeConfidenceInterval(0.01)
		if err != nil {
			t.Fatalf("Couldn't compute confidence interval: %v", err)
		}
		if confInt.LowerBound < 0 || confInt.UpperBound < 0 {
			t.Errorf("Confidence interval=%+v should have non-negative lower and upper bounds", confInt)
		}
	}
}

// Tests that Count.ComputeConfidenceInterval() returns the same interval when called for the same alpha twice.
func TestCountComputeConfidenceInterval_ReturnsSameResultForSameAlpha(t *testing.T) {
	for _, tc := range []struct {
		n         noise.Noise
		trueCount int64
	}{
		{noise.Gaussian(), 0}, // Clamping possible
		{noise.Gaussian(), 100_000_000},
		{noise.Laplace(), 0}, // Clamping possible
		{noise.Laplace(), 100_000_000},
	} {
		count := getCount(t, tc.n)
		count.IncrementBy(tc.trueCount)
		_, err := count.Result()
		if err != nil {
			t.Fatalf("With %v, couldn't compute dp result: %v", tc.n, err)
		}

		confInt1, err := count.ComputeConfidenceInterval(arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		confInt2, err := count.ComputeConfidenceInterval(arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		if confInt1.LowerBound != confInt2.LowerBound || confInt1.UpperBound != confInt2.UpperBound {
			t.Errorf("With %v, expected confInt1=%+v and confInt2=%+v with the same alpha to be equal", tc.n, confInt1, confInt2)
		}
	}
}

// Tests that Count.ComputeConfidenceInterval()'s result for small alpha is contained in the result
// for large alpha.
func TestCountComputeConfidenceInterval_ResultForSmallAlphaContainedInResultForLargeAlpha(t *testing.T) {
	for _, tc := range []struct {
		n         noise.Noise
		trueCount int64
	}{
		{noise.Gaussian(), 0}, // Clamping possible
		{noise.Gaussian(), 100_000_000},
		{noise.Laplace(), 0}, // Clamping possible
		{noise.Laplace(), 100_000_000},
	} {
		count := getCount(t, tc.n)
		count.IncrementBy(tc.trueCount)
		_, err := count.Result()
		if err != nil {
			t.Fatalf("With %v, couldn't compute dp result: %v", tc.n, err)
		}

		smallAlphaConfInt, err := count.ComputeConfidenceInterval(arbitraryAlpha * 0.5)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		largeAlphaConfInt, err := count.ComputeConfidenceInterval(arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		if smallAlphaConfInt.LowerBound > largeAlphaConfInt.LowerBound {
			t.Errorf("With %v, expected smallAlphaConfInt's (%+v) lower bound to be smaller than largeAlphaConfInt's (%v) lower bound", tc.n, smallAlphaConfInt, largeAlphaConfInt)
		}
		if smallAlphaConfInt.UpperBound < largeAlphaConfInt.UpperBound {
			t.Errorf("With %v, expected smallAlphaConfInt's (%+v) upper bound to be larger than largeAlphaConfInt's (%v) upper bound", tc.n, smallAlphaConfInt, largeAlphaConfInt)
		}
	}
}

// Tests that Count.ComputeConfidenceInterval() matches the underlying noise implementation's
// confidence interval with the same parameters.
func TestCountComputeConfidenceInterval_MatchesNoiseConfidenceInterval(t *testing.T) {
	for _, tc := range []struct {
		n     noise.Noise
		delta float64
	}{
		{noise.Gaussian(), arbitraryDelta},
		{noise.Laplace(), 0.0},
	} {
		lInf := int64(1) // Always 1 for Count.
		count := getCount(t, tc.n)
		err := count.IncrementBy(100_000_000) // incrementing by large number to prevent clamping
		if err != nil {
			t.Fatalf("With %v, couldn't increment count: %v", tc.n, err)
		}
		result, err := count.Result()
		if err != nil {
			t.Fatalf("With %v, couldn't compute dp result: %v", tc.n, err)
		}

		countConfInt, err := count.ComputeConfidenceInterval(arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		noiseConfInt, err := tc.n.ComputeConfidenceIntervalInt64(result, arbitraryMaxPartitionsContributed, lInf, arbitraryEpsilon, tc.delta, arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		if countConfInt.LowerBound != noiseConfInt.LowerBound || countConfInt.UpperBound != noiseConfInt.UpperBound {
			t.Errorf("With %v, countConfInt (%+v) and noiseConfInt (%+v) to be equal", tc.n, countConfInt, noiseConfInt)
		}
	}
}

// Tests that Count.ComputeConfidenceInterval() satisfies the confidence level for a given alpha.
func TestCountComputeConfidenceInterval_SatisfiesConfidenceLevel(t *testing.T) {
	rawCount := int64(14523)
	for _, tc := range []struct {
		n        noise.Noise
		alpha    float64
		wantHits int
	}{
		// Assuming that the true alpha of the confidence interval mechanism is 0.1, i.e., the raw value
		// is within the confidence interval with probability of at least 0.9, then the hits count will
		// be at least 89546 with probability greater than 1 - 10⁻⁶.
		{noise.Gaussian(), 0.1, 89546},
		{noise.Laplace(), 0.1, 89546},
		// Assuming that the true alpha of the confidence interval mechanism is 0.9, i.e., the raw value
		// is within the confidence interval with probability of at least 0.1, then the hits count will
		// be at least 9552 with probability greater than 1 - 10⁻⁶.
		{noise.Gaussian(), 0.9, 9552},
		{noise.Laplace(), 0.9, 9552},
	} {
		hits := 0
		for i := 0; i < 100000; i++ {
			count := getCount(t, tc.n)
			err := count.IncrementBy(rawCount)
			if err != nil {
				t.Fatalf("With %v, couldn't increment count: %v", tc.n, err)
			}
			_, err = count.Result()
			if err != nil {
				t.Fatalf("With %v, couldn't compute dp result: %v", tc.n, err)
			}

			confInt, err := count.ComputeConfidenceInterval(tc.alpha)
			if err != nil {
				t.Fatalf("With noise=%v alpha=%f, couldn't compute confidence interval: %v", tc.n, tc.alpha, err)
			}

			if confInt.LowerBound <= float64(rawCount) && float64(rawCount) <= confInt.UpperBound {
				hits++
			}
		}
		if hits < tc.wantHits {
			t.Errorf("With noise=%v alpha=%f, got %d hits, i.e. raw output within the confidence interval, wanted at least %d", tc.n, tc.alpha, hits, tc.wantHits)
		}
	}
}

// Tests that Count.ComputeConfidenceInterval() returns errors correctly with different Count aggregation states.
func TestCountComputeConfidenceInterval_StateChecks(t *testing.T) {
	for _, tc := range []struct {
		state   aggregationState
		wantErr bool
	}{
		{resultReturned, false},
		{defaultState, true},
		{merged, true},
		{serialized, true},
	} {
		c := getNoiselessCount(t)
		c.state = tc.state

		if _, err := c.ComputeConfidenceInterval(0.1); (err != nil) != tc.wantErr {
			t.Errorf("ComputeConfidenceInterval: when state %v for err got %v, wantErr %t", tc.state, err, tc.wantErr)
		}
	}
}
