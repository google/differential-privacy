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

func getBoundedSumInt64(t *testing.T, n noise.Noise, lower, upper int64) *BoundedSumInt64 {
	t.Helper()
	delta := arbitraryDelta
	if n == noise.Laplace() {
		delta = 0.0
	}
	bs, err := NewBoundedSumInt64(&BoundedSumInt64Options{
		Epsilon:                      arbitraryEpsilon,
		Delta:                        delta,
		Noise:                        n,
		MaxPartitionsContributed:     arbitraryMaxPartitionsContributed,
		maxContributionsPerPartition: arbitraryMaxContributionsPerPartition,
		Lower:                        lower,
		Upper:                        upper})
	if err != nil {
		t.Fatalf("Couldn't get bounded sum with noise=%v lower=%d upper=%d: %v", n, lower, upper, err)
	}
	return bs
}

func getBoundedSumFloat64(t *testing.T, n noise.Noise, lower, upper float64) *BoundedSumFloat64 {
	t.Helper()
	delta := arbitraryDelta
	if n == noise.Laplace() {
		delta = 0.0
	}
	bs, err := NewBoundedSumFloat64(&BoundedSumFloat64Options{
		Epsilon:                      arbitraryEpsilon,
		Delta:                        delta,
		Noise:                        n,
		MaxPartitionsContributed:     arbitraryMaxPartitionsContributed,
		maxContributionsPerPartition: arbitraryMaxContributionsPerPartition,
		Lower:                        lower,
		Upper:                        upper})
	if err != nil {
		t.Fatalf("Couldn't get bounded sum with noise=%v lower=%f upper=%f: %v", n, lower, upper, err)
	}
	return bs
}

// Tests that BoundedSumInt64.ComputeConfidenceInterval() does not return negative intervals when
// bound are non-negative.
func TestSumInt64ComputeConfidenceInterval_ClampsNegativeSubinterval(t *testing.T) {
	for i := 0; i < 1000; i++ {
		bs := getBoundedSumInt64(t, noise.Gaussian(), 0, 1)
		_, err := bs.Result()
		if err != nil {
			t.Fatalf("Couldn't compute dp result: %v", err)
		}

		// Using a large alpha to get a small confidence interval. This increases the chance of both
		// the lower and the upper bound being clamped.
		confInt, err := bs.ComputeConfidenceInterval(0.99)
		if err != nil {
			t.Fatalf("Couldn't compute confidence interval: %v", err)
		}
		if confInt.LowerBound < 0 || confInt.UpperBound < 0 {
			t.Errorf("Confidence interval=%+v should have non-negative lower and upper bounds", confInt)
		}

		// Using a small alpha to get a large confidence interval. This increases the chance of only
		// the the upper bound being clamped.
		confInt, err = bs.ComputeConfidenceInterval(0.01)
		if err != nil {
			t.Fatalf("Couldn't compute confidence interval: %v", err)
		}
		if confInt.LowerBound < 0 || confInt.UpperBound < 0 {
			t.Errorf("Confidence interval=%+v should have non-negative lower and upper bounds", confInt)
		}
	}
}

// Tests that BoundedSumFloat64.ComputeConfidenceInterval() does not return negative intervals when
// bound are non-negative.
func TestSumFloat64ComputeConfidenceInterval_ClampsNegativeSubinterval(t *testing.T) {
	for i := 0; i < 1000; i++ {
		bs := getBoundedSumFloat64(t, noise.Gaussian(), 0.0, 1.0)
		_, err := bs.Result()
		if err != nil {
			t.Fatalf("Couldn't compute dp result: %v", err)
		}

		// Using a large alpha to get a small confidence interval. This increases the chance of both
		// the lower and the upper bound being clamped.
		confInt, err := bs.ComputeConfidenceInterval(0.99)
		if err != nil {
			t.Fatalf("Couldn't compute confidence interval: %v", err)
		}
		if confInt.LowerBound < 0 || confInt.UpperBound < 0 {
			t.Errorf("Confidence interval=%+v should have non-negative lower and upper bounds", confInt)
		}

		// Using a small alpha to get a large confidence interval. This increases the chance of only
		// the the upper bound being clamped.
		confInt, err = bs.ComputeConfidenceInterval(0.01)
		if err != nil {
			t.Fatalf("Couldn't compute confidence interval: %v", err)
		}
		if confInt.LowerBound < 0 || confInt.UpperBound < 0 {
			t.Errorf("Confidence interval=%+v should have non-negative lower and upper bounds", confInt)
		}
	}
}

// Tests that BoundedSumInt64.ComputeConfidenceInterval() does not return positive intervals when
// bound are non-positive.
func TestSumInt64ComputeConfidenceInterval_ClampsPositiveSubinterval(t *testing.T) {
	for i := 0; i < 1000; i++ {
		bs := getBoundedSumInt64(t, noise.Gaussian(), -1, 0)
		_, err := bs.Result()
		if err != nil {
			t.Fatalf("Couldn't compute dp result: %v", err)
		}

		// Using a large alpha to get a small confidence interval. This increases the chance of both
		// the lower and the upper bound being clamped.
		confInt, err := bs.ComputeConfidenceInterval(0.99)
		if err != nil {
			t.Fatalf("Couldn't compute confidence interval: %v", err)
		}
		if confInt.LowerBound > 0 || confInt.UpperBound > 0 {
			t.Errorf("Confidence interval=%+v should have non-positive lower and upper bounds", confInt)
		}

		// Using a small alpha to get a large confidence interval. This increases the chance of only
		// the the upper bound being clamped.
		confInt, err = bs.ComputeConfidenceInterval(0.01)
		if err != nil {
			t.Fatalf("Couldn't compute confidence interval: %v", err)
		}
		if confInt.LowerBound > 0 || confInt.UpperBound > 0 {
			t.Errorf("Confidence interval=%+v should have non-positive lower and upper bounds", confInt)
		}
	}
}

// Tests that BoundedSumFloat64.ComputeConfidenceInterval() does not return positive intervals when
// bound are non-positive.
func TestSumFloat64ComputeConfidenceInterval_ClampsPositiveSubinterval(t *testing.T) {
	for i := 0; i < 1000; i++ {
		bs := getBoundedSumFloat64(t, noise.Gaussian(), -1.0, 0.0)
		_, err := bs.Result()
		if err != nil {
			t.Fatalf("Couldn't compute dp result: %v", err)
		}

		// Using a large alpha to get a small confidence interval. This increases the chance of both
		// the lower and the upper bound being clamped.
		confInt, err := bs.ComputeConfidenceInterval(0.99)
		if err != nil {
			t.Fatalf("Couldn't compute confidence interval: %v", err)
		}
		if confInt.LowerBound > 0 || confInt.UpperBound > 0 {
			t.Errorf("Confidence interval=%+v should have non-positive lower and upper bounds", confInt)
		}

		// Using a small alpha to get a large confidence interval. This increases the chance of only
		// the the upper bound being clamped.
		confInt, err = bs.ComputeConfidenceInterval(0.01)
		if err != nil {
			t.Fatalf("Couldn't compute confidence interval: %v", err)
		}
		if confInt.LowerBound > 0 || confInt.UpperBound > 0 {
			t.Errorf("Confidence interval=%+v should have non-positive lower and upper bounds", confInt)
		}
	}
}

// Tests that BoundedSumInt64.ComputeConfidenceInterval() returns the same interval when called for
// the same alpha twice.
func TestSumInt64ComputeConfidenceInterval_ReturnsSameResultForSameAlpha(t *testing.T) {
	for _, tc := range []struct {
		n noise.Noise
	}{
		{noise.Gaussian()},
		{noise.Laplace()},
	} {
		bs := getBoundedSumInt64(t, tc.n, arbitraryLowerInt64, arbitraryUpperInt64)
		_, err := bs.Result()
		if err != nil {
			t.Fatalf("Couldn't compute dp result: %v", err)
		}

		confInt1, err := bs.ComputeConfidenceInterval(arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		confInt2, err := bs.ComputeConfidenceInterval(arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		if confInt1.LowerBound != confInt2.LowerBound || confInt1.UpperBound != confInt2.UpperBound {
			t.Errorf("With %v, expected confInt1=%+v and confInt2=%+v with the same alpha to be equal", tc.n, confInt1, confInt2)
		}
	}
}

// Tests that BoundedSumFloat64.ComputeConfidenceInterval() returns the same interval when called for
// the same alpha twice.
func TestSumFloat64ComputeConfidenceInterval_ReturnsSameResultForSameAlpha(t *testing.T) {
	for _, tc := range []struct {
		n noise.Noise
	}{
		{noise.Gaussian()},
		{noise.Laplace()},
	} {
		bs := getBoundedSumFloat64(t, tc.n, arbitraryLower, arbitraryUpper)
		_, err := bs.Result()
		if err != nil {
			t.Fatalf("Couldn't compute dp result: %v", err)
		}

		confInt1, err := bs.ComputeConfidenceInterval(arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		confInt2, err := bs.ComputeConfidenceInterval(arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		if confInt1.LowerBound != confInt2.LowerBound || confInt1.UpperBound != confInt2.UpperBound {
			t.Errorf("With %v, expected confInt1=%+v and confInt2=%+v with the same alpha to be equal", tc.n, confInt1, confInt2)
		}
	}
}

// Tests that BoundedSumInt64.ComputeConfidenceInterval()'s result for small alpha is contained in
// the result for large alpha.
func TestSumInt64ComputeConfidenceInterval_ResultForSmallAlphaContainedInResultForLargeAlpha(t *testing.T) {
	for _, tc := range []struct {
		n noise.Noise
	}{
		{noise.Gaussian()},
		{noise.Laplace()},
	} {
		bs := getBoundedSumInt64(t, tc.n, arbitraryLowerInt64, arbitraryUpperInt64)
		_, err := bs.Result()
		if err != nil {
			t.Fatalf("Couldn't compute dp result: %v", err)
		}

		smallAlphaConfInt, err := bs.ComputeConfidenceInterval(arbitraryAlpha * 0.5)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		largeAlphaConfInt, err := bs.ComputeConfidenceInterval(arbitraryAlpha)
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

// Tests that BoundedSumFloat64.ComputeConfidenceInterval()'s result for small alpha is contained in
// the result for large alpha.
func TestSumFloat64ComputeConfidenceInterval_ResultForSmallAlphaContainedInResultForLargeAlpha(t *testing.T) {
	for _, tc := range []struct {
		n noise.Noise
	}{
		{noise.Gaussian()},
		{noise.Laplace()},
	} {
		bs := getBoundedSumFloat64(t, tc.n, arbitraryLower, arbitraryUpper)
		_, err := bs.Result()
		if err != nil {
			t.Fatalf("Couldn't compute dp result: %v", err)
		}

		smallAlphaConfInt, err := bs.ComputeConfidenceInterval(arbitraryAlpha * 0.5)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		largeAlphaConfInt, err := bs.ComputeConfidenceInterval(arbitraryAlpha)
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

// Tests that BoundedSumInt64.ComputeConfidenceInterval() matches the underlying noise implementation's
// confidence interval with the same parameters.
func TestSumInt64ComputeConfidenceInterval_MatchesNoiseConfidenceInterval(t *testing.T) {
	for _, tc := range []struct {
		n     noise.Noise
		delta float64
	}{
		{noise.Gaussian(), arbitraryDelta},
		{noise.Laplace(), 0.0},
	} {
		lInf := arbitraryMaxContributionsPerPartition * 2
		bs := getBoundedSumInt64(t, tc.n, -2, 1)
		// Lower and upper bound have different signs, so no clamping should occur.
		result, err := bs.Result()
		if err != nil {
			t.Fatalf("Couldn't compute dp result: %v", err)
		}

		bsConfInt, err := bs.ComputeConfidenceInterval(arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		noiseConfInt, err := tc.n.ComputeConfidenceIntervalInt64(result, arbitraryMaxPartitionsContributed, lInf, arbitraryEpsilon, tc.delta, arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		if bsConfInt.LowerBound != noiseConfInt.LowerBound || bsConfInt.UpperBound != noiseConfInt.UpperBound {
			t.Errorf("With %v, bsConfInt (%+v) and noiseConfInt (%+v) to be equal", tc.n, bsConfInt, noiseConfInt)
		}
	}
}

// Tests that BoundedSumFloat64.ComputeConfidenceInterval() matches the underlying noise implementation's
// confidence interval with the same parameters.
func TestSumFloat64ComputeConfidenceInterval_MatchesNoiseConfidenceInterval(t *testing.T) {
	for _, tc := range []struct {
		n     noise.Noise
		delta float64
	}{
		{noise.Gaussian(), arbitraryDelta},
		{noise.Laplace(), 0.0},
	} {
		lInf := float64(arbitraryMaxContributionsPerPartition) * 4.63
		bs := getBoundedSumFloat64(t, tc.n, -1.0, 4.63)
		// Lower and upper bound have different signs, so no clamping should occur.
		result, err := bs.Result()
		if err != nil {
			t.Fatalf("Couldn't compute dp result: %v", err)
		}

		bsConfInt, err := bs.ComputeConfidenceInterval(arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		noiseConfInt, err := tc.n.ComputeConfidenceIntervalFloat64(result, arbitraryMaxPartitionsContributed, lInf, arbitraryEpsilon, tc.delta, arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		if bsConfInt.LowerBound != noiseConfInt.LowerBound || bsConfInt.UpperBound != noiseConfInt.UpperBound {
			t.Errorf("With %v, bsConfInt (%+v) and noiseConfInt (%+v) to be equal", tc.n, bsConfInt, noiseConfInt)
		}
	}
}

// Tests that BoundedSumInt64.ComputeConfidenceInterval() satisfies the confidence level for a given alpha.
func TestSumInt64ComputeConfidenceInterval_SatisfiesConfidenceLevel(t *testing.T) {
	rawValue := int64(1)
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
			bs := getBoundedSumInt64(t, tc.n, arbitraryLowerInt64, arbitraryUpperInt64)
			err := bs.Add(rawValue)
			if err != nil {
				t.Fatalf("Couldn't add to sum: %v", err)
			}
			_, err = bs.Result()
			if err != nil {
				t.Fatalf("Couldn't compute dp result: %v", err)
			}

			confInt, err := bs.ComputeConfidenceInterval(tc.alpha)
			if err != nil {
				t.Fatalf("With noise=%v alpha=%f, couldn't compute confidence interval: %v", tc.n, tc.alpha, err)
			}

			if confInt.LowerBound <= float64(rawValue) && float64(rawValue) <= confInt.UpperBound {
				hits++
			}
		}
		if hits < tc.wantHits {
			t.Errorf("With noise=%v alpha=%f, got %d hits, i.e. raw output within the confidence interval, wanted at least %d", tc.n, tc.alpha, hits, tc.wantHits)
		}
	}
}

// Tests that BoundedSumFloat64.ComputeConfidenceInterval() satisfies the confidence level for a given alpha.
func TestSumFloat64ComputeConfidenceInterval_SatisfiesConfidenceLevel(t *testing.T) {
	rawValue := 1.0
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
			bs := getBoundedSumFloat64(t, tc.n, arbitraryLower, arbitraryUpper)
			err := bs.Add(rawValue)
			if err != nil {
				t.Fatalf("Couldn't add to sum: %v", err)
			}
			_, err = bs.Result()
			if err != nil {
				t.Fatalf("Couldn't compute dp result: %v", err)
			}

			confInt, err := bs.ComputeConfidenceInterval(tc.alpha)
			if err != nil {
				t.Fatalf("With noise=%v alpha=%f, couldn't compute confidence interval: %v", tc.n, tc.alpha, err)
			}

			if confInt.LowerBound <= rawValue && rawValue <= confInt.UpperBound {
				hits++
			}
		}
		if hits < tc.wantHits {
			t.Errorf("With noise=%v alpha=%f, got %d hits, i.e. raw output within the confidence interval, wanted at least %d", tc.n, tc.alpha, hits, tc.wantHits)
		}
	}
}

// Tests that BoundedSumInt64.ComputeConfidenceInterval() returns errors correctly with different aggregation states.
func TestSumInt64ComputeConfidenceInterval_StateChecks(t *testing.T) {
	for _, tc := range []struct {
		state   aggregationState
		wantErr bool
	}{
		{resultReturned, false},
		{defaultState, true},
		{merged, true},
		{serialized, true},
	} {
		bs := getNoiselessBSI(t)
		bs.state = tc.state

		if _, err := bs.ComputeConfidenceInterval(0.1); (err != nil) != tc.wantErr {
			t.Errorf("ComputeConfidenceInterval: when state %v for err got %v, wantErr %t", tc.state, err, tc.wantErr)
		}
	}
}

// Tests that BoundedSumFloat64.ComputeConfidenceInterval() returns errors correctly with different aggregation states.
func TestSumFloat64ComputeConfidenceInterval_StateChecks(t *testing.T) {
	for _, tc := range []struct {
		state   aggregationState
		wantErr bool
	}{
		{resultReturned, false},
		{defaultState, true},
		{merged, true},
		{serialized, true},
	} {
		bs := getNoiselessBSF(t)
		bs.state = tc.state

		if _, err := bs.ComputeConfidenceInterval(0.1); (err != nil) != tc.wantErr {
			t.Errorf("ComputeConfidenceInterval: when state %v for err got %v, wantErr %t", tc.state, err, tc.wantErr)
		}
	}
}
