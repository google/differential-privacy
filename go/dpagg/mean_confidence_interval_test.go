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

	"github.com/google/differential-privacy/go/noise"
)

func getBoundedMeanFloat64(n noise.Noise, lower, upper float64) *BoundedMeanFloat64 {
	delta := arbitraryDelta
	if n == noise.Laplace() {
		delta = 0.0
	}
	return NewBoundedMeanFloat64(&BoundedMeanFloat64Options{
		Epsilon:                      arbitraryEpsilon,
		Delta:                        delta,
		Noise:                        n,
		MaxPartitionsContributed:     arbitraryMaxPartitionsContributed,
		MaxContributionsPerPartition: arbitraryMaxContributionsPerPartition,
		Lower:                        lower,
		Upper:                        upper})
}

// Tests that BoundedMeanFloat64.ComputeConfidenceInterval() returns valid bounds with empty
// inputs.
func TestMeanComputeConfidenceInterval_EmptyMeanClampsToBounds(t *testing.T) {
	for _, tc := range []struct {
		desc  string
		lower float64
		upper float64
	}{
		{"opposite sign bounds", -1.0, 1.0},
		{"positive bounds", 1.0, 2.0},
		{"negative bounds", -2.0, -1.0},
	} {
		for i := 0; i < 1000; i++ {
			// For empty instances of mean, the confidence interval of the denominator is likely to contain
			// negative values. This should not cause the mean's confidence interval to exceed the bounds.
			bm := getBoundedMeanFloat64(noise.Gaussian(), tc.lower, tc.upper)
			bm.Result()

			// Using a large alpha to get small confidence intervals. This increases the chance of the
			// denominator's confidence interval to be completely negative.
			confInt, err := bm.ComputeConfidenceInterval(0.99)
			if err != nil {
				t.Fatalf("With %s, couldn't compute confidence interval: %v", tc.desc, err)
			}
			if confInt.LowerBound > confInt.UpperBound {
				t.Errorf("With %s, confidence interval=%+v's lower bound should be smaller than its upper bound", tc.desc, confInt)
			}
			if confInt.LowerBound < tc.lower || confInt.UpperBound > tc.upper {
				t.Errorf("With %s, confidence interval=%+v should be contained between lower and upper bounds of BoundedMean", tc.desc, confInt)
			}

			// Using a small alpha to get large confidence intervals. This increases the chance of the
			// denominator's confidence interval to be partially negative.
			confInt, err = bm.ComputeConfidenceInterval(0.01)
			if err != nil {
				t.Fatalf("With %s, couldn't compute confidence interval: %v", tc.desc, err)
			}
			if confInt.LowerBound > confInt.UpperBound {
				t.Errorf("With %s, confidence interval=%+v's lower bound should be smaller than its upper bound", tc.desc, confInt)
			}
			if confInt.LowerBound < tc.lower || confInt.UpperBound > tc.upper {
				t.Errorf("With %s, confidence interval=%+v should be contained between lower and upper bounds of BoundedMean", tc.desc, confInt)
			}
		}
	}
}

// Tests that BoundedMeanFloat64.ComputeConfidenceInterval() returns valid bounds with raw
// value at bounds.
func TestMeanComputeConfidenceInterval_RawValueAtBoundsClampsToBounds(t *testing.T) {
	for _, tc := range []struct {
		desc     string
		rawValue float64
	}{
		{"raw value at lower bound", arbitraryLower},
		{"raw value at upper bound", arbitraryUpper},
	} {
		for i := 0; i < 1000; i++ {
			bm := getBoundedMeanFloat64(noise.Gaussian(), arbitraryLower, arbitraryUpper)
			bm.Add(tc.rawValue)
			bm.Result()

			// Using a large alpha to get small confidence intervals. This increases the chance of the
			// denominator's confidence interval to be completely negative.
			confInt, err := bm.ComputeConfidenceInterval(0.99)
			if err != nil {
				t.Fatalf("With %s, couldn't compute confidence interval: %v", tc.desc, err)
			}
			if confInt.LowerBound > confInt.UpperBound {
				t.Errorf("With %s, confidence interval=%+v's lower bound should be smaller than its upper bound", tc.desc, confInt)
			}
			if confInt.LowerBound < arbitraryLower || confInt.UpperBound > arbitraryUpper {
				t.Errorf("With %s, confidence interval=%+v should be contained between lower and upper bounds of BoundedMean", tc.desc, confInt)
			}

			// Using a small alpha to get large confidence intervals. This increases the chance of the
			// denominator's confidence interval to be partially negative.
			confInt, err = bm.ComputeConfidenceInterval(0.01)
			if err != nil {
				t.Fatalf("With %s, couldn't compute confidence interval: %v", tc.desc, err)
			}
			if confInt.LowerBound > confInt.UpperBound {
				t.Errorf("With %s, confidence interval=%+v's lower bound should be smaller than its upper bound", tc.desc, confInt)
			}
			if confInt.LowerBound < arbitraryLower || confInt.UpperBound > arbitraryUpper {
				t.Errorf("With %s, confidence interval=%+v should be contained between lower and upper bounds of BoundedMean", tc.desc, confInt)
			}
		}
	}
}

// Tests that BoundedMeanFloat64.ComputeConfidenceInterval() returns the same interval when called for
// the same alpha twice.
func TestMeanComputeConfidenceInterval_ReturnsSameResultForSameAlpha(t *testing.T) {
	for _, tc := range []struct {
		n noise.Noise
	}{
		{noise.Gaussian()},
		{noise.Laplace()},
	} {
		bm := getBoundedMeanFloat64(tc.n, arbitraryLower, arbitraryUpper)
		bm.Result()

		confInt1, err := bm.ComputeConfidenceInterval(arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		confInt2, err := bm.ComputeConfidenceInterval(arbitraryAlpha)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		if confInt1.LowerBound != confInt2.LowerBound || confInt1.UpperBound != confInt2.UpperBound {
			t.Errorf("With %v, expected confInt1=%+v and confInt2=%+v with the same alpha to be equal", tc.n, confInt1, confInt2)
		}
	}
}

// Tests that BoundedMeanFloat64.ComputeConfidenceInterval()'s result for small alpha is contained in
// the result for large alpha.
func TestMeanComputeConfidenceInterval_ResultForSmallAlphaContainedInResultForLargeAlpha(t *testing.T) {
	for _, tc := range []struct {
		n          noise.Noise
		numEntries int
	}{
		{noise.Gaussian(), 0}, // Clamping possible
		{noise.Gaussian(), 1000},
		{noise.Laplace(), 0}, // Clamping possible
		{noise.Laplace(), 1000},
	} {
		bm := getBoundedMeanFloat64(tc.n, arbitraryLower, arbitraryUpper)
		// Adding many entries prevents clamping.
		for i := 0; i < tc.numEntries; i++ {
			bm.Add(0.5)
		}
		bm.Result()

		smallAlphaConfInt, err := bm.ComputeConfidenceInterval(arbitraryAlpha * 0.5)
		if err != nil {
			t.Fatalf("With %v, couldn't compute confidence interval: %v", tc.n, err)
		}
		largeAlphaConfInt, err := bm.ComputeConfidenceInterval(arbitraryAlpha)
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

// Tests that BoundedMeanFloat64.ComputeConfidenceInterval() satisfies the confidence level for a given alpha.
func TestMeanComputeConfidenceInterval_SatisfiesConfidenceLevel(t *testing.T) {
	emptyInput := 0
	oneInput := 1
	manyInputs := 100
	// Choosing the midpoint between lower and upper to maximize the variance of the result. This
	// should increase the likelihood of detecting potential violations of the confidence level.
	// This also works for the case of empty (no) inputs because in that case, midpoint between
	// lower and upper is returned as the raw mean.
	rawValue := 1.5
	for _, tc := range []struct {
		n         noise.Noise
		alpha     float64
		numInputs int
		wantHits  int
	}{
		// Assuming that the true alpha of the confidence interval mechanism is 0.1, i.e., the raw value
		// is within the confidence interval with probability of at least 0.9, then the hits count will
		// be at least 2176 with probability greater than 1 - 10⁻⁶.
		{noise.Gaussian(), 0.1, emptyInput, 2176},
		{noise.Laplace(), 0.1, emptyInput, 2176},
		// Assuming that the true alpha of the confidence interval mechanism is 0.9, i.e., the raw value
		// is within the confidence interval with probability of at least 0.1, then the hits count will
		// be at least 182 with probability greater than 1 - 10⁻⁶.
		{noise.Gaussian(), 0.9, emptyInput, 182},
		{noise.Laplace(), 0.9, emptyInput, 182},
		{noise.Gaussian(), 0.1, oneInput, 2176},
		{noise.Laplace(), 0.1, oneInput, 2176},
		{noise.Gaussian(), 0.9, oneInput, 182},
		{noise.Laplace(), 0.9, oneInput, 182},
		{noise.Gaussian(), 0.1, manyInputs, 2176},
		{noise.Laplace(), 0.1, manyInputs, 2176},
		{noise.Gaussian(), 0.9, manyInputs, 182},
		{noise.Laplace(), 0.9, manyInputs, 182},
	} {
		hits := 0
		for i := 0; i < 2500; i++ {
			bm := getBoundedMeanFloat64(tc.n, 1.0, 2.0)
			for i := 0; i < tc.numInputs; i++ {
				bm.Add(rawValue)
			}
			bm.Result()

			confInt, err := bm.ComputeConfidenceInterval(tc.alpha)
			if err != nil {
				t.Fatalf("With noise=%v alpha=%f, numInputs=%d, couldn't compute confidence interval: %v", tc.n, tc.alpha, tc.numInputs, err)
			}

			if confInt.LowerBound <= float64(rawValue) && float64(rawValue) <= confInt.UpperBound {
				hits++
			}
		}
		if hits < tc.wantHits {
			t.Errorf("With noise=%v alpha=%f, numInputs=%d, fot %d hits, i.e. raw output within the confidence interval, wanted at least %d", tc.n, tc.alpha, tc.numInputs, hits, tc.wantHits)
		}
	}
}

// Tests that BoundedMeanFloat64.ComputeConfidenceInterval() returns errors correctly with different aggregation states.
func TestMeanComputeConfidenceInterval_StateChecks(t *testing.T) {
	for _, tc := range []struct {
		state   aggregationState
		wantErr bool
	}{
		{resultReturned, false},
		{defaultState, true},
		{merged, true},
		{serialized, true},
	} {
		bm := getNoiselessBMF()
		// Count and sum have to be also set to the same state
		// to allow ComputeConfidenceInterval calls.
		bm.state = tc.state
		bm.Count.state = tc.state
		bm.NormalizedSum.state = tc.state

		if _, err := bm.ComputeConfidenceInterval(0.1); (err != nil) != tc.wantErr {
			t.Errorf("ComputeConfidenceInterval: when state %v for err got %v, wantErr %t", tc.state, err, tc.wantErr)
		}
	}
}
