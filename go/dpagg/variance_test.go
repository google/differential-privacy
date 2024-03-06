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
	"math"
	"reflect"
	"testing"

	"github.com/google/differential-privacy/go/v3/noise"
	"github.com/google/differential-privacy/go/v3/rand"
	"github.com/google/go-cmp/cmp"
)

func TestNewBoundedVariance(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		opt     *BoundedVarianceOptions
		want    *BoundedVariance
		wantErr bool
	}{
		{"MaxPartitionsContributed is not set",
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noNoise{},
				MaxContributionsPerPartition: 2,
			},
			nil,
			true},
		{"MaxContributionsPerPartition is not set",
			&BoundedVarianceOptions{
				Epsilon:                  ln3,
				Delta:                    tenten,
				Lower:                    -1,
				Upper:                    5,
				Noise:                    noNoise{},
				MaxPartitionsContributed: 2,
			},
			nil,
			true},
		{"Noise is not set",
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        0,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     1,
			},
			&BoundedVariance{
				lower:    -1,
				upper:    5,
				state:    defaultState,
				midPoint: 2,
				Count: Count{
					epsilon:         ln3 / 3,
					delta:           0,
					l0Sensitivity:   1,
					lInfSensitivity: 2,
					noiseKind:       noise.LaplaceNoise,
					Noise:           noise.Laplace(),
					count:           0,
					state:           defaultState,
				},
				NormalizedSum: BoundedSumFloat64{
					epsilon:         ln3 / 3,
					delta:           0,
					l0Sensitivity:   1,
					lInfSensitivity: 6,
					lower:           -3,
					upper:           3,
					noiseKind:       noise.LaplaceNoise,
					Noise:           noise.Laplace(),
					sum:             0,
					state:           defaultState,
				},
				NormalizedSumOfSquares: BoundedSumFloat64{
					epsilon:         ln3 - ln3/3 - ln3/3,
					delta:           0,
					l0Sensitivity:   1,
					lInfSensitivity: 18,
					lower:           0,
					upper:           9,
					noiseKind:       noise.LaplaceNoise,
					Noise:           noise.Laplace(),
					sum:             0,
					state:           defaultState,
				},
			},
			false},
		{"Epsilon is not set",
			&BoundedVarianceOptions{
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Laplace(),
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     1,
			},
			nil,
			true},
		{"Negative Epsilon",
			&BoundedVarianceOptions{
				Epsilon:                      -1,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Laplace(),
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     1,
			},
			nil,
			true},
		{"Delta is not set with Gaussian noise",
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     1,
			},
			nil,
			true},
		{"Negative Delta",
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        -1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Laplace(),
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     1,
			},
			nil,
			true},
		{"Upper==Lower",
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        5,
				Upper:                        5,
				Noise:                        noise.Laplace(),
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     1,
			},
			nil,
			true},
		{"Upper<Lower",
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        6,
				Upper:                        5,
				Noise:                        noise.Laplace(),
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     1,
			},
			nil,
			true},
	} {
		bv, err := NewBoundedVariance(tc.opt)
		if (err != nil) != tc.wantErr {
			t.Errorf("With %s, got=%v error, wantErr: %t", tc.desc, err, tc.wantErr)
		}
		if !reflect.DeepEqual(bv, tc.want) {
			t.Errorf("NewBoundedVariance: when %s got %+v, want %+v", tc.desc, bv, tc.want)
		}
	}
}

func TestBVNoInput(t *testing.T) {
	lower, upper := -1.0, 5.0
	bv := getNoiselessBV(t, lower, upper)
	got, err := bv.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	// count = 0 => variance should be 0.0
	want := 0.0
	if !ApproxEqual(got, want) {
		t.Errorf("BoundedVariance: when there is no input data got=%f, want=%f", got, want)
	}
}

func TestBVAdd(t *testing.T) {
	lower, upper := -1.0, 5.0
	bv := getNoiselessBV(t, lower, upper)
	bv.Add(1.5)
	bv.Add(2.5)
	bv.Add(3.5)
	bv.Add(4.5)
	got, err := bv.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	want := 1.25
	if !ApproxEqual(got, want) {
		t.Errorf("Add: when dataset with elements inside boundaries got %f, want %f", got, want)
	}
}

func TestBVAddIgnoresNaN(t *testing.T) {
	lower, upper := -1.0, 5.0
	bv := getNoiselessBV(t, lower, upper)
	bv.Add(1)
	bv.Add(math.NaN())
	bv.Add(3)
	got, err := bv.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	want := 1.0
	if !ApproxEqual(got, want) {
		t.Errorf("Add: when dataset contains NaN got %f, want %f", got, want)
	}
}

func TestBVReturns0IfSingleEntryIsAdded(t *testing.T) {
	lower, upper := -1.0, 5.0
	bv := getNoiselessBV(t, lower, upper)

	bv.Add(1.2345)
	got, err := bv.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	want := 0.0 // single entry means 0 variance
	if !ApproxEqual(got, want) {
		t.Errorf("BoundedVariance: when dataset contains single entry got %f, want %f", got, want)
	}
}

func TestBVClamp(t *testing.T) {
	lower, upper := 2.0, 5.0
	bv := getNoiselessBV(t, lower, upper)

	bv.Add(1.0) // clamped to 2.0
	bv.Add(3.5)
	bv.Add(7.5) // clamped to 5.0
	got, err := bv.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	want := 1.5
	if !ApproxEqual(got, want) {
		t.Errorf("Add: when dataset with elements outside boundaries got %f, want %f", got, want)
	}
}

func TestBoundedVarianceResultSetsStateCorrectly(t *testing.T) {
	lower, upper := -1.0, 5.0
	bv := getNoiselessBV(t, lower, upper)
	_, err := bv.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}

	if bv.state != resultReturned {
		t.Errorf("BoundedVariance should have its state set to ResultReturned, got %v, want ResultReturned", bv.state)
	}
}

func TestBVNoiseIsCorrectlyCalled(t *testing.T) {
	bv := getMockBV(t)
	bv.Add(1)
	bv.Add(2)
	got, _ := bv.Result() // will fail if parameters are wrong. See mockBVNoise implementation for details.
	want := 0.25

	if !ApproxEqual(got, want) {
		t.Errorf("Add: when dataset = {1, 2} got %f, want %f", got, want)
	}
}

type mockBVNoise struct {
	t *testing.T
	noise.Noise
}

// AddNoiseInt64 checks that the parameters passed are the ones we expect.
func (mn mockBVNoise) AddNoiseInt64(x, l0, lInf int64, eps, del float64) (int64, error) {
	if x != 2 && x != 0 {
		// AddNoiseInt64 is initially called with a placeholder value of 0, so we don't want to fail when that happens
		mn.t.Errorf("AddNoiseInt64: for parameter x got %d, want %d", x, 2)
	}
	if l0 != 1 {
		mn.t.Errorf("AddNoiseInt64: for parameter l0Sensitivity got %d, want %d", l0, 1)
	}
	if lInf != 1 {
		mn.t.Errorf("AddNoiseInt64: for parameter lInfSensitivity got %d, want %d", lInf, 5)
	}
	if !ApproxEqual(eps, ln3/3) {
		mn.t.Errorf("AddNoiseInt64: for parameter epsilon got %f, want %f", eps, ln3*0.5)
	}
	if !ApproxEqual(del, tenten/3) {
		mn.t.Errorf("AddNoiseInt64: for parameter delta got %f, want %f", del, tenten*0.5)
	}
	return x, nil
}

// AddNoiseFloat64 checks that the parameters passed are the ones we expect.
func (mn mockBVNoise) AddNoiseFloat64(x float64, l0 int64, lInf, eps, del float64) (float64, error) {
	if !ApproxEqual(x, 1.0) && !ApproxEqual(x, -1.0) && !ApproxEqual(x, 0.0) {
		// AddNoiseFloat64 is initially called with a placeholder value of 0, so we don't want to fail when that happens
		// Then, for normalizedSum it is called with a value of -1.0 (1.0-2.0 + 2.0-2.0 = -1.0)
		// Finally, for normalizedSumOfSquares it is called with a value of 1.0 ((1.0-2.0)**2 + (2.0-2.0)**2 = 1.0)
		mn.t.Errorf("AddNoiseFloat64: for parameter x got %f, want one of {%f, %f, %f}", x, 0.0, -1.0, 1.0)
	}
	if l0 != 1 {
		mn.t.Errorf("AddNoiseFloat64: for parameter l0Sensitivity got %d, want %d", l0, 1)
	}
	if !ApproxEqual(lInf, 9.0) && !ApproxEqual(lInf, 3.0) && !ApproxEqual(lInf, 1.0) {
		// AddNoiseFloat64 is initially called with an lInf of 1, so we don't want to fail when that happens
		// Then, for normalizedSum it is called with an lInf of 3.0
		// Finally, for normalizedSumOfSquares it is called with an lInf of 9.0
		mn.t.Errorf("AddNoiseFloat64: for parameter lInfSensitivity got %f, want %f", lInf, 3.0)
	}
	if !ApproxEqual(eps, ln3/3) {
		mn.t.Errorf("AddNoiseFloat64: for parameter epsilon got %f, want %f", eps, ln3*0.5)
	}
	if !ApproxEqual(del, tenten/3) {
		mn.t.Errorf("AddNoiseFloat64: for parameter delta got %f, want %f", del, tenten*0.5)
	}
	return x, nil
}

func getMockBV(t *testing.T) *BoundedVariance {
	t.Helper()
	bv, err := NewBoundedVariance(&BoundedVarianceOptions{
		Epsilon:                      ln3,
		Delta:                        tenten,
		MaxPartitionsContributed:     1,
		MaxContributionsPerPartition: 1,
		Lower:                        -1,
		Upper:                        5,
		Noise:                        mockBVNoise{t: t},
	})
	if err != nil {
		t.Fatalf("Couldn't get mock BV: %v", err)
	}
	return bv
}

func getNoiselessBV(t *testing.T, lower, upper float64) *BoundedVariance {
	t.Helper()
	bv, err := NewBoundedVariance(&BoundedVarianceOptions{
		Epsilon:                      ln3,
		Delta:                        tenten,
		MaxPartitionsContributed:     1,
		MaxContributionsPerPartition: 1,
		Lower:                        lower,
		Upper:                        upper,
		Noise:                        noNoise{},
	})
	if err != nil {
		t.Fatalf("Couldn't get noiseless BV: %v", err)
	}
	return bv
}

func TestBVReturnsResultInsidePossibleBoundaries(t *testing.T) {
	lower := rand.Uniform() * 100
	upper := lower + rand.Uniform()*100

	bv, err := NewBoundedVariance(&BoundedVarianceOptions{
		Epsilon:                      ln3,
		MaxPartitionsContributed:     1,
		MaxContributionsPerPartition: 1,
		Lower:                        lower,
		Upper:                        upper,
		Noise:                        noise.Laplace(),
	})
	if err != nil {
		t.Fatalf("Couldn't initialize bv: %v", err)
	}

	for i := 0; i <= 1000; i++ {
		bv.Add(rand.Uniform() * 300 * rand.Sign())
	}

	res, err := bv.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	if res < 0 {
		t.Errorf("BoundedVariance: variance can't be smaller than 0, got variance %f, want to be >= %f", res, 0.0)
	}

	if res > computeMaxVariance(lower, upper) {
		t.Errorf("BoundedVariance: variance can't be larger than max variance, got %f, want to be <= %f", res, computeMaxVariance(lower, upper))
	}
}

func TestMergeBoundedVariance(t *testing.T) {
	lower, upper := -1.0, 5.0
	bv1 := getNoiselessBV(t, lower, upper)
	bv2 := getNoiselessBV(t, lower, upper)
	bv1.Add(1)
	bv1.Add(1)
	bv2.Add(3)
	bv2.Add(3)
	err := bv1.Merge(bv2)
	if err != nil {
		t.Fatalf("Couldn't merge bv1 and bv2: %v", err)
	}
	got, err := bv1.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	want := 1.0 // Would be 0.0 if merge didn't work.
	if !ApproxEqual(got, want) {
		t.Errorf("Merge: when merging 2 instances of BoundedVariance got %f, want %f", got, want)
	}
	if bv2.state != merged {
		t.Errorf("Merge: when merging 2 instances of BoundedVariance for bv2.state got %v, want Merged", bv2.state)
	}
}

func TestCheckMergeBoundedVarianceCompatibility(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		opt1    *BoundedVarianceOptions
		opt2    *BoundedVarianceOptions
		wantErr bool
	}{
		{"same options",
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			false},
		{"different epsilon",
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedVarianceOptions{
				Epsilon:                      2,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			true},
		{"different delta",
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenfive,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			true},
		{"different MaxPartitionsContributed",
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     2,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			true},
		{"different lower bound",
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        0,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			true},
		{"different upper bound",
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        6,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			true},
		{"different noise",
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        0,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Laplace(),
				MaxContributionsPerPartition: 2,
			},
			true},
		{"different maxContributionsPerPartition",
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedVarianceOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 5,
			},
			true},
	} {
		bv1, err := NewBoundedVariance(tc.opt1)
		if err != nil {
			t.Fatalf("Couldn't initialize bv1: %v", err)
		}
		bv2, err := NewBoundedVariance(tc.opt2)
		if err != nil {
			t.Fatalf("Couldn't initialize bv2: %v", err)
		}

		if err := checkMergeBoundedVariance(bv1, bv2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: when %s for err got %v, wantErr %t", tc.desc, err, tc.wantErr)
		}
	}
}

// Tests that checkMergeBoundedVariance() returns errors correctly with different BoundedVariance aggregation states.
func TestCheckMergeBoundedVarianceStateChecks(t *testing.T) {
	for _, tc := range []struct {
		state1  aggregationState
		state2  aggregationState
		wantErr bool
	}{
		{defaultState, defaultState, false},
		{resultReturned, defaultState, true},
		{defaultState, resultReturned, true},
		{serialized, defaultState, true},
		{defaultState, serialized, true},
		{defaultState, merged, true},
		{merged, defaultState, true},
	} {
		lower, upper := 0.0, 5.0
		bv1 := getNoiselessBV(t, lower, upper)
		bv2 := getNoiselessBV(t, lower, upper)

		bv1.state = tc.state1
		bv2.state = tc.state2

		if err := checkMergeBoundedVariance(bv1, bv2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: when states [%v, %v] for err got %v, wantErr %t", tc.state1, tc.state2, err, tc.wantErr)
		}
	}
}

func TestBVEquallyInitialized(t *testing.T) {
	for _, tc := range []struct {
		desc  string
		bv1   *BoundedVariance
		bv2   *BoundedVariance
		equal bool
	}{
		{
			"equal parameters",
			&BoundedVariance{
				lower:                  0,
				upper:                  2,
				midPoint:               1,
				NormalizedSumOfSquares: BoundedSumFloat64{},
				NormalizedSum:          BoundedSumFloat64{},
				Count:                  Count{},
				state:                  defaultState},
			&BoundedVariance{
				lower:                  0,
				upper:                  2,
				midPoint:               1,
				NormalizedSumOfSquares: BoundedSumFloat64{},
				NormalizedSum:          BoundedSumFloat64{},
				Count:                  Count{},
				state:                  defaultState},
			true,
		},
		{
			"different lower",
			&BoundedVariance{
				lower:                  -1,
				upper:                  2,
				midPoint:               0.5,
				NormalizedSumOfSquares: BoundedSumFloat64{},
				NormalizedSum:          BoundedSumFloat64{},
				Count:                  Count{},
				state:                  defaultState},
			&BoundedVariance{
				lower:                  0,
				upper:                  2,
				midPoint:               1,
				NormalizedSumOfSquares: BoundedSumFloat64{},
				NormalizedSum:          BoundedSumFloat64{},
				Count:                  Count{},
				state:                  defaultState},
			false,
		},
		{
			"different upper",
			&BoundedVariance{
				lower:                  0,
				upper:                  2,
				midPoint:               1,
				NormalizedSumOfSquares: BoundedSumFloat64{},
				NormalizedSum:          BoundedSumFloat64{},
				Count:                  Count{},
				state:                  defaultState},
			&BoundedVariance{
				lower:                  0,
				upper:                  3,
				midPoint:               1.5,
				NormalizedSumOfSquares: BoundedSumFloat64{},
				NormalizedSum:          BoundedSumFloat64{},
				Count:                  Count{},
				state:                  defaultState},
			false,
		},
		{
			"different count",
			&BoundedVariance{
				lower:                  0,
				upper:                  3,
				midPoint:               1.5,
				NormalizedSumOfSquares: BoundedSumFloat64{},
				NormalizedSum:          BoundedSumFloat64{},
				Count:                  Count{epsilon: ln3},
				state:                  defaultState},
			&BoundedVariance{
				lower:                  0,
				upper:                  3,
				midPoint:               1.5,
				NormalizedSumOfSquares: BoundedSumFloat64{},
				NormalizedSum:          BoundedSumFloat64{},
				Count:                  Count{epsilon: 1},
				state:                  defaultState},
			false,
		},
		{
			"different normalizedSum",
			&BoundedVariance{
				lower:                  0,
				upper:                  3,
				midPoint:               1.5,
				NormalizedSumOfSquares: BoundedSumFloat64{},
				NormalizedSum:          BoundedSumFloat64{epsilon: ln3},
				Count:                  Count{},
				state:                  defaultState},
			&BoundedVariance{
				lower:                  0,
				upper:                  3,
				midPoint:               1.5,
				NormalizedSumOfSquares: BoundedSumFloat64{},
				NormalizedSum:          BoundedSumFloat64{epsilon: 1},
				Count:                  Count{},
				state:                  defaultState},
			false,
		},
		{
			"different normalizedSumOfSquares",
			&BoundedVariance{
				lower:                  0,
				upper:                  3,
				midPoint:               1.5,
				NormalizedSumOfSquares: BoundedSumFloat64{epsilon: ln3},
				NormalizedSum:          BoundedSumFloat64{},
				Count:                  Count{},
				state:                  defaultState},
			&BoundedVariance{
				lower:                  0,
				upper:                  3,
				midPoint:               1.5,
				NormalizedSumOfSquares: BoundedSumFloat64{epsilon: 1},
				NormalizedSum:          BoundedSumFloat64{},
				Count:                  Count{},
				state:                  defaultState},
			false,
		},
		{
			"different state",
			&BoundedVariance{
				lower:                  0,
				upper:                  2,
				midPoint:               1,
				NormalizedSumOfSquares: BoundedSumFloat64{},
				NormalizedSum:          BoundedSumFloat64{},
				Count:                  Count{},
				state:                  defaultState},
			&BoundedVariance{
				lower:                  0,
				upper:                  2,
				midPoint:               1,
				NormalizedSumOfSquares: BoundedSumFloat64{},
				NormalizedSum:          BoundedSumFloat64{},
				Count:                  Count{},
				state:                  merged},
			false,
		},
	} {
		if bvEquallyInitialized(tc.bv1, tc.bv2) != tc.equal {
			t.Errorf("bvEquallyInitialized: when %s got %t, want %t", tc.desc, !tc.equal, tc.equal)
		}
	}
}

func compareBoundedVariance(bv1, bv2 *BoundedVariance) bool {
	return bv1.lower == bv2.lower &&
		bv1.upper == bv2.upper &&
		compareCount(&bv1.Count, &bv2.Count) &&
		compareBoundedSumFloat64(&bv1.NormalizedSum, &bv2.NormalizedSum) &&
		compareBoundedSumFloat64(&bv1.NormalizedSumOfSquares, &bv2.NormalizedSumOfSquares) &&
		bv1.midPoint == bv2.midPoint &&
		bv1.state == bv2.state
}

// Tests that serialization for BoundedVariance works as expected.
func TestBVSerialization(t *testing.T) {
	for _, tc := range []struct {
		desc string
		opts *BoundedVarianceOptions
	}{
		{"default options", &BoundedVarianceOptions{
			Epsilon:                      ln3,
			Lower:                        0,
			Upper:                        1,
			Delta:                        0,
			MaxContributionsPerPartition: 1,
			MaxPartitionsContributed:     1,
		}},
		{"non-default options", &BoundedVarianceOptions{
			Lower:                        -100,
			Upper:                        555,
			Epsilon:                      ln3,
			Delta:                        1e-5,
			MaxPartitionsContributed:     5,
			MaxContributionsPerPartition: 6,
			Noise:                        noise.Gaussian(),
		}},
	} {
		bv, err := NewBoundedVariance(tc.opts)
		if err != nil {
			t.Fatalf("Couldn't initialize bv: %v", err)
		}
		bvUnchanged, err := NewBoundedVariance(tc.opts)
		if err != nil {
			t.Fatalf("Couldn't initialize bvUnchanged: %v", err)
		}
		bytes, err := encode(bv)
		if err != nil {
			t.Fatalf("encode(BoundedVariance) error: %v", err)
		}
		bvUnmarshalled := new(BoundedVariance)
		if err := decode(bvUnmarshalled, bytes); err != nil {
			t.Fatalf("decode(BoundedVariance) error: %v", err)
		}
		// Check that encoding -> decoding is the identity function.
		if !cmp.Equal(bvUnchanged, bvUnmarshalled, cmp.Comparer(compareBoundedVariance)) {
			t.Errorf("decode(encode(_)): when %s got %+v, want %+v", tc.desc, bvUnmarshalled, bvUnchanged)
		}
		if bv.state != serialized {
			t.Errorf("BoundedVariance should have its state set to Serialized, got %v , want Serialized", bv.state)
		}
	}
}

// Tests that GobEncode() returns errors correctly with different BoundedVariance aggregation states.
func TestBoundedVarianceSerializationStateChecks(t *testing.T) {
	for _, tc := range []struct {
		state   aggregationState
		wantErr bool
	}{
		{defaultState, false},
		{merged, true},
		{serialized, false},
		{resultReturned, true},
	} {
		lower, upper := 0.0, 5.0
		bv := getNoiselessBV(t, lower, upper)
		bv.state = tc.state

		if _, err := bv.GobEncode(); (err != nil) != tc.wantErr {
			t.Errorf("GobEncode: when state %v for err got %v, wantErr %t", tc.state, err, tc.wantErr)
		}
	}
}
