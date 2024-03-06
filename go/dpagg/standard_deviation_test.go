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

func TestNewBoundedStandardDeviation(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		opt     *BoundedStandardDeviationOptions
		want    *BoundedStandardDeviation
		wantErr bool
	}{
		{"MaxPartitionsContributed is not set",
			&BoundedStandardDeviationOptions{
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
			&BoundedStandardDeviationOptions{
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
			&BoundedStandardDeviationOptions{
				Epsilon:                      ln3,
				Delta:                        0,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     1,
			},
			&BoundedStandardDeviation{
				Variance: BoundedVariance{
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
				}},
			false},
		{"Epsilon is not set",
			&BoundedStandardDeviationOptions{
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
			&BoundedStandardDeviationOptions{
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
			&BoundedStandardDeviationOptions{
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
			&BoundedStandardDeviationOptions{
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
			&BoundedStandardDeviationOptions{
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
			&BoundedStandardDeviationOptions{
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
		bstdv, err := NewBoundedStandardDeviation(tc.opt)
		if (err != nil) != tc.wantErr {
			t.Errorf("With %s, got=%v error, wantErr: %t", tc.desc, err, tc.wantErr)
		}
		if !reflect.DeepEqual(bstdv, tc.want) {
			t.Errorf("NewBoundedStandardDeviation: when %s got %+v, want %+v", tc.desc, bstdv, tc.want)
		}
	}
}

func TestBSTDVNoInput(t *testing.T) {
	lower, upper := -1.0, 5.0
	bstdv := getNoiselessBSTDV(t, lower, upper)
	got, err := bstdv.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	// count = 0 => standard deviation should be 0.0
	want := 0.0
	if !ApproxEqual(got, want) {
		t.Errorf("BoundedStandardDeviation: when there is no input data got=%f, want=%f", got, want)
	}
}

func TestBSTDVAdd(t *testing.T) {
	lower, upper := -1.0, 5.0
	bstdv := getNoiselessBSTDV(t, lower, upper)
	bstdv.Add(1)
	bstdv.Add(1)
	bstdv.Add(2)
	bstdv.Add(2)
	got, err := bstdv.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	want := 0.50
	if !ApproxEqual(got, want) {
		t.Errorf("Add: when dataset with elements inside boundaries got %f, want %f", got, want)
	}
}

func TestBSTDVAddIgnoresNaN(t *testing.T) {
	lower, upper := -1.0, 5.0
	bstdv := getNoiselessBSTDV(t, lower, upper)
	bstdv.Add(1)
	bstdv.Add(math.NaN())
	bstdv.Add(3)
	got, err := bstdv.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	want := 1.0
	if !ApproxEqual(got, want) {
		t.Errorf("Add: when dataset contains NaN got %f, want %f", got, want)
	}
}

func TestBSTDVReturns0IfSingleEntryIsAdded(t *testing.T) {
	lower, upper := -1.0, 5.0
	bstdv := getNoiselessBSTDV(t, lower, upper)

	bstdv.Add(1.2345)
	got, err := bstdv.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	want := 0.0 // single entry means 0 standard deviation
	if !ApproxEqual(got, want) {
		t.Errorf("BoundedStandardDeviation: when dataset contains single entry got %f, want %f", got, want)
	}
}

func TestBSTDVClamp(t *testing.T) {
	lower, upper := 2.0, 4.0
	bstdv := getNoiselessBSTDV(t, lower, upper)

	bstdv.Add(0.0) // clamped to 2.0
	bstdv.Add(1.0) // clamped to 2.0
	bstdv.Add(4.0)
	bstdv.Add(7.0) // clamped to 4.0
	got, err := bstdv.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	want := 1.0
	if !ApproxEqual(got, want) {
		t.Errorf("Add: when dataset with elements outside boundaries got %f, want %f", got, want)
	}
}

func TestBoundedStandardDeviationResultSetsStateCorrectly(t *testing.T) {
	lower, upper := -1.0, 5.0
	bstdv := getNoiselessBSTDV(t, lower, upper)
	_, err := bstdv.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}

	if bstdv.state != resultReturned {
		t.Errorf("BoundedStandardDeviation should have its state set to ResultReturned, got %v, want ResultReturned", bstdv.state)
	}
}

func TestBVBSTDVNoiseIsCorrectlyCalled(t *testing.T) {
	bstdv := getMockBSTDV(t)
	bstdv.Add(1)
	bstdv.Add(2)
	got, err := bstdv.Result() // will fail if parameters are wrong. See mockBVNoise implementation for details.
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	want := 0.50

	if !ApproxEqual(got, want) {
		t.Errorf("Add: when dataset = {1, 2} got %f, want %f", got, want)
	}
}

func getMockBSTDV(t *testing.T) *BoundedStandardDeviation {
	t.Helper()
	bstdv, err := NewBoundedStandardDeviation(&BoundedStandardDeviationOptions{
		Epsilon:                      ln3,
		Delta:                        tenten,
		MaxPartitionsContributed:     1,
		MaxContributionsPerPartition: 1,
		Lower:                        -1,
		Upper:                        5,
		Noise:                        mockBVNoise{t: t},
	})
	if err != nil {
		t.Fatalf("Couldn't get mock BSTDV: %v", err)
	}
	return bstdv
}

func getNoiselessBSTDV(t *testing.T, lower, upper float64) *BoundedStandardDeviation {
	t.Helper()
	bstdv, err := NewBoundedStandardDeviation(&BoundedStandardDeviationOptions{
		Epsilon:                      ln3,
		Delta:                        tenten,
		MaxPartitionsContributed:     1,
		MaxContributionsPerPartition: 1,
		Lower:                        lower,
		Upper:                        upper,
		Noise:                        noNoise{},
	})
	if err != nil {
		t.Fatalf("Couldn't get noiseless BSTDV: %v", err)
	}
	return bstdv
}

func TestBSTDVReturnsResultInsidePossibleBoundaries(t *testing.T) {
	lower := rand.Uniform() * 100
	upper := lower + rand.Uniform()*100

	bstdv, err := NewBoundedStandardDeviation(&BoundedStandardDeviationOptions{
		Epsilon:                      ln3,
		MaxPartitionsContributed:     1,
		MaxContributionsPerPartition: 1,
		Lower:                        lower,
		Upper:                        upper,
		Noise:                        noise.Laplace(),
	})
	if err != nil {
		t.Fatalf("Couldn't initialize bstdv: %v", err)
	}

	for i := 0; i <= 1000; i++ {
		bstdv.Add(rand.Uniform() * 300 * rand.Sign())
	}

	res, err := bstdv.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	if res < 0 {
		t.Errorf("BoundedStandardDeviation: stdv can't be smaller than 0, got stdv %f, want to be >= %f", res, 0.0)
	}

	if res > math.Sqrt(computeMaxVariance(lower, upper)) {
		t.Errorf("BoundedStandardDeviation: stdv can't be larger than max stdv, got %f, want to be <= %f", res, math.Sqrt(computeMaxVariance(lower, upper)))
	}
}

func TestMergeBoundedStandardDeviation(t *testing.T) {
	lower, upper := -1.0, 5.0
	bstdv1 := getNoiselessBSTDV(t, lower, upper)
	bstdv2 := getNoiselessBSTDV(t, lower, upper)
	bstdv1.Add(1)
	bstdv1.Add(1)
	bstdv2.Add(3)
	bstdv2.Add(3)
	err := bstdv1.Merge(bstdv2)
	if err != nil {
		t.Fatalf("Couldn't merge bstdv1 and bstdv2: %v", err)
	}
	got, err := bstdv1.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	want := 1.0 // Would be 0.0 if merge didn't work.
	if !ApproxEqual(got, want) {
		t.Errorf("Merge: when merging 2 instances of BoundedStandardDeviation got %f, want %f", got, want)
	}
	if bstdv2.state != merged {
		t.Errorf("Merge: when merging 2 instances of BoundedStandardDeviation for bv2.state got %v, want Merged", bstdv2.state)
	}
}

func TestCheckMergeBoundedStandardDeviationCompatibility(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		opt1    *BoundedStandardDeviationOptions
		opt2    *BoundedStandardDeviationOptions
		wantErr bool
	}{
		{"same options",
			&BoundedStandardDeviationOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedStandardDeviationOptions{
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
			&BoundedStandardDeviationOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedStandardDeviationOptions{
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
			&BoundedStandardDeviationOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedStandardDeviationOptions{
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
			&BoundedStandardDeviationOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedStandardDeviationOptions{
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
			&BoundedStandardDeviationOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedStandardDeviationOptions{
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
			&BoundedStandardDeviationOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedStandardDeviationOptions{
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
			&BoundedStandardDeviationOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedStandardDeviationOptions{
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
			&BoundedStandardDeviationOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedStandardDeviationOptions{
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
		bstdv1, err := NewBoundedStandardDeviation(tc.opt1)
		if err != nil {
			t.Fatalf("Couldn't initialize bstdv1: %v", err)
		}
		bstdv2, err := NewBoundedStandardDeviation(tc.opt2)
		if err != nil {
			t.Fatalf("Couldn't initialize bstdv2: %v", err)
		}

		if err := checkMergeBoundedStandardDeviation(bstdv1, bstdv2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: when %s for err got %v, wantErr %t", tc.desc, err, tc.wantErr)
		}
	}
}

// Tests that checkMergeBoundedStandardDeviation() returns errors correctly with different BoundedStandardDeviation aggregation states.
func TestCheckMergeBoundedStandardDeviationStateChecks(t *testing.T) {
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
		bstdv1 := getNoiselessBSTDV(t, lower, upper)
		bstdv2 := getNoiselessBSTDV(t, lower, upper)

		bstdv1.state = tc.state1
		bstdv2.state = tc.state2

		if err := checkMergeBoundedStandardDeviation(bstdv1, bstdv2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: when states [%v, %v] for err got %v, wantErr %t", tc.state1, tc.state2, err, tc.wantErr)
		}
	}
}

func TestBSTDVEquallyInitialized(t *testing.T) {
	for _, tc := range []struct {
		desc  string
		bv1   *BoundedStandardDeviation
		bv2   *BoundedStandardDeviation
		equal bool
	}{
		{
			"equal parameters",
			&BoundedStandardDeviation{
				Variance: BoundedVariance{},
				state:    defaultState},
			&BoundedStandardDeviation{
				Variance: BoundedVariance{},
				state:    defaultState},
			true,
		},
		{
			"different variance",
			&BoundedStandardDeviation{
				Variance: BoundedVariance{lower: -1.0},
				state:    defaultState},
			&BoundedStandardDeviation{
				Variance: BoundedVariance{lower: 0.0},
				state:    defaultState},
			false,
		},
		{
			"different state",
			&BoundedStandardDeviation{
				Variance: BoundedVariance{},
				state:    defaultState},
			&BoundedStandardDeviation{
				Variance: BoundedVariance{},
				state:    merged},
			false,
		},
	} {
		if bstdvEquallyInitialized(tc.bv1, tc.bv2) != tc.equal {
			t.Errorf("bstdvEquallyInitialized: when %s got %t, want %t", tc.desc, !tc.equal, tc.equal)
		}
	}
}

func compareBoundedStandardDeviation(bstdv1, bstdv2 *BoundedStandardDeviation) bool {
	return compareBoundedVariance(&bstdv1.Variance, &bstdv2.Variance) &&
		bstdv1.state == bstdv2.state
}

// Tests that serialization for BoundedStandardDeviation works as expected.
func TestBSTDVSerialization(t *testing.T) {
	for _, tc := range []struct {
		desc string
		opts *BoundedStandardDeviationOptions
	}{
		{"default options", &BoundedStandardDeviationOptions{
			Epsilon:                      ln3,
			Lower:                        0,
			Upper:                        1,
			Delta:                        0,
			MaxContributionsPerPartition: 1,
			MaxPartitionsContributed:     1,
		}},
		{"non-default options", &BoundedStandardDeviationOptions{
			Lower:                        -100,
			Upper:                        555,
			Epsilon:                      ln3,
			Delta:                        1e-5,
			MaxPartitionsContributed:     5,
			MaxContributionsPerPartition: 6,
			Noise:                        noise.Gaussian(),
		}},
	} {
		bstdv, err := NewBoundedStandardDeviation(tc.opts)
		if err != nil {
			t.Fatalf("Couldn't initialize bstdv: %v", err)
		}
		bstdvUnchanged, err := NewBoundedStandardDeviation(tc.opts)
		if err != nil {
			t.Fatalf("Couldn't initialize bstdvUnchanged: %v", err)
		}
		bytes, err := encode(bstdv)
		if err != nil {
			t.Fatalf("encode(BoundedStandardDeviation) error: %v", err)
		}
		bstdvUnmarshalled := new(BoundedStandardDeviation)
		if err := decode(bstdvUnmarshalled, bytes); err != nil {
			t.Fatalf("decode(BoundedStandardDeviation) error: %v", err)
		}
		// Check that encoding -> decoding is the identity function.
		if !cmp.Equal(bstdvUnchanged, bstdvUnmarshalled, cmp.Comparer(compareBoundedStandardDeviation)) {
			t.Errorf("decode(encode(_)): when %s got %+v, want %+v", tc.desc, bstdvUnmarshalled, bstdvUnchanged)
		}
		if bstdv.state != serialized {
			t.Errorf("BoundedStandardDeviation should have its state set to Serialized, got %v , want Serialized", bstdv.state)
		}
	}
}

// Tests that GobEncode() returns errors correctly with different BoundedStandardDeviation aggregation states.
func TestBoundedStandardDeviationSerializationStateChecks(t *testing.T) {
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
		bv := getNoiselessBSTDV(t, lower, upper)
		bv.state = tc.state

		if _, err := bv.GobEncode(); (err != nil) != tc.wantErr {
			t.Errorf("GobEncode: when state %v for err got %v, wantErr %t", tc.state, err, tc.wantErr)
		}
	}
}
