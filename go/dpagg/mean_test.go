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

package dpagg

import (
	"math"
	"reflect"
	"testing"

	"github.com/google/differential-privacy/go/noise"
	"github.com/google/differential-privacy/go/rand"
	"github.com/google/go-cmp/cmp"
)

func TestNewBoundedMeanFloat64(t *testing.T) {
	for _, tc := range []struct {
		desc string
		opt  *BoundedMeanFloat64Options
		want *BoundedMeanFloat64
	}{
		{"MaxPartitionsContributed is not set",
			&BoundedMeanFloat64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noNoise{},
				MaxContributionsPerPartition: 2,
			},
			&BoundedMeanFloat64{
				lower:    -1,
				upper:    5,
				state:    Default,
				midPoint: 2,
				count: Count{
					epsilon:         ln3 * 0.5,
					delta:           tenten * 0.5,
					l0Sensitivity:   1,
					lInfSensitivity: 2,
					noise:           noNoise{},
					count:           0,
					state:           Default,
				},
				normalizedSum: BoundedSumFloat64{
					epsilon:         ln3 * 0.5,
					delta:           tenten * 0.5,
					l0Sensitivity:   1,
					lInfSensitivity: 6,
					lower:           -3,
					upper:           3,
					noise:           noNoise{},
					sum:             0,
					state:           Default,
				},
			}},
		{"Noise is not set",
			&BoundedMeanFloat64Options{
				Epsilon:                      ln3,
				Delta:                        0,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     1,
			},
			&BoundedMeanFloat64{
				lower:    -1,
				upper:    5,
				state:    Default,
				midPoint: 2,
				count: Count{
					epsilon:         ln3 * 0.5,
					delta:           0,
					l0Sensitivity:   1,
					lInfSensitivity: 2,
					noiseKind:       noise.LaplaceNoise,
					noise:           noise.Laplace(),
					count:           0,
					state:           Default,
				},
				normalizedSum: BoundedSumFloat64{
					epsilon:         ln3 * 0.5,
					delta:           0,
					l0Sensitivity:   1,
					lInfSensitivity: 6,
					lower:           -3,
					upper:           3,
					noiseKind:       noise.LaplaceNoise,
					noise:           noise.Laplace(),
					sum:             0,
					state:           Default,
				},
			}},
	} {
		got := NewBoundedMeanFloat64(tc.opt)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("NewBoundedMeanFloat64: when %s got %v, want %v", tc.desc, got, tc.want)
		}
	}
}

func TestBMNoInputFloat64(t *testing.T) {
	bmf := getNoiselessBMF()
	got := bmf.Result()
	// count = 0 => returns midPoint = 2
	want := 2.0
	if !ApproxEqual(got, want) {
		t.Errorf("BoundedMean: when there is no input data got=%f, want=%f", got, want)
	}
}

func TestBMAddFloat64(t *testing.T) {
	bmf := getNoiselessBMF()
	bmf.Add(1.5)
	bmf.Add(2.5)
	bmf.Add(3.5)
	bmf.Add(4.5)
	got := bmf.Result()
	want := 3.0
	if !ApproxEqual(got, want) {
		t.Errorf("Add: when dataset with elements inside boundaries got %f, want %f", got, want)
	}
}

func TestBMAddFloat64IgnoresNaN(t *testing.T) {
	bmf := getNoiselessBMF()
	bmf.Add(1)
	bmf.Add(math.NaN())
	bmf.Add(3)
	got := bmf.Result()
	want := 2.0
	if !ApproxEqual(got, want) {
		t.Errorf("Add: when dataset contains NaN got %f, want %f", got, want)
	}
}

func TestBMReturnsMidPointForEmptyInputFloat64(t *testing.T) {
	bmf := getNoiselessBMF()
	// lower = -1, upper = 5
	// normalized sum inside BM has bounds: lower = -3, upper = 3
	// midPoint = 2
	got := bmf.Result()
	want := 2.0 // midPoint
	if !ApproxEqual(got, want) {
		t.Errorf("BoundedMean: when dataset is empty got %f, want %f", got, want)
	}
}

func TestBMReturnsEntryIfSingleEntryIsAddedFloat64(t *testing.T) {
	bmf := getNoiselessBMF()
	// lower = -1, upper = 5
	// normalized sum inside BM has bounds: lower = -3, upper = 3
	// midPoint = 2

	bmf.Add(1.2345)
	got := bmf.Result()
	want := 1.2345 // single entry
	if !ApproxEqual(got, want) {
		t.Errorf("BoundedMean: when dataset contains single entry got %f, want %f", got, want)
	}
}

func TestBMClampFloat64(t *testing.T) {
	bmf := getNoiselessBMF()
	// lower = -1, upper = 5
	// normalized sum inside BM has bounds: lower = -3, upper = 3
	// midPoint = 2

	bmf.Add(3.5)  // clamp(3.5) - midPoint = 3.5 - 2 = 1.5 -> to normalizedSum
	bmf.Add(8.3)  // clamp(8.3) - midPoint = 5 - 2 = 3 -> to normalizedSum
	bmf.Add(-7.5) // clamp(-7.5) - midPoint = -1 - 2 = -3 -> to normalizedSum
	got := bmf.Result()
	want := 2.5
	if !ApproxEqual(got, want) { // clamp (normalizedSum/count + mid point) = 1.5 / 3 + 2 = 2.5
		t.Errorf("Add: when dataset with elements outside boundaries got %f, want %f", got, want)
	}
}

func TestBoundedMeanFloat64ResultSetsStateCorrectly(t *testing.T) {
	bm := getNoiselessBMF()
	bm.Result()

	if bm.state != ResultReturned {
		t.Errorf("BoundedMeanFloat64 should have its state set to ResultReturned. got %v, want ResultReturned", bm.state)
	}
}

func TestBMNoiseIsCorrectlyCalledFloat64(t *testing.T) {
	bmf := getMockBMF(t)
	bmf.Add(1)
	bmf.Add(2)
	got := bmf.Result() // will fail if parameters are wrong
	want := 5.0
	// clamp((noised normalizedSum / noised count) + mid point) = clamp((-1 + 100) / (2 + 10) + 2) = 5
	if !ApproxEqual(got, want) {
		t.Errorf("Add: when dataset = {1, 2} got %f, want %f", got, want)
	}
}

func TestBMReturnsResultInsideProvidedBoundariesFloat64(t *testing.T) {
	lower := rand.Uniform() * 100
	upper := lower + rand.Uniform()*100

	bmf := NewBoundedMeanFloat64(&BoundedMeanFloat64Options{
		Epsilon:                      ln3,
		MaxPartitionsContributed:     1,
		MaxContributionsPerPartition: 1,
		Lower:                        lower,
		Upper:                        upper,
		Noise:                        noise.Laplace(),
	})

	for i := 0; i <= 1000; i++ {
		bmf.Add(rand.Uniform() * 300 * rand.Sign())
	}

	res := bmf.Result()
	if res < lower {
		t.Errorf("BoundedMean: result is outside of boundaries, got %f, want to be >= %f", res, lower)
	}

	if res > upper {
		t.Errorf("BoundedMean: result is outside of boundaries, got %f, want to be <= %f", res, upper)
	}
}

type mockBMNoise struct {
	t *testing.T
	noise.Noise
}

// AddNoiseInt64 checks that the parameters passed are the ones we expect.
func (mn mockBMNoise) AddNoiseInt64(x, l0, lInf int64, eps, del float64) int64 {
	if x != 2 && x != 0 {
		// AddNoiseInt64 is initially called with a dummy value of 0, so we don't want to fail when that happens
		mn.t.Errorf("AddNoiseInt64: for parameter x got %d, want %d", x, 2)
	}
	if l0 != 1 {
		mn.t.Errorf("AddNoiseInt64: for parameter l0Sensitivity got %d, want %d", l0, 1)
	}
	if lInf != 1 {
		mn.t.Errorf("AddNoiseInt64: for parameter lInfSensitivity got %d, want %d", lInf, 5)
	}
	if !ApproxEqual(eps, ln3*0.5) {
		mn.t.Errorf("AddNoiseInt64: for parameter epsilon got %f, want %f", eps, ln3*0.5)
	}
	if !ApproxEqual(del, tenten*0.5) {
		mn.t.Errorf("AddNoiseInt64: for parameter delta got %f, want %f", del, tenten*0.5)
	}
	return x + 10
}

// AddNoiseFloat64 checks that the parameters passed are the ones we expect.
func (mn mockBMNoise) AddNoiseFloat64(x float64, l0 int64, lInf, eps, del float64) float64 {
	if !ApproxEqual(x, -1.0) && !ApproxEqual(x, 0.0) {
		// AddNoiseFloat64 is initially called with a dummy value of 0, so we don't want to fail when that happens
		mn.t.Errorf("AddNoiseFloat64: for parameter x got %f, want %f", x, 0.0)
	}
	if l0 != 1 {
		mn.t.Errorf("AddNoiseFloat64: for parameter l0Sensitivity got %d, want %d", l0, 1)
	}
	if !ApproxEqual(lInf, 3.0) && !ApproxEqual(lInf, 1.0) {
		// AddNoiseFloat64 is initially called with a dummy value of 1, so we don't want to fail when that happens
		mn.t.Errorf("AddNoiseFloat64: for parameter lInfSensitivity got %f, want %f", lInf, 3.0)
	}
	if !ApproxEqual(eps, ln3*0.5) {
		mn.t.Errorf("AddNoiseFloat64: for parameter epsilon got %f, want %f", eps, ln3*0.5)
	}
	if !ApproxEqual(del, tenten*0.5) {
		mn.t.Errorf("AddNoiseFloat64: for parameter delta got %f, want %f", del, tenten*0.5)
	}
	return x + 100
}

func getNoiselessBMF() *BoundedMeanFloat64 {
	return NewBoundedMeanFloat64(&BoundedMeanFloat64Options{
		Epsilon:                      ln3,
		Delta:                        tenten,
		MaxPartitionsContributed:     1,
		MaxContributionsPerPartition: 1,
		Lower:                        -1,
		Upper:                        5,
		Noise:                        noNoise{},
	})
}

func getMockBMF(t *testing.T) *BoundedMeanFloat64 {
	return NewBoundedMeanFloat64(&BoundedMeanFloat64Options{
		Epsilon:                      ln3,
		Delta:                        tenten,
		MaxPartitionsContributed:     1,
		MaxContributionsPerPartition: 1,
		Lower:                        -1,
		Upper:                        5,
		Noise:                        mockBMNoise{t: t},
	})
}

func TestMergeBoundedMeanFloat64(t *testing.T) {
	bm1 := getNoiselessBMF()
	bm2 := getNoiselessBMF()
	bm1.Add(1)
	bm1.Add(2)
	bm1.Add(3.5)
	bm1.Add(4)
	bm2.Add(4.5)
	bm1.Merge(bm2)
	got := bm1.Result()
	want := 3.0
	if !ApproxEqual(got, want) {
		t.Errorf("Merge: when merging 2 instances of BoundedMean got %f, want %f", got, want)
	}
	if bm2.state != Merged {
		t.Errorf("Merge: when merging 2 instances of BoundedMean for bm2.state got %v, want Merged", bm2.state)
	}
}

func TestCheckMergeBoundedMeanFloat64Compatibility(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		opt1    *BoundedMeanFloat64Options
		opt2    *BoundedMeanFloat64Options
		wantErr bool
	}{
		{"same options",
			&BoundedMeanFloat64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedMeanFloat64Options{
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
			&BoundedMeanFloat64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedMeanFloat64Options{
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
			&BoundedMeanFloat64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedMeanFloat64Options{
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
			&BoundedMeanFloat64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedMeanFloat64Options{
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
			&BoundedMeanFloat64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedMeanFloat64Options{
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
			&BoundedMeanFloat64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedMeanFloat64Options{
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
			&BoundedMeanFloat64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedMeanFloat64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Laplace(),
				MaxContributionsPerPartition: 2,
			},
			true},
		{"different maxContributionsPerPartition",
			&BoundedMeanFloat64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				MaxContributionsPerPartition: 2,
			},
			&BoundedMeanFloat64Options{
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
		bm1 := NewBoundedMeanFloat64(tc.opt1)
		bm2 := NewBoundedMeanFloat64(tc.opt2)

		if err := checkMergeBoundedMeanFloat64(bm1, bm2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: when %v for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

// Tests that checkMergeBoundedMeanFloat64() returns errors correctly with different BoundedMeanFloat64 aggregation states.
func TestCheckMergeBoundedMeanFloat64StateChecks(t *testing.T) {
	for _, tc := range []struct {
		state1  aggregationState
		state2  aggregationState
		wantErr bool
	}{
		{Default, Default, false},
		{ResultReturned, Default, true},
		{Default, ResultReturned, true},
		{Serialized, Default, true},
		{Default, Serialized, true},
		{Default, Merged, true},
		{Merged, Default, true},
	} {
		bm1 := getNoiselessBMF()
		bm2 := getNoiselessBMF()

		bm1.state = tc.state1
		bm2.state = tc.state2

		if err := checkMergeBoundedMeanFloat64(bm1, bm2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: for states[%v , %v] err got %v, want %t", tc.state1, tc.state2, err, tc.wantErr)
		}
	}
}

func TestBMEquallyInitializedFloat64(t *testing.T) {
	for _, tc := range []struct {
		desc  string
		bm1   *BoundedMeanFloat64
		bm2   *BoundedMeanFloat64
		equal bool
	}{
		{
			"equal parameters",
			&BoundedMeanFloat64{
				lower:         0,
				upper:         2,
				midPoint:      1,
				normalizedSum: BoundedSumFloat64{},
				count:         Count{},
				state:         Default},
			&BoundedMeanFloat64{
				lower:         0,
				upper:         2,
				midPoint:      1,
				normalizedSum: BoundedSumFloat64{},
				count:         Count{},
				state:         Default},
			true,
		},
		{
			"different lower",
			&BoundedMeanFloat64{
				lower:         -1,
				upper:         2,
				midPoint:      1,
				normalizedSum: BoundedSumFloat64{},
				count:         Count{},
				state:         Default},
			&BoundedMeanFloat64{
				lower:         0,
				upper:         2,
				midPoint:      1,
				normalizedSum: BoundedSumFloat64{},
				count:         Count{},
				state:         Default},
			false,
		},
		{
			"different upper",
			&BoundedMeanFloat64{
				lower:         0,
				upper:         2,
				midPoint:      1,
				normalizedSum: BoundedSumFloat64{},
				count:         Count{},
				state:         Default},
			&BoundedMeanFloat64{
				lower:         0,
				upper:         3,
				midPoint:      1,
				normalizedSum: BoundedSumFloat64{},
				count:         Count{},
				state:         Default},
			false,
		},
		{
			"different count",
			&BoundedMeanFloat64{
				lower:         0,
				upper:         2,
				midPoint:      1,
				normalizedSum: BoundedSumFloat64{},
				count:         Count{epsilon: ln3},
				state:         Default},
			&BoundedMeanFloat64{
				lower:         0,
				upper:         2,
				midPoint:      1,
				normalizedSum: BoundedSumFloat64{},
				count:         Count{epsilon: 1},
				state:         Default},
			false,
		},
		{
			"different normalizedSum",
			&BoundedMeanFloat64{
				lower:         0,
				upper:         2,
				midPoint:      1,
				normalizedSum: BoundedSumFloat64{epsilon: ln3},
				count:         Count{},
				state:         Default},
			&BoundedMeanFloat64{
				lower:         0,
				upper:         2,
				midPoint:      1,
				normalizedSum: BoundedSumFloat64{epsilon: 1},
				count:         Count{},
				state:         Default},
			false,
		},
	} {
		if bmEquallyInitializedFloat64(tc.bm1, tc.bm2) != tc.equal {
			t.Errorf("bmEquallyInitializedFloat64: when %v got %t, want %t", tc.desc, !tc.equal, tc.equal)
		}
	}
}

func compareBoundedMeanFloat64(bm1, bm2 *BoundedMeanFloat64) bool {
	return bm1.lower == bm2.lower &&
		bm1.upper == bm2.upper &&
		compareCount(&bm1.count, &bm2.count) &&
		compareBoundedSumFloat64(&bm1.normalizedSum, &bm2.normalizedSum) &&
		bm1.midPoint == bm2.midPoint &&
		bm1.state == bm2.state
}

// Tests that serialization for BoundedMeanFloat64 works as expected.
func TestBMFloat64Serialization(t *testing.T) {
	for _, tc := range []struct {
		desc string
		opts *BoundedMeanFloat64Options
	}{
		{"default options", &BoundedMeanFloat64Options{
			Epsilon:                      ln3,
			Lower:                        0,
			Upper:                        1,
			Delta:                        0,
			MaxContributionsPerPartition: 1,
		}},
		{"non-default options", &BoundedMeanFloat64Options{
			Lower:                        -100,
			Upper:                        555,
			Epsilon:                      ln3,
			Delta:                        1e-5,
			MaxPartitionsContributed:     5,
			MaxContributionsPerPartition: 6,
			Noise:                        noise.Gaussian(),
		}},
	} {
		bm, bmUnchanged := NewBoundedMeanFloat64(tc.opts), NewBoundedMeanFloat64(tc.opts)
		bytes, err := encode(bm)
		if err != nil {
			t.Fatalf("encode(BoundedMeanFloat64) error: %v", err)
		}
		bmUnmarshalled := new(BoundedMeanFloat64)
		if err := decode(bmUnmarshalled, bytes); err != nil {
			t.Fatalf("decode(BoundedMeanFloat64) error: %v", err)
		}
		// Check that encoding -> decoding is the identity function.
		if !cmp.Equal(bmUnchanged, bmUnmarshalled, cmp.Comparer(compareBoundedMeanFloat64)) {
			t.Errorf("decode(encode(_)): when %s got %v, want %v", tc.desc, bmUnmarshalled, bm)
		}
		if bm.state == Serialized {
			t.Errorf("BoundedMean should have its state set to Serialized, got %v , want Serialized", bm.state)
		}
	}
}

// Tests that GobEncode() returns errors correctly with different BoundedMeanFloat64 aggregation states.
func TestBoundedMeanFloat64SerializationStateChecks(t *testing.T) {
	for _, tc := range []struct {
		state   aggregationState
		wantErr bool
	}{
		{Default, false},
		{Merged, true},
		{Serialized, true},
		{ResultReturned, true},
	} {
		bm := getNoiselessBMF()
		bm.state = tc.state

		_, err := bm.GobEncode()
		if (err != nil) != tc.wantErr {
			t.Errorf("GobEncode: for state %v err got %v, want %t", tc.state, err, tc.wantErr)
		}
	}
}

func TestMeanComputeConfidenceIntervalForExplicitAlphaNumFractionsItIntoCorrectMeanConfInt(t *testing.T) {
	noNoise := noNoise{} // To skip initial argument checking.
	for _, tc := range []struct {
		meanOpt      *BoundedMeanFloat64Options
		sumConfInt   noise.ConfidenceInterval
		countConfInt noise.ConfidenceInterval
		want         noise.ConfidenceInterval
	}{
		{ // Positive lower and upper bounds for sum.
			meanOpt:      &BoundedMeanFloat64Options{
				Lower: 0.0,
				Upper: 10.0,
				Noise: noNoise,
				MaxContributionsPerPartition: 1,
			},
			sumConfInt:   noise.ConfidenceInterval{LowerBound: 0.0, UpperBound: 10.0},
			countConfInt: noise.ConfidenceInterval{LowerBound: 5.0, UpperBound: 10.0},
			want:         noise.ConfidenceInterval{LowerBound: 5.0, UpperBound: 7.0},
		},
		{ // Negative lower and upper bounds for sum.
			meanOpt:      &BoundedMeanFloat64Options{
				Lower: 0.0,
				Upper: 10.0,
				Noise: noNoise,
				MaxContributionsPerPartition: 1,
			},
			sumConfInt:   noise.ConfidenceInterval{LowerBound: -10.0, UpperBound: -1.0},
			countConfInt: noise.ConfidenceInterval{LowerBound: 2.0, UpperBound: 4.0},
			want:         noise.ConfidenceInterval{LowerBound: 0.0, UpperBound: 4.75},
		},
		{ // Negative lower bound for sum.
			meanOpt:      &BoundedMeanFloat64Options{
				Lower: 0.0,
				Upper: 10.0,
				Noise: noNoise,
				MaxContributionsPerPartition: 1,
			},
			sumConfInt:   noise.ConfidenceInterval{LowerBound: -10.0, UpperBound: 0.0},
			countConfInt: noise.ConfidenceInterval{LowerBound: 1.0, UpperBound: 10.0},
			want:         noise.ConfidenceInterval{LowerBound: 0.0, UpperBound: 5.0},
		},
		// Clamp too low bounds.
		{
			meanOpt:      &BoundedMeanFloat64Options{
				Lower: 0.0,
				Upper: 10.0,
				Noise: noNoise,
				MaxContributionsPerPartition: 1,
			},
			sumConfInt:   noise.ConfidenceInterval{LowerBound: -100.0, UpperBound: -10.0},
			countConfInt: noise.ConfidenceInterval{LowerBound: 0.0, UpperBound: 1.0},
			want:         noise.ConfidenceInterval{LowerBound: 0.0, UpperBound: 0.0},
		},
		// Clamp too high bounds.
		{
			meanOpt:      &BoundedMeanFloat64Options{
				Lower: 0.0,
				Upper: 10.0,
				Noise: noNoise,
				MaxContributionsPerPartition: 1,
			},
			sumConfInt:   noise.ConfidenceInterval{LowerBound: 100.0, UpperBound: 1000.0},
			countConfInt: noise.ConfidenceInterval{LowerBound: 1.0, UpperBound: 10.0},
			want:         noise.ConfidenceInterval{LowerBound: 10.0, UpperBound: 10.0},
		},
	} {
		mean := NewBoundedMeanFloat64(tc.meanOpt)
		mean.normalizedSum.noise = getMockConfInt(tc.sumConfInt)
		mean.count.noise = getMockConfInt(tc.countConfInt)
		mean.Result()
		got, _ := mean.computeConfidenceIntervalForExplicitAlphaNum(0.1, 0.05) // Parameters are ignored.
		if !ApproxEqual(got.LowerBound, tc.want.LowerBound) {
			t.Errorf("TestMeanComputeConfidenceIntervalForExplicitAlphaNumFractionsIntoCorrectMeanConfInt(ConfidenceInterval{%f, %f})=%0.10f, want %0.10f, LowerBounds are not equal",
				tc.meanOpt.Lower, tc.meanOpt.Upper, got.LowerBound, tc.want.LowerBound)
		}
		if !ApproxEqual(got.UpperBound, tc.want.UpperBound) {
			t.Errorf("TestMeanComputeConfidenceIntervalForExplicitAlphaNumFractionsIntoCorrectMeanConfInt(ConfidenceInterval{%f, %f})=%0.10f, want %0.10f, UpperBounds are not equal",
				tc.meanOpt.Lower, tc.meanOpt.Upper, got.UpperBound, tc.want.UpperBound)
		}
	}
}

// This test was designed to be deterministic. It checks that bounding is done correctly.
func TestMeanComputeConfidenceIntervalForExplicitAlphaNumResultConfIntInsideProvidedBoundaries(t *testing.T) {
	lower := 0.0
	upper := 1.0
	meanOpt := &BoundedMeanFloat64Options{
		Epsilon: 1.0,
		Delta: 0.123,
		Lower: lower,
		Upper: upper,
		Noise: noNoise{noise.Gaussian()},
		MaxContributionsPerPartition: 1,
	}
	mean := NewBoundedMeanFloat64(meanOpt)
	mean.Add(10.0)
	mean.Result()
	meanAlpha := []float64{0.1, 0.3, 0.5, 0.9, 0.99}
	alphaFraction := []float64{0.1, 0.25, 0.5, 0.5, 0.9}
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			alphaNum := meanAlpha[i] * alphaFraction[j]
			meanConfInt, err := mean.computeConfidenceIntervalForExplicitAlphaNum(meanAlpha[i], alphaNum)
			if err != nil {
				t.Errorf("TestMeanComputeConfidenceIntervalForExplicitAlphaNum(%f, %f) got err %v", meanAlpha, alphaNum, err)
			}
			if meanConfInt.LowerBound < lower || meanConfInt.LowerBound > upper {
				t.Errorf("TestMeanComputeConfidenceIntervalForExplicitAlphaNum: lower bound = %f is outside of provided boundaries = [%f, %f]",
					meanConfInt.LowerBound, lower, upper)
			}
			if meanConfInt.UpperBound < lower || meanConfInt.UpperBound > upper {
				t.Errorf("TestMeanComputeConfidenceIntervalForExplicitAlphaNum: upper bound = %f is outside of provided boundaries = 	[%f, %f]",
					meanConfInt.UpperBound, lower, upper)
			}
		}
	}
}

// ComputeConfidenceInterval checks that the parameters passed are the ones we expect for sum.
func (mn mockBMNoise) ComputeConfidenceIntervalFloat64(noisedX float64, l0Sensitivity int64, lInfSensitivity, epsilon, delta, alpha float64) (noise.ConfidenceInterval, error) {
	if noisedX != 100 {
		mn.t.Errorf("ComputeConfidenceIntervalFloat64: for parameter noisedX got %f, want %f", noisedX, 100.0)
	}
	if l0Sensitivity != 1 {
		mn.t.Errorf("ComputeConfidenceIntervalFloat64: for parameter l0Sensitivity got %d, want %d", l0Sensitivity, 1)
	}
	if lInfSensitivity != 3 {
		mn.t.Errorf("ComputeConfidenceIntervalFloat64: for parameter lInfSensitivity got %f, want %d", lInfSensitivity, 3)
	}
	if !ApproxEqual(epsilon, ln3*0.5) {
		mn.t.Errorf("ComputeConfidenceIntervalFloat64: for parameter epsilon got %f, want %f", epsilon, ln3*0.5)
	}
	if !ApproxEqual(delta, tenten*0.5) {
		mn.t.Errorf("ComputeConfidenceIntervalFloat64: for parameter delta got %f, want %f", delta, tenten*0.5)
	}
	if !ApproxEqual(alpha, 0.025/0.975) {
		mn.t.Errorf("ComputeConfidenceIntervalFloat64: for parameter alpha got %f, want %f", alpha, 0.02564103)
	}
	return noise.ConfidenceInterval{}, nil
}

// ComputeConfidenceIntervalInt64 checks that the parameters passed are the ones we expect for count.
func (mn mockBMNoise) ComputeConfidenceIntervalInt64(noisedX, l0Sensitivity, lInfSensitivity int64, epsilon, delta, alpha float64) (noise.ConfidenceInterval, error) {
	if noisedX != 10 {
		mn.t.Errorf("ComputeConfidenceIntervalInt64: for parameter noisedX got %d, want %d", noisedX, 10)
	}
	if l0Sensitivity != 1 {
		mn.t.Errorf("ComputeConfidenceIntervalInt64: for parameter l0Sensitivity got %d, want %d", l0Sensitivity, 1)
	}
	if lInfSensitivity != 1 {
		mn.t.Errorf("ComputeConfidenceIntervalInt64: for parameter lInfSensitivity got %d, want %d", lInfSensitivity, 1)
	}
	if !ApproxEqual(epsilon, ln3*0.5) {
		mn.t.Errorf("ComputeConfidenceIntervalInt64: for parameter epsilon got %f, want %f", epsilon, ln3*0.5)
	}
	if !ApproxEqual(delta, tenten*0.5) {
		mn.t.Errorf("ComputeConfidenceIntervalInt64: for parameter delta got %f, want %f", delta, tenten*0.5)
	}
	if alpha != alphaLevel/2 {
		mn.t.Errorf("ComputeConfidenceIntervalInt64: for parameter alpha got %f, want %f", alpha, alphaLevel/2)
	}
	return noise.ConfidenceInterval{}, nil
}

func TestMeanComputeConfidenceIntervalCallsNoiseComputeConfidenceIntervalCorrectly(t *testing.T) {
	bmf := getMockBMF(t)
	bmf.Result()
	bmf.computeConfidenceIntervalForExplicitAlphaNum(alphaLevel, alphaLevel/2)
}

// Tests that ComputeConfidenceInterval() returns errors correctly with different BoundedMeanFloat64 aggregation states.
func TestBoundedMeanFloat64ComputeConfidenceIntervalStateChecks(t *testing.T) {
	for _, tc := range []struct {
		state   aggregationState
		wantErr bool
	}{
		{ResultReturned, false},
		{Default, true},
		{Merged, true},
		{Serialized, true},
	} {
		bm := getNoiselessBMF()
		// Count and sum have to be also set to the same state
		// to allow ComputeConfidenceInterval calls.
		bm.state = tc.state
		bm.count.state = tc.state
		bm.normalizedSum.state = tc.state

		_, err := bm.ComputeConfidenceInterval(0.1)
		if (err != nil) != tc.wantErr {
			t.Errorf("ComputeConfidenceInterval: for state %v err got %v, want %t", tc.state, err, tc.wantErr)
		}
	}
}
