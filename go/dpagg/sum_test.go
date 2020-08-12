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
	"github.com/google/go-cmp/cmp"
	"github.com/grd/stat"
)

func getNoiselessBSI() *BoundedSumInt64 {
	return NewBoundedSumInt64(&BoundedSumInt64Options{
		Epsilon:                  ln3,
		Delta:                    tenten,
		MaxPartitionsContributed: 1,
		Lower:                    -1,
		Upper:                    5,
		Noise:                    noNoise{},
	})
}

func getNoiselessBSF() *BoundedSumFloat64 {
	return NewBoundedSumFloat64(&BoundedSumFloat64Options{
		Epsilon:                  ln3,
		Delta:                    tenten,
		MaxPartitionsContributed: 1,
		Lower:                    -1,
		Upper:                    5,
		Noise:                    noNoise{},
	})
}

func compareBoundedSumInt64(bs1, bs2 *BoundedSumInt64) bool {
	return bs1.epsilon == bs2.epsilon &&
		bs1.delta == bs2.delta &&
		bs1.l0Sensitivity == bs2.l0Sensitivity &&
		bs1.lInfSensitivity == bs2.lInfSensitivity &&
		bs1.lower == bs2.lower &&
		bs1.upper == bs2.upper &&
		bs1.noise == bs2.noise &&
		bs1.noiseKind == bs2.noiseKind &&
		bs1.sum == bs2.sum &&
		bs1.resultReturned == bs2.resultReturned
}

// Tests that serialization for BoundedSumInt64 works as expected.
func TestBoundedSumInt64Serialization(t *testing.T) {
	for _, tc := range []struct {
		desc string
		opts *BoundedSumInt64Options
	}{
		{"default options", &BoundedSumInt64Options{
			Epsilon: ln3,
			Delta:   0,
			Lower:   0,
			Upper:   1,
		}},
		{"non-default options", &BoundedSumInt64Options{
			Epsilon:                  ln3,
			Delta:                    1e-5,
			MaxPartitionsContributed: 5,
			Lower:                    0,
			Upper:                    1,
			Noise:                    noise.Gaussian(),
		}},
	} {
		bs, bsUnchanged := NewBoundedSumInt64(tc.opts), NewBoundedSumInt64(tc.opts)
		bytes, err := encode(bs)
		if err != nil {
			t.Fatalf("encode(BoundedSumInt64) error: %v", err)
		}
		bsUnmarshalled := new(BoundedSumInt64)
		if err := decode(bsUnmarshalled, bytes); err != nil {
			t.Fatalf("encode(BoundedSumInt64) error: %v", err)
		}
		// Check that encoding -> decoding is the identity function.
		if !cmp.Equal(bsUnchanged, bsUnmarshalled, cmp.Comparer(compareBoundedSumInt64)) {
			t.Errorf("decode(encode(_)): when %s got %v, want %v", tc.desc, bsUnmarshalled, bs)
		}
		// Check that the original BoundedSumInt64 has its resultReturned set to true after serialization.
		if !bs.resultReturned {
			t.Errorf("BoundedSumInt64 %v should have its resultReturned set to true after being serialized", bs)
		}
	}
}

func compareBoundedSumFloat64(bs1, bs2 *BoundedSumFloat64) bool {
	return bs1.epsilon == bs2.epsilon &&
		bs1.delta == bs2.delta &&
		bs1.l0Sensitivity == bs2.l0Sensitivity &&
		bs1.lInfSensitivity == bs2.lInfSensitivity &&
		bs1.lower == bs2.lower &&
		bs1.upper == bs2.upper &&
		bs1.noise == bs2.noise &&
		bs1.noiseKind == bs2.noiseKind &&
		bs1.sum == bs2.sum &&
		bs1.resultReturned == bs2.resultReturned
}

// Tests that serialization for BoundedSumFloat64 works as expected.
func TestBoundedSumFloat64Serialization(t *testing.T) {
	for _, tc := range []struct {
		desc string
		opts *BoundedSumFloat64Options
	}{
		{"default options", &BoundedSumFloat64Options{
			Epsilon: ln3,
			Delta:   0,
			Lower:   0,
			Upper:   1,
		}},
		{"non-default options", &BoundedSumFloat64Options{
			Epsilon:                  ln3,
			Delta:                    1e-5,
			MaxPartitionsContributed: 5,
			Lower:                    0,
			Upper:                    1,
			Noise:                    noise.Gaussian(),
		}},
	} {
		bs, bsUnchanged := NewBoundedSumFloat64(tc.opts), NewBoundedSumFloat64(tc.opts)
		bytes, err := encode(bs)
		if err != nil {
			t.Fatalf("encode(BoundedSumFloat64) error: %v", err)
		}
		bsUnmarshalled := new(BoundedSumFloat64)
		if err := decode(bsUnmarshalled, bytes); err != nil {
			t.Fatalf("decode(BoundedSumFloat64) error: %v", err)
		}
		// Check that encoding -> decoding is the identity function.
		if !cmp.Equal(bsUnchanged, bsUnmarshalled, cmp.Comparer(compareBoundedSumFloat64)) {
			t.Errorf("decode(encode(_)): when %s got %v, want %v", tc.desc, bsUnmarshalled, bs)
		}
		// Check that the original BoundedSumFloat64 has its resultReturned set to true after serialization.
		if !bs.resultReturned {
			t.Errorf("BoundedSumFloat64 %v should have its resultReturned set to true after being serialized", bs)
		}
	}
}

func TestGetLInfInt(t *testing.T) {
	for _, tc := range []struct {
		desc                         string
		lower                        int64
		upper                        int64
		maxContributionsPerPartition int64
		want                         int64
		wantErr                      bool
	}{
		{"lower > 0 & upper > 0 & maxContributionsPerPartition = 1", 3, 5, 1, 5, false},
		{"lower < 0 & upper > 0 & maxContributionsPerPartition = 1", -7, 5, 1, 7, false},
		{"lower < 0 & upper < 0 & maxContributionsPerPartition = 1", -7, -5, 1, 7, false},
		{"lower = math.MinInt64 & upper > 0 & maxContributionsPerPartition = 1", math.MinInt64, 5, 1, 0, true},
		{"lower < 0 & upper = math.MinInt64 & maxContributionsPerPartition = 1", -100, math.MinInt64, 1, 0, true},
		{"lower < 0 & upper = math.MinInt64 & maxContributionsPerPartition = 1", -100, math.MinInt64, 1, 0, true},
		{"lower > 0 & upper = math.MaxInt64 & maxContributionsPerPartition = math.MaxInt64", 3, math.MaxInt64, math.MaxInt64, 0, true},
		{"lower > 0 & upper = math.MaxInt64 & maxContributionsPerPartition = 2", 3, math.MaxInt64, 2, 0, true},
		{"lower > math.MinInt64 + 1 & upper > 0 & maxContributionsPerPartition = 2", math.MinInt64 + 1, 2, 2, 0, true},
	} {
		got, err := getLInfInt(tc.lower, tc.upper, tc.maxContributionsPerPartition)
		if (err != nil) != tc.wantErr {
			t.Errorf("getLInfInt: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
		if err != nil {
			continue
		}
		if got != tc.want {
			t.Errorf("getLInfInt: when %s got %d, want %d", tc.desc, got, tc.want)
		}
	}
}

func TestGetLInfFloat(t *testing.T) {
	for _, tc := range []struct {
		desc                         string
		lower                        float64
		upper                        float64
		maxContributionsPerPartition int64
		want                         float64
		wantErr                      bool
	}{
		{"lower > 0 & upper > 0", 3, 5, 1, 5, false},
		{"lower < 0 & upper > 0", -7, 5, 1, 7, false},
		{"lower < 0 & upper < 0", -7, -5, 1, 7, false},
		{"lower > 0 & upper = math.MaxFloat64 & maxContributionsPerPartition = 2", 3, math.MaxFloat64, 2, 0, true},
		{"lower = -math.MaxFloat64 & upper > 0 & maxContributionsPerPartition = 2", -math.MaxFloat64, 2, 2, 0, true},
	} {
		got, err := getLInfFloat(tc.lower, tc.upper, tc.maxContributionsPerPartition)
		if (err != nil) != tc.wantErr {
			t.Errorf("getLInfFloat: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
		if err != nil {
			continue
		}
		if got != tc.want {
			t.Errorf("getLInfFloat: when %s got %f, want %f", tc.desc, got, tc.want)
		}
	}
}

func TestNewBoundedSumInt64(t *testing.T) {
	for _, tc := range []struct {
		desc string
		opt  *BoundedSumInt64Options
		want *BoundedSumInt64
	}{
		{"MaxPartitionsContributed is not set",
			&BoundedSumInt64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noNoise{},
				maxContributionsPerPartition: 2,
			},
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           tenten,
				l0Sensitivity:   1,
				lInfSensitivity: 10,
				lower:           -1,
				upper:           5,
				noise:           noNoise{},
				sum:             0,
				resultReturned:  false,
			}},
		{"maxContributionsPerPartition is not set",
			&BoundedSumInt64Options{
				Epsilon:                  ln3,
				Delta:                    0,
				MaxPartitionsContributed: 1,
				Lower:                    -1,
				Upper:                    5,
				Noise:                    noNoise{},
			},
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 5,
				lower:           -1,
				upper:           5,
				noise:           noNoise{},
				sum:             0,
				resultReturned:  false,
			}},
		{"Noise is not set",
			&BoundedSumInt64Options{
				Epsilon:                      ln3,
				Delta:                        0,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				maxContributionsPerPartition: 2,
			},
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 10,
				lower:           -1,
				upper:           5,
				noise:           noise.Laplace(),
				noiseKind:       noise.LaplaceNoise,
				sum:             0,
				resultReturned:  false,
			}},
	} {
		got := NewBoundedSumInt64(tc.opt)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("NewBoundedSumInt64: when %s got %v, want %v", tc.desc, got, tc.want)
		}
	}
}

func TestNewBoundedSumFloat64(t *testing.T) {
	for _, tc := range []struct {
		desc string
		opt  *BoundedSumFloat64Options
		want *BoundedSumFloat64
	}{
		{"MaxPartitionsContributed is not set",
			&BoundedSumFloat64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noNoise{},
				maxContributionsPerPartition: 2,
			},
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           tenten,
				l0Sensitivity:   1,
				lInfSensitivity: 10,
				lower:           -1,
				upper:           5,
				noise:           noNoise{},
				sum:             0,
				resultReturned:  false,
			}},
		{"maxContributionsPerPartition is not set",
			&BoundedSumFloat64Options{
				Epsilon:                  ln3,
				Delta:                    0,
				MaxPartitionsContributed: 1,
				Lower:                    -1,
				Upper:                    5,
				Noise:                    noNoise{},
			},
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 5,
				lower:           -1,
				upper:           5,
				noise:           noNoise{},
				sum:             0,
				resultReturned:  false,
			}},
		{"Noise is not set",
			&BoundedSumFloat64Options{
				Epsilon:                      ln3,
				Delta:                        0,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				maxContributionsPerPartition: 2,
			},
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 10,
				lower:           -1,
				upper:           5,
				noise:           noise.Laplace(),
				noiseKind:       noise.LaplaceNoise,
				sum:             0,
				resultReturned:  false,
			}},
	} {
		got := NewBoundedSumFloat64(tc.opt)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("NewBoundedSumFloat64: when %s got %v, want %v", tc.desc, got, tc.want)
		}
	}
}

func TestAddInt64(t *testing.T) {
	bsi := getNoiselessBSI()
	bsi.Add(1)
	bsi.Add(2)
	bsi.Add(3)
	bsi.Add(4)
	got := bsi.Result()
	const want = 10
	if got != want {
		t.Errorf("Add: when 1, 2, 3, 4 were added got %d, want %d", got, want)
	}
}

func TestAddFloat64(t *testing.T) {
	bsf := getNoiselessBSF()
	bsf.Add(1.5)
	bsf.Add(2.5)
	bsf.Add(3.5)
	bsf.Add(4.5)
	got := bsf.Result()
	want := 12.0
	if !ApproxEqual(got, want) {
		t.Errorf("Add: when 1.5, 2.5, 3.5, 4.5 were added got %f, want %f", got, want)
	}
}

func TestAddFloat64IgnoresNaN(t *testing.T) {
	bsf := getNoiselessBSF()
	bsf.Add(1)
	bsf.Add(math.NaN())
	got := bsf.Result()
	want := 1.0
	if !ApproxEqual(got, want) {
		t.Errorf("Add: when NaN was added got %f, want %f", got, want)
	}
}

func TestMergeBoundedSumInt64(t *testing.T) {
	bs1 := getNoiselessBSI()
	bs2 := getNoiselessBSI()
	bs1.Add(1)
	bs1.Add(2)
	bs1.Add(3)
	bs1.Add(4)
	bs2.Add(5)
	bs1.Merge(bs2)
	got := bs1.Result()
	const want = 15
	if got != want {
		t.Errorf("Merge: when merging 2 instances of Sum got %d, want %d", got, want)
	}
	if !bs2.resultReturned {
		t.Errorf("Merge: when merging 2 instances of Sum for bs2.resultReturned got false, want true")
	}
}

func TestMergeBoundedSumFloat64(t *testing.T) {
	bs1 := getNoiselessBSF()
	bs2 := getNoiselessBSF()
	bs1.Add(1)
	bs1.Add(2)
	bs1.Add(3.5)
	bs1.Add(4)
	bs2.Add(4.5)
	bs1.Merge(bs2)
	got := bs1.Result()
	want := 15.0
	if !ApproxEqual(got, want) {
		t.Errorf("Add: when 1, 2, 3.5, 4, 4.5 were added got %f, want %f", got, want)
	}
	if !bs2.resultReturned {
		t.Errorf("Add: when 1, 2, 3.5, 4, 4.5 were added for bs2.resultReturned got false, want true")
	}
}

func TestCheckMergeBoundedSumInt64(t *testing.T) {
	for _, tc := range []struct {
		desc          string
		opt1          *BoundedSumInt64Options
		opt2          *BoundedSumInt64Options
		returnResult1 bool
		returnResult2 bool
		wantErr       bool
	}{
		{"same options, all fields filled",
			&BoundedSumInt64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				maxContributionsPerPartition: 2,
			},
			&BoundedSumInt64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				maxContributionsPerPartition: 2,
			},
			false,
			false,
			false},
		{"same options, only required fields filled",
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			false,
			false,
			false},
		{"same options, first result returned",
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			true,
			false,
			true},
		{"same options, second result returned",
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			false,
			true,
			true},
		{"different epsilon",
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			&BoundedSumInt64Options{
				Epsilon: 2,
				Lower:   -1,
				Upper:   5,
			},
			false,
			false,
			true},
		{"different delta",
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Delta:   tenten,
				Lower:   -1,
				Upper:   5,
				Noise:   noise.Gaussian(),
			},
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Delta:   tenfive,
				Lower:   -1,
				Upper:   5,
				Noise:   noise.Gaussian(),
			},
			false,
			false,
			true},
		{"different MaxPartitionsContributed",
			&BoundedSumInt64Options{
				Epsilon:                  ln3,
				MaxPartitionsContributed: 1,
				Lower:                    -1,
				Upper:                    5,
			},
			&BoundedSumInt64Options{
				Epsilon:                  ln3,
				MaxPartitionsContributed: 2,
				Lower:                    -1,
				Upper:                    5,
			},
			false,
			false,
			true},
		{"different maxContributionsPerPartition",
			&BoundedSumInt64Options{
				Epsilon:                      ln3,
				Lower:                        -1,
				Upper:                        5,
				maxContributionsPerPartition: 2,
			},
			&BoundedSumInt64Options{
				Epsilon:                      ln3,
				Lower:                        -1,
				Upper:                        5,
				maxContributionsPerPartition: 5,
			},
			false,
			false,
			true},
		{"different lower bound",
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Lower:   0,
				Upper:   5,
			},
			false,
			false,
			true},
		{"different upper bound",
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   6,
			},
			false,
			false,
			true},
		{"different noise",
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Delta:   tenten,
				Lower:   0,
				Upper:   5,
				Noise:   noise.Gaussian(),
			},
			&BoundedSumInt64Options{
				Epsilon: ln3,
				Lower:   0,
				Upper:   5,
				Noise:   noise.Laplace(),
			},
			false,
			false,
			true},
	} {
		bs1 := NewBoundedSumInt64(tc.opt1)
		bs2 := NewBoundedSumInt64(tc.opt2)

		if tc.returnResult1 {
			bs1.Result()
		}
		if tc.returnResult2 {
			bs2.Result()
		}

		if err := checkMergeBoundedSumInt64(bs1, bs2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: when %v for err got got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckMergeBoundedSumFloat64(t *testing.T) {
	for _, tc := range []struct {
		desc            string
		opt1            *BoundedSumFloat64Options
		opt2            *BoundedSumFloat64Options
		resultReturned1 bool
		resultReturned2 bool
		wantErr         bool
	}{
		{"same options, all fields filled",
			&BoundedSumFloat64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				maxContributionsPerPartition: 2,
			},
			&BoundedSumFloat64Options{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noise.Gaussian(),
				maxContributionsPerPartition: 2,
			},
			false,
			false,
			false},
		{"same options, only required fields filled",
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			false,
			false,
			false},
		{"same options, first result returned",
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			true,
			false,
			true},
		{"same options, second result returned",
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			false,
			true,
			true},
		{"different epsilon",
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			&BoundedSumFloat64Options{
				Epsilon: 2,
				Lower:   -1,
				Upper:   5,
			},
			false,
			false,
			true},
		{"different delta",
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Delta:   tenten,
				Lower:   -1,
				Upper:   5,
				Noise:   noise.Gaussian(),
			},
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Delta:   tenfive,
				Lower:   -1,
				Upper:   5,
				Noise:   noise.Gaussian(),
			},
			false,
			false,
			true},
		{"different MaxPartitionsContributed",
			&BoundedSumFloat64Options{
				Epsilon:                  ln3,
				MaxPartitionsContributed: 1,
				Lower:                    -1,
				Upper:                    5,
			},
			&BoundedSumFloat64Options{
				Epsilon:                  ln3,
				MaxPartitionsContributed: 2,
				Lower:                    -1,
				Upper:                    5,
			},
			false,
			false,
			true},
		{"different maxContributionsPerPartition",
			&BoundedSumFloat64Options{
				Epsilon:                      ln3,
				Lower:                        -1,
				Upper:                        5,
				maxContributionsPerPartition: 2,
			},
			&BoundedSumFloat64Options{
				Epsilon:                      ln3,
				Lower:                        -1,
				Upper:                        5,
				maxContributionsPerPartition: 5,
			},
			false,
			false,
			true},
		{"different lower bound",
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Lower:   0,
				Upper:   5,
			},
			false,
			false,
			true},
		{"different upper bound",
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   5,
			},
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Lower:   -1,
				Upper:   6,
			},
			false,
			false,
			true},
		{"different noise",
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Delta:   tenten,
				Lower:   -1,
				Upper:   5,
				Noise:   noise.Gaussian(),
			},
			&BoundedSumFloat64Options{
				Epsilon: ln3,
				Lower:   0,
				Upper:   5,
				Noise:   noise.Laplace(),
			},
			false,
			false,
			true},
	} {
		bs1 := NewBoundedSumFloat64(tc.opt1)
		bs2 := NewBoundedSumFloat64(tc.opt2)

		bs1.resultReturned = tc.resultReturned1
		bs2.resultReturned = tc.resultReturned2

		if err := checkMergeBoundedSumFloat64(bs1, bs2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: when %v for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestBSClampInt64(t *testing.T) {
	bsi := getNoiselessBSI()
	bsi.Add(4)  // not clamped
	bsi.Add(8)  // clamped to 5
	bsi.Add(-7) // clamped to -1
	got := bsi.Result()
	const want = 8
	if got != want {
		t.Errorf("Add: when 4, 8, -7 were added got %d, want %d", got, want)
	}
}

func TestBSClampFloat64(t *testing.T) {
	bsf := getNoiselessBSF()
	bsf.Add(3.5)  // not clamped
	bsf.Add(8.3)  // clamped to 5
	bsf.Add(-7.5) // clamped to -1
	got := bsf.Result()
	want := 7.5
	if !ApproxEqual(got, want) {
		t.Errorf("Add: when 3.5, 8.3, -7.5 were added got %f, want %f", got, want)
	}
}

func TestThresholdedResultInt64(t *testing.T) {
	// ThresholdedResult outputs the result when it is more than the threshold (5 using noNoise)
	bs1 := getNoiselessBSI()
	bs1.Add(1)
	bs1.Add(2)
	bs1.Add(3)
	bs1.Add(4)
	got := bs1.ThresholdedResult(5)
	if got == nil || *got != 10 {
		t.Errorf("ThresholdedResult(5): when 1, 2, 3, 4 were added got %v, want 10", got)
	}

	// ThresholdedResult outputs nil when it is less than the threshold
	bs2 := getNoiselessBSI()
	bs2.Add(1)
	bs2.Add(2)
	got = bs2.ThresholdedResult(5) // the parameter here is for the reader's eyes, the actual threshold value (5) is specified in noNoise.Threshold()
	if got != nil {
		t.Errorf("ThresholdedResult(5): when 1,2 were added got %v, want nil", got)
	}
}

func TestThresholdedResultFloat64(t *testing.T) {
	// ThresholdedResult outputs the result when it is more than the threshold (5 using noNoise)
	bs1 := getNoiselessBSF()
	bs1.Add(1.5)
	bs1.Add(2.5)
	bs1.Add(3.5)
	bs1.Add(4.5)
	got := bs1.ThresholdedResult(0.1)
	if got == nil || *got != 12 {
		t.Errorf("ThresholdedResult(0.1): when 1.5, 2.5, 3.5, 4.5 were added got %v, want 12", got)
	}

	// ThresholdedResult outputs nil when it is less than the threshold
	bs2 := getNoiselessBSF()
	bs2.Add(1)
	bs2.Add(2.5)
	got = bs2.ThresholdedResult(5) // the parameter here is for the reader's eyes, the actual threshold value (5) is specified in noNoise.Threshold()
	if got != nil {
		t.Errorf("ThresholdedResult(5): when 1, 2.5 were added got %v, want nil", got)
	}
}

type mockNoise struct {
	t *testing.T
	noise.Noise
}

// AddNoiseInt64 checks that the parameters passed are the ones we expect.
func (mn mockNoise) AddNoiseInt64(x, l0, lInf int64, eps, del float64) int64 {
	if x != 10 && x != 0 { // AddNoiseInt64 is initially called with a dummy value of 0, so we don't want to fail when that happens
		mn.t.Errorf("AddNoiseInt64: for parameter x got %d, want %d", x, 10)
	}
	if l0 != 1 {
		mn.t.Errorf("AddNoiseInt64: for parameter l0Sensitivity got %d, want %d", l0, 1)
	}
	if lInf != 5 {
		mn.t.Errorf("AddNoiseInt64: for parameter lInfSensitivity got %d, want %d", lInf, 5)
	}
	if !ApproxEqual(eps, ln3) {
		mn.t.Errorf("AddNoiseInt64: for parameter epsilon got %f, want %f", eps, ln3)
	}
	if !ApproxEqual(del, tenten) {
		mn.t.Errorf("AddNoiseInt64: for parameter delta got %f, want %f", del, tenten)
	}
	return 0 // ignored
}

// AddNoiseFloat64 checks that the parameters passed are the ones we expect.
func (mn mockNoise) AddNoiseFloat64(x float64, l0 int64, lInf, eps, del float64) float64 {
	if !ApproxEqual(x, 12.0) && !ApproxEqual(x, 0.0) {
		// AddNoiseFloat64 is initially called with a dummy value of 0, so we don't want to fail when that happens
		mn.t.Errorf("AddNoiseFloat64: for parameter x  got %f, want %f", x, 12.0)
	}
	if l0 != 1 {
		mn.t.Errorf("AddNoiseFloat64: for parameter l0Sensitivity got %d, want %d", l0, 1)
	}
	if !ApproxEqual(lInf, 5.0) {
		mn.t.Errorf("AddNoiseFloat64: for parameter lInfSensitivity got %f, want %f", lInf, 5.0)
	}
	if !ApproxEqual(eps, ln3) {
		mn.t.Errorf("AddNoiseFloat64: for parameter epsilon got %f, want %f", eps, ln3)
	}
	if !ApproxEqual(del, tenten) {
		mn.t.Errorf("AddNoiseFloat64: for parameter delta got %f, want %f", del, tenten)
	}
	return 0 // ignored
}

// Threshold checks that the parameters passed are the ones we expect.
func (mn mockNoise) Threshold(l0 int64, lInf, eps, del, thresholdDelta float64) float64 {
	if !ApproxEqual(thresholdDelta, 10.0) {
		mn.t.Errorf("Threshold: for parameter thresholdDelta got %f, want %f", thresholdDelta, 10.0)
	}
	if l0 != 1 {
		mn.t.Errorf("Threshold: for parameter l0Sensitivity got %d, want %d", l0, 1)
	}
	if !ApproxEqual(lInf, 5.0) {
		mn.t.Errorf("Threshold: for parameter l0Sensitivity got %f, want %f", lInf, 5.0)
	}
	if !ApproxEqual(eps, ln3) {
		mn.t.Errorf("Threshold: for parameter epsilon got %f, want %f", eps, ln3)
	}
	if !ApproxEqual(del, tenten) {
		mn.t.Errorf("Threshold: for parameter delta got %f, want %f", del, tenten)
	}
	return 0 // ignored
}

func getMockBSI(t *testing.T) *BoundedSumInt64 {
	return NewBoundedSumInt64(&BoundedSumInt64Options{
		Epsilon:                  ln3,
		Delta:                    tenten,
		MaxPartitionsContributed: 1,
		Lower:                    -1,
		Upper:                    5,
		Noise:                    mockNoise{t: t},
	})
}

func getMockBSF(t *testing.T) *BoundedSumFloat64 {
	return NewBoundedSumFloat64(&BoundedSumFloat64Options{
		Epsilon:                  ln3,
		Delta:                    tenten,
		MaxPartitionsContributed: 1,
		Lower:                    -1,
		Upper:                    5,
		Noise:                    mockNoise{t: t},
	})
}

func TestNoiseIsCorrectlyCalledInt64(t *testing.T) {
	bsi := getMockBSI(t)
	bsi.Add(1)
	bsi.Add(2)
	bsi.Add(3)
	bsi.Add(4)
	bsi.Result() // will fail if parameters are wrong
}

func TestNoiseIsCorrectlyCalledFloat64(t *testing.T) {
	bsf := getMockBSF(t)
	bsf.Add(3)
	bsf.Add(2)
	bsf.Add(3)
	bsf.Add(4)
	bsf.Result() // will fail if parameters are wrong
}

func TestThresholdsCorrectlyCalledForSumFloat64(t *testing.T) {
	bsf := getMockBSF(t)
	bsf.Add(3)
	bsf.Add(2)
	bsf.Add(3)
	bsf.Add(4)
	bsf.ThresholdedResult(10) // will fail if parameters are wrong
}

func TestThresholdsCorrectlyCalledForSumInt64(t *testing.T) {
	bsf := getMockBSI(t)
	bsf.Add(1)
	bsf.Add(2)
	bsf.Add(3)
	bsf.Add(4)
	bsf.ThresholdedResult(10) // will fail if parameters are wrong
}

func TestBSEquallyInitializedInt64(t *testing.T) {
	for _, tc := range []struct {
		desc  string
		bs1   *BoundedSumInt64
		bs2   *BoundedSumInt64
		equal bool
	}{
		{
			"equal parameters",
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			true,
		},
		{
			"different epsilon",
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			&BoundedSumInt64{
				epsilon:         1,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			false,
		},
		{
			"different delta",
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0.5,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.GaussianNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0.6,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.GaussianNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			false,
		},
		{
			"different l0Sensitivity",
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   2,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			false,
		},
		{
			"different lInfSensitivity",
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 2,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			false,
		},
		{
			"different lower",
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           -1,
				upper:           1,
				sum:             1,
				resultReturned:  false},
			false,
		},
		{
			"different upper",
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			&BoundedSumInt64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           2,
				sum:             1,
				resultReturned:  false},
			false,
		},
	} {
		if bsEquallyInitializedint64(tc.bs1, tc.bs2) != tc.equal {
			t.Errorf("bsEquallyInitializedint64: when %v got %t, want %t", tc.desc, !tc.equal, tc.equal)
		}
	}
}

func TestBSEquallyInitializedFloat64(t *testing.T) {
	for _, tc := range []struct {
		desc  string
		bs1   *BoundedSumFloat64
		bs2   *BoundedSumFloat64
		equal bool
	}{
		{
			"equal parameters",
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			true,
		},
		{
			"different epsilon",
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			&BoundedSumFloat64{
				epsilon:         1,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			false,
		},
		{
			"different delta",
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0.5,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.GaussianNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0.6,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.GaussianNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			false,
		},
		{
			"different l0Sensitivity",
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   2,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			false,
		},
		{
			"different lInfSensitivity",
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 2,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			false,
		},
		{
			"different lower",
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           -1,
				upper:           1,
				sum:             1,
				resultReturned:  false},
			false,
		},
		{
			"different upper",
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           1,
				sum:             0,
				resultReturned:  false},
			&BoundedSumFloat64{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				lower:           0,
				upper:           2,
				sum:             1,
				resultReturned:  false},
			false,
		},
	} {
		if bsEquallyInitializedFloat64(tc.bs1, tc.bs2) != tc.equal {
			t.Errorf("bsEquallyInitializedFloat64: when %v got %t, want %t", tc.desc, !tc.equal, tc.equal)
		}
	}
}

func TestBoundedSumInt64IsUnbiased(t *testing.T) {
	const numberOfSamples = 100000
	for _, tc := range []struct {
		desc     string
		opt      *BoundedSumInt64Options
		rawEntry int64
		variance float64
	}{
		{
			opt: &BoundedSumInt64Options{
				Epsilon:                  ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 1,
				Lower:                    0,
				Upper:                    1,
				Noise:                    noise.Gaussian(),
			},
			rawEntry: 0,
			variance: 11.9, // approximated via a simulation
		}, {
			opt: &BoundedSumInt64Options{
				Epsilon:                  2.0 * ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 1,
				Lower:                    0,
				Upper:                    1,
				Noise:                    noise.Gaussian(),
			},
			rawEntry: 0,
			variance: 3.5, // approximated via a simulation
		}, {
			opt: &BoundedSumInt64Options{
				Epsilon:                  ln3,
				Delta:                    0.01,
				MaxPartitionsContributed: 1,
				Lower:                    0,
				Upper:                    1,
				Noise:                    noise.Gaussian(),
			},
			rawEntry: 0,
			variance: 3.2, // approximated via a simulation
		}, {
			opt: &BoundedSumInt64Options{
				Epsilon:                  ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 25,
				Lower:                    0,
				Upper:                    1,
				Noise:                    noise.Gaussian(),
			},
			rawEntry: 0,
			variance: 295.0, // approximated via a simulation
		}, {
			opt: &BoundedSumInt64Options{
				Epsilon:                  ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 1,
				Lower:                    0,
				Upper:                    1,
				Noise:                    noise.Gaussian(),
			},
			rawEntry: 1,
			variance: 11.9, // approximated via a simulation
		}, {
			opt: &BoundedSumInt64Options{
				Epsilon:                  ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 1,
				Lower:                    -1,
				Upper:                    1,
				Noise:                    noise.Gaussian(),
			},
			rawEntry: -1,
			variance: 11.9, // approximated via a simulation
		}, {
			opt: &BoundedSumInt64Options{
				Epsilon:                  ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 1,
				Lower:                    0,
				Upper:                    1,
				Noise:                    noise.Laplace(),
			},
			rawEntry: 0,
			variance: 1.8, // approximated via a simulation
		}, {
			opt: &BoundedSumInt64Options{
				Epsilon:                  2.0 * ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 1,
				Lower:                    0,
				Upper:                    1,
				Noise:                    noise.Laplace(),
			},
			rawEntry: 0,
			variance: 0.5, // approximated via a simulation
		}, {
			opt: &BoundedSumInt64Options{
				Epsilon:                  ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 25,
				Lower:                    0,
				Upper:                    1,
				Noise:                    noise.Laplace(),
			},
			rawEntry: 0,
			variance: 1035.0, // approximated via a simulation
		}, {
			opt: &BoundedSumInt64Options{
				Epsilon:                  ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 1,
				Lower:                    0,
				Upper:                    1,
				Noise:                    noise.Laplace(),
			},
			rawEntry: 1,
			variance: 1.8, // approximated via a simulation
		}, {
			opt: &BoundedSumInt64Options{
				Epsilon:                  ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 1,
				Lower:                    -1,
				Upper:                    1,
				Noise:                    noise.Laplace(),
			},
			rawEntry: -1,
			variance: 1.8, // approximated via a simulation
		},
	} {
		sumSamples := make(stat.IntSlice, numberOfSamples)
		for i := 0; i < numberOfSamples; i++ {
			sum := NewBoundedSumInt64(tc.opt)
			sum.Add(tc.rawEntry)
			sumSamples[i] = sum.Result()
		}
		sampleMean := stat.Mean(sumSamples)
		// Assuming that sum is unbiased, each sample should have a mean of tc.rawEntry
		// and a variance of tc.variance. The resulting sampleMean is approximately Gaussian
		// distributed with the same mean and a variance of tc.variance / numberOfSamples.
		//
		// The tolerance is set to the 99.9995% quantile of the anticipated distribution
		// of sampleMean. Thus, the test falsely rejects with a probability of 10⁻⁵.
		tolerance := 4.41717 * math.Sqrt(tc.variance/float64(numberOfSamples))

		if math.Abs(sampleMean-float64(tc.rawEntry)) > tolerance {
			t.Errorf("got mean = %f, want %f (parameters %+v)", sampleMean, float64(tc.rawEntry), tc)
		}
	}
}

func TestBoundedSumFloat64IsUnbiased(t *testing.T) {
	const numberOfSamples = 100000
	for _, tc := range []struct {
		desc     string
		opt      *BoundedSumFloat64Options
		rawEntry float64
		variance float64
	}{
		{
			opt: &BoundedSumFloat64Options{
				Epsilon:                  ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 1,
				Lower:                    0.0,
				Upper:                    1.0,
				Noise:                    noise.Gaussian(),
			},
			rawEntry: 0.0,
			variance: 11.735977,
		}, {
			opt: &BoundedSumFloat64Options{
				Epsilon:                  2.0 * ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 1,
				Lower:                    0.0,
				Upper:                    1.0,
				Noise:                    noise.Gaussian(),
			},
			rawEntry: 0.0,
			variance: 3.3634987,
		}, {
			opt: &BoundedSumFloat64Options{
				Epsilon:                  ln3,
				Delta:                    0.01,
				MaxPartitionsContributed: 1,
				Lower:                    0.0,
				Upper:                    1.0,
				Noise:                    noise.Gaussian(),
			},
			rawEntry: 0.0,
			variance: 3.0625,
		}, {
			opt: &BoundedSumFloat64Options{
				Epsilon:                  ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 25,
				Lower:                    0.0,
				Upper:                    1.0,
				Noise:                    noise.Gaussian(),
			},
			rawEntry: 0.0,
			variance: 293.399425,
		}, {
			opt: &BoundedSumFloat64Options{
				Epsilon:                  ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 1,
				Lower:                    -0.5,
				Upper:                    0.0,
				Noise:                    noise.Gaussian(),
			},
			rawEntry: 0.0,
			variance: 2.93399425,
		}, {
			opt: &BoundedSumFloat64Options{
				Epsilon:                  ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 1,
				Lower:                    0.0,
				Upper:                    1.0,
				Noise:                    noise.Gaussian(),
			},
			rawEntry: 1.0,
			variance: 11.735977,
		}, {
			opt: &BoundedSumFloat64Options{
				Epsilon:                  ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 1.0,
				Lower:                    -1.0,
				Upper:                    1.0,
				Noise:                    noise.Gaussian(),
			},
			rawEntry: -1.0,
			variance: 11.735977,
		}, {
			opt: &BoundedSumFloat64Options{
				Epsilon:                  ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 1,
				Lower:                    0.0,
				Upper:                    1.0,
				Noise:                    noise.Laplace(),
			},
			rawEntry: 0.0,
			variance: 2.0 / (ln3 * ln3),
		}, {
			opt: &BoundedSumFloat64Options{
				Epsilon:                  2.0 * ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 1,
				Lower:                    0.0,
				Upper:                    1.0,
				Noise:                    noise.Laplace(),
			},
			rawEntry: 0.0,
			variance: 2.0 / (4.0 * ln3 * ln3),
		}, {
			opt: &BoundedSumFloat64Options{
				Epsilon:                  ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 25,
				Lower:                    0.0,
				Upper:                    1.0,
				Noise:                    noise.Laplace(),
			},
			rawEntry: 0,
			variance: 2.0 * 625.0 / (ln3 * ln3),
		}, {
			opt: &BoundedSumFloat64Options{
				Epsilon:                  ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 1,
				Lower:                    -0.5,
				Upper:                    0.0,
				Noise:                    noise.Laplace(),
			},
			rawEntry: 0.0,
			variance: 2.0 / (4.0 * ln3 * ln3),
		}, {
			opt: &BoundedSumFloat64Options{
				Epsilon:                  ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 1,
				Lower:                    0.0,
				Upper:                    1.0,
				Noise:                    noise.Laplace(),
			},
			rawEntry: 1.0,
			variance: 2.0 / (ln3 * ln3),
		}, {
			opt: &BoundedSumFloat64Options{
				Epsilon:                  ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 1,
				Lower:                    -1.0,
				Upper:                    1.0,
				Noise:                    noise.Laplace(),
			},
			rawEntry: -1.0,
			variance: 2.0 / (ln3 * ln3),
		},
	} {
		sumSamples := make(stat.Float64Slice, numberOfSamples)
		for i := 0; i < numberOfSamples; i++ {
			sum := NewBoundedSumFloat64(tc.opt)
			sum.Add(tc.rawEntry)
			sumSamples[i] = sum.Result()
		}
		sampleMean := stat.Mean(sumSamples)
		// Assuming that sum is unbiased, each sample should have a mean of tc.rawEntry
		// and a variance of tc.variance. The resulting sampleMean is approximately Gaussian
		// distributed with the same mean and a variance of tc.variance / numberOfSamples.
		//
		// The tolerance is set to the 99.9995% quantile of the anticipated distribution
		// of sampleMean. Thus, the test falsely rejects with a probability of 10⁻⁵.
		tolerance := 4.41717 * math.Sqrt(tc.variance/float64(numberOfSamples))

		if math.Abs(sampleMean-tc.rawEntry) > tolerance {
			t.Errorf("got mean = %f, want %f (parameters %+v)", sampleMean, tc.rawEntry, tc)
		}
	}
}
