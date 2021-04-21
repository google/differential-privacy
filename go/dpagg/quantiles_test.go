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
	"math/rand"
	"reflect"
	"sort"
	"testing"

	"github.com/google/differential-privacy/go/noise"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestNewBoundedQuantiles(t *testing.T) {
	for _, tc := range []struct {
		desc string
		opt  *BoundedQuantilesOptions
		want *BoundedQuantiles
	}{
		{"MaxPartitionsContributed is not set",
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noNoise{},
				MaxContributionsPerPartition: 2,
				TreeHeight:                   4,
				BranchingFactor:              16,
			},
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             tenten,
				lower:             -1,
				upper:             5,
				l0Sensitivity:     4, // Uses default MaxPartitionsContributed of 1.
				lInfSensitivity:   2,
				Noise:             noNoise{},
				noiseKind:         noise.Unrecognised,
				treeHeight:        4,
				branchingFactor:   16,
				numLeaves:         65536,
				leftmostLeafIndex: 4369,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             defaultState,
			}},
		{"Noise is not set",
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        0,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
			},
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             0,
				lower:             -1,
				upper:             5,
				l0Sensitivity:     8,
				lInfSensitivity:   2,
				noiseKind:         noise.LaplaceNoise,
				Noise:             noise.Laplace(),
				treeHeight:        4,
				branchingFactor:   16,
				numLeaves:         65536,
				leftmostLeafIndex: 4369,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             defaultState,
			}},
		{"Tree Height and Branching Factor are not set",
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				Noise:                        noNoise{},
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     1,
			},
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             tenten,
				lower:             -1,
				upper:             5,
				l0Sensitivity:     4,
				lInfSensitivity:   2,
				Noise:             noNoise{},
				noiseKind:         noise.Unrecognised,
				treeHeight:        4,  // Uses default treeHeight of 4.
				branchingFactor:   16, // Uses default branchingFactor of 16.
				numLeaves:         65536,
				leftmostLeafIndex: 4369,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             defaultState,
			}},
	} {
		got := NewBoundedQuantiles(tc.opt)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("NewBoundedQuantiles: when %s got %+v, want %+v", tc.desc, got, tc.want)
		}
	}
}

func TestBQNoiseIsCorrectlyCalled(t *testing.T) {
	bq := getMockBQ(t)
	bq.Add(1.0)
	bq.Result(0.5) // will fail if parameters are wrong
}

func TestBQNoInput(t *testing.T) {
	bq := getNoiselessBQ()
	got := bq.Result(0.5)
	want := 0.0 // When there are no inputs, we linearly interpolate.
	if !ApproxEqual(got, want) {
		t.Errorf("Result: when there is no input data got=%f, want=%f", got, want)
	}
}

func TestBQAdd(t *testing.T) {
	bq := getNoiselessBQ()
	entries := createEntries()
	for _, i := range entries {
		bq.Add(i)
	}
	sort.Float64s(entries)
	for _, rank := range getRanks() {
		got := bq.Result(rank)
		want := entries[int(math.Round(float64((len(entries)-1))*adjustRank(rank)))]
		// When no noise is added, computeResult should return a value that differs from the true
		// quantile by no more than the size of the buckets the range is partitioned into, i.e.,
		// (upper - lower) / branchingFactor^treeHeight.
		tolerance := 0.001 // (upper - lower) / branchingFactor^treeHeight
		if !cmp.Equal(got, want, cmpopts.EquateApprox(0, tolerance)) {
			t.Errorf("Add: for rank %f got %f, want %f", rank, got, want)
		}
	}
}

func TestBQAddIgnoresNaN(t *testing.T) {
	bq := getNoiselessBQ()
	entries := createEntries()
	for _, i := range entries {
		bq.Add(i)
	}
	for i := 0; i < 10; i++ {
		// Adding multiple NaN's to make sure it would pollute the result if they were not ignored.
		bq.Add(math.NaN())
	}
	sort.Float64s(entries)
	got := bq.Result(0.5)
	want := entries[500]
	// When no noise is added, computeResult should return a value that differs from the true
	// quantile by no more than the size of the buckets the range is partitioned into, i.e.,
	// (upper - lower) / branchingFactor^treeHeight.
	tolerance := 0.001 // (upper - lower) / branchingFactor^treeHeight
	if !cmp.Equal(got, want, cmpopts.EquateApprox(0, tolerance)) {
		t.Errorf("Add: when NaN was added for rank %f got %f, want %f", 0.5, got, want)
	}
}

func TestBQClamp(t *testing.T) {
	bq := getNoiselessBQ()
	for i := 0; i < 500; i++ {
		bq.Add(-100.0) // Clamped to -5.
		bq.Add(100.)   // Clamped to 5.
	}
	// When no noise is added, computeResult should return a value that differs from the true
	// quantile by no more than the size of the buckets the range is partitioned into, i.e.,
	// (upper - lower) / branchingFactor^treeHeight.
	tolerance := 0.001 // (upper - lower) / branchingFactor^treeHeight

	rank := 0.25
	got := bq.Result(rank)
	want := -5.0
	if !cmp.Equal(got, want, cmpopts.EquateApprox(0, tolerance)) {
		t.Errorf("Add: Did not clamp to lower bound, rank %f got %f, want %f", rank, got, want)
	}

	rank = 0.75
	got = bq.Result(rank)
	want = 5.0
	if !cmp.Equal(got, want, cmpopts.EquateApprox(0, tolerance)) {
		t.Errorf("Add: Did not clamp to upper bound, rank %f got %f, want %f", rank, got, want)
	}
}

// Tests that multiple calls for the same rank returns the same result.
func TestBQMultipleCallsForTheSameRank(t *testing.T) {
	bq := getNoiselessBQ()
	for _, i := range createEntries() {
		bq.Add(i)
	}
	for _, rank := range getRanks() {
		got := bq.Result(rank)
		want := bq.Result(rank)
		if !cmp.Equal(got, want) {
			t.Errorf("Add: Wanted the same result for multiple calls for rank %f got %f, want %f", rank, got, want)
		}
	}
}

// Tests that Result() is invariant to entry order.
func TestBQInvariantToEntryOrder(t *testing.T) {
	bq1 := getNoiselessBQ()
	bq2 := getNoiselessBQ()
	entries := createEntries()
	// The list of entries contains 1001 elements. However, we only add the first 997. The reason
	// is that 997 is a prime number, which allows us to shuffle the entires easily using modular
	// arithmetic.
	for i := 0; i < 997; i++ {
		bq1.Add(entries[i])
		// Adding entries with an arbitrary step length of 643. Because the two values are coprime,
		// all entries between 0 and 997 will be added.
		bq2.Add(entries[i*643%997])
	}
	for _, rank := range getRanks() {
		got := bq1.Result(rank)
		want := bq2.Result(rank)
		if !cmp.Equal(got, want) {
			t.Errorf("Add: Wanted the same result for same list of entries with a different order for rank %f got %f, want %f", rank, got, want)
		}
	}
}

// Tests that pre-clamping before Add and not clamping and having the library do the clamping yields
// the same result.
func TestBQInvariantToPreClamping(t *testing.T) {
	bq1 := getNoiselessBQ()
	bq1.lower = -1.0
	bq1.upper = 1.0
	bq2 := getNoiselessBQ()
	bq2.lower = -1.0
	bq2.upper = 1.0

	for _, i := range createEntries() {
		bq1.Add(i)
		bq2.Add(math.Min(math.Max(-1.0, i), 1.0))
	}
	for _, rank := range getRanks() {
		got := bq1.Result(rank)
		want := bq2.Result(rank)
		if !cmp.Equal(got, want) {
			t.Errorf("Add: Wanted the same result for pre-clamped entries and regularly clamped entries for rank %f got %f, want %f", rank, got, want)
		}
	}
}

// Tests that Result(rank) increases monotonically with rank even with noise.
func TestBQIncreasesMonotonically(t *testing.T) {
	bq := getNoiselessBQ()
	bq.Noise = noise.Gaussian() // This property should hold even if noise is added.

	for _, i := range createEntries() {
		bq.Add(i)
	}
	lastResult := math.Inf(-1)
	for _, rank := range getRanks() {
		got := bq.Result(rank)
		if got < lastResult {
			t.Errorf("Add: Expected monotonically increasing result for rank %f got %f, lastResult %f", rank, got, lastResult)
		}
		lastResult = got
	}
}

func TestBoundedQuantilesResultSetsStateCorrectly(t *testing.T) {
	bq := getNoiselessBQ()
	bq.Result(0.5)

	if bq.state != resultReturned {
		t.Errorf("BoundedQuantiles should have its state set to ResultReturned, got %v, want ResultReturned", bq.state)
	}
}

// getRanks returns 1001 ranks equally distributed between 0.0 and 1.0 (both inclusive).
func getRanks() []float64 {
	ranks := make([]float64, 1001)
	for i := 0; i <= 1000; i++ {
		ranks[i] = float64(i) / 1000.0
	}
	return ranks
}

// createEntries returns 1001 random entries between -5 and 5.
func createEntries() []float64 {
	entries := make([]float64, 1001)
	for i := 0; i <= 1000; i++ {
		entries[i] = rand.Float64()*10 - 5
	}
	return entries
}

func getNoiselessBQ() *BoundedQuantiles {
	return NewBoundedQuantiles(&BoundedQuantilesOptions{
		Epsilon:                      ln3,
		Delta:                        tenten,
		MaxPartitionsContributed:     1,
		MaxContributionsPerPartition: 1,
		Lower:                        -5,
		Upper:                        5,
		TreeHeight:                   4,
		BranchingFactor:              10,
		Noise:                        noNoise{},
	})
}

func getMockBQ(t *testing.T) *BoundedQuantiles {
	return NewBoundedQuantiles(&BoundedQuantilesOptions{
		Epsilon:                      ln3,
		Delta:                        tenten,
		MaxPartitionsContributed:     2,
		MaxContributionsPerPartition: 3,
		Lower:                        -5,
		Upper:                        5,
		TreeHeight:                   4,
		BranchingFactor:              10,
		Noise:                        mockBQNoise{t: t},
	})
}

type mockBQNoise struct {
	t *testing.T
	noise.Noise
}

// AddNoiseFloat64 checks that the parameters passed are the ones we expect.
func (mn mockBQNoise) AddNoiseFloat64(x float64, l0 int64, lInf, eps, del float64) float64 {
	if !ApproxEqual(x, 1.0) && !ApproxEqual(x, 0.0) {
		// We have a single element in the tree, meaning that bucket counts will either be 0 or 1.
		mn.t.Errorf("AddNoiseFloat64: for parameter x got %f, want 0.0 or 1.0", x)
	}
	if l0 != 8 { // treeHeight * maxPartitionsContributed
		mn.t.Errorf("AddNoiseFloat64: for parameter l0Sensitivity got %d, want %d", l0, 8)
	}
	if !ApproxEqual(lInf, 3.0) { // maxContributionsPerPartition
		mn.t.Errorf("AddNoiseFloat64: for parameter lInfSensitivity got %f, want %f", lInf, 3.0)
	}
	if !ApproxEqual(eps, ln3) {
		mn.t.Errorf("AddNoiseFloat64: for parameter epsilon got %f, want %f", eps, ln3)
	}
	if !ApproxEqual(del, tenten) {
		mn.t.Errorf("AddNoiseFloat64: for parameter delta got %f, want %f", del, tenten)
	}
	return x
}

func TestCheckMergeBoundedQuantilesCompatibility(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		opt1    *BoundedQuantilesOptions
		opt2    *BoundedQuantilesOptions
		wantErr bool
	}{
		{"same options",
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			false},
		{"different epsilon",
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			&BoundedQuantilesOptions{
				Epsilon:                      1,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			true},
		{"different delta",
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenfive,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			true},
		{"different MaxPartitionsContributed",
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     1,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			true},
		{"different lower bound",
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        0,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			true},
		{"different upper bound",
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        6,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			true},
		{"different noise",
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        0,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Laplace(),
			},
			true},
		{"different maxContributionsPerPartition",
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 1,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			true},
		{"different treeHeight",
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   5,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			true},
		{"different branchingFactor",
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   4,
				BranchingFactor:              16,
				Noise:                        noise.Gaussian(),
			},
			&BoundedQuantilesOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Lower:                        -1,
				Upper:                        5,
				MaxContributionsPerPartition: 2,
				MaxPartitionsContributed:     2,
				TreeHeight:                   5,
				BranchingFactor:              8,
				Noise:                        noise.Gaussian(),
			},
			true},
	} {
		bq1 := NewBoundedQuantiles(tc.opt1)
		bq2 := NewBoundedQuantiles(tc.opt2)

		if err := checkMergeBoundedQuantiles(bq1, bq2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: when %s for err got %v, wantErr %t", tc.desc, err, tc.wantErr)
		}
	}
}

// Tests that checkMergeBoundedQuantiles() returns errors correctly with different BoundedQuantiles aggregation states.
func TestCheckMergeBoundedQuantilesStateChecks(t *testing.T) {
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
		bq1 := getNoiselessBQ()
		bq2 := getNoiselessBQ()

		bq1.state = tc.state1
		bq2.state = tc.state2

		if err := checkMergeBoundedQuantiles(bq1, bq2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: when states [%v, %v] for err got %v, wantErr %t", tc.state1, tc.state2, err, tc.wantErr)
		}
	}
}

func TestBQEquallyInitialized(t *testing.T) {
	for _, tc := range []struct {
		desc  string
		bq1   *BoundedQuantiles
		bq2   *BoundedQuantiles
		equal bool
	}{
		{
			"equal parameters",
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             0,
				lower:             -1,
				upper:             5,
				l0Sensitivity:     8,
				lInfSensitivity:   2,
				noiseKind:         noise.LaplaceNoise,
				Noise:             noise.Laplace(),
				treeHeight:        4,
				branchingFactor:   16,
				numLeaves:         65536,
				leftmostLeafIndex: 4369,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             defaultState},
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             0,
				lower:             -1,
				upper:             5,
				l0Sensitivity:     8,
				lInfSensitivity:   2,
				noiseKind:         noise.LaplaceNoise,
				Noise:             noise.Laplace(),
				treeHeight:        4,
				branchingFactor:   16,
				numLeaves:         65536,
				leftmostLeafIndex: 4369,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             defaultState},
			true,
		},
		{
			"different lower",
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             0,
				lower:             -3,
				upper:             5,
				l0Sensitivity:     8,
				lInfSensitivity:   2,
				noiseKind:         noise.LaplaceNoise,
				Noise:             noise.Laplace(),
				treeHeight:        4,
				branchingFactor:   16,
				numLeaves:         65536,
				leftmostLeafIndex: 4369,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             defaultState},
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             0,
				lower:             -1,
				upper:             5,
				l0Sensitivity:     8,
				lInfSensitivity:   2,
				noiseKind:         noise.LaplaceNoise,
				Noise:             noise.Laplace(),
				treeHeight:        4,
				branchingFactor:   16,
				numLeaves:         65536,
				leftmostLeafIndex: 4369,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             defaultState},
			false,
		},
		{
			"different upper",
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             0,
				lower:             -1,
				upper:             7,
				l0Sensitivity:     8,
				lInfSensitivity:   2,
				noiseKind:         noise.LaplaceNoise,
				Noise:             noise.Laplace(),
				treeHeight:        4,
				branchingFactor:   16,
				numLeaves:         65536,
				leftmostLeafIndex: 4369,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             defaultState},
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             0,
				lower:             -1,
				upper:             5,
				l0Sensitivity:     8,
				lInfSensitivity:   2,
				noiseKind:         noise.LaplaceNoise,
				Noise:             noise.Laplace(),
				treeHeight:        4,
				branchingFactor:   16,
				numLeaves:         65536,
				leftmostLeafIndex: 4369,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             defaultState},
			false,
		},
		{
			"different treeHeight",
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             0,
				lower:             -1,
				upper:             5,
				l0Sensitivity:     6,
				lInfSensitivity:   2,
				noiseKind:         noise.LaplaceNoise,
				Noise:             noise.Laplace(),
				treeHeight:        3,
				branchingFactor:   16,
				numLeaves:         4096,
				leftmostLeafIndex: 273,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             defaultState},
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             0,
				lower:             -1,
				upper:             5,
				l0Sensitivity:     8,
				lInfSensitivity:   2,
				noiseKind:         noise.LaplaceNoise,
				Noise:             noise.Laplace(),
				treeHeight:        4,
				branchingFactor:   16,
				numLeaves:         65536,
				leftmostLeafIndex: 4369,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             defaultState},
			false,
		},
		{
			"different branchingFactor",
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             0,
				lower:             -1,
				upper:             5,
				l0Sensitivity:     8,
				lInfSensitivity:   2,
				noiseKind:         noise.LaplaceNoise,
				Noise:             noise.Laplace(),
				treeHeight:        4,
				branchingFactor:   8,
				numLeaves:         4096,
				leftmostLeafIndex: 585,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             defaultState},
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             0,
				lower:             -1,
				upper:             5,
				l0Sensitivity:     8,
				lInfSensitivity:   2,
				noiseKind:         noise.LaplaceNoise,
				Noise:             noise.Laplace(),
				treeHeight:        4,
				branchingFactor:   16,
				numLeaves:         65536,
				leftmostLeafIndex: 4369,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             defaultState},
			false,
		},
		{
			"different state",
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             0,
				lower:             -1,
				upper:             5,
				l0Sensitivity:     8,
				lInfSensitivity:   2,
				noiseKind:         noise.LaplaceNoise,
				Noise:             noise.Laplace(),
				treeHeight:        4,
				branchingFactor:   16,
				numLeaves:         65536,
				leftmostLeafIndex: 4369,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             defaultState},
			&BoundedQuantiles{
				epsilon:           ln3,
				delta:             0,
				lower:             -1,
				upper:             5,
				l0Sensitivity:     8,
				lInfSensitivity:   2,
				noiseKind:         noise.LaplaceNoise,
				Noise:             noise.Laplace(),
				treeHeight:        4,
				branchingFactor:   16,
				numLeaves:         65536,
				leftmostLeafIndex: 4369,
				tree:              make(map[int]int64),
				noisedTree:        make(map[int]float64),
				state:             merged},
			false,
		},
	} {
		if bqEquallyInitialized(tc.bq1, tc.bq2) != tc.equal {
			t.Errorf("bqEquallyInitialized: when %s got %t, want %t", tc.desc, !tc.equal, tc.equal)
		}
	}
}

// Tests that serialization for BoundedQuantiles works as expected.
func TestBQSerialization(t *testing.T) {
	for _, tc := range []struct {
		desc string
		opts *BoundedQuantilesOptions
	}{
		{"default options", &BoundedQuantilesOptions{
			Epsilon:                      ln3,
			Lower:                        0,
			Upper:                        1,
			Delta:                        0,
			MaxContributionsPerPartition: 1,
		}},
		{"non-default options", &BoundedQuantilesOptions{
			Lower:                        -100,
			Upper:                        555,
			Epsilon:                      ln3,
			Delta:                        1e-5,
			MaxPartitionsContributed:     5,
			MaxContributionsPerPartition: 6,
			TreeHeight:                   3,
			BranchingFactor:              12,
			Noise:                        noise.Gaussian(),
		}},
	} {
		bq, bqUnchanged := NewBoundedQuantiles(tc.opts), NewBoundedQuantiles(tc.opts)
		// Insert same elements to both.
		bq.Add(1.0)
		bqUnchanged.Add(1.0)
		bq.Add(2.0)
		bqUnchanged.Add(2.0)
		bytes, err := encode(bq)
		if err != nil {
			t.Fatalf("encode(BoundedQuantiles) error: %v", err)
		}
		bqUnmarshalled := new(BoundedQuantiles)
		if err := decode(bqUnmarshalled, bytes); err != nil {
			t.Fatalf("decode(BoundedQuantiles) error: %v", err)
		}
		// Check that encoding -> decoding is the identity function.
		if !cmp.Equal(bqUnchanged, bqUnmarshalled, cmp.Comparer(compareBoundedQuantiles)) {
			t.Errorf("decode(encode(_)): when %s got %v, want %v", tc.desc, bqUnmarshalled, bq)
		}
		if bq.state != serialized {
			t.Errorf("BoundedQuantiles should have its state set to Serialized, got %v , want Serialized", bq.state)
		}
	}
}

// Tests that GobEncode() returns errors correctly with different BoundedQuantiles aggregation states.
func TestBQSerializationStateChecks(t *testing.T) {
	for _, tc := range []struct {
		state   aggregationState
		wantErr bool
	}{
		{defaultState, false},
		{merged, true},
		{serialized, true},
		{resultReturned, true},
	} {
		bq := getNoiselessBQ()
		bq.state = tc.state

		if _, err := bq.GobEncode(); (err != nil) != tc.wantErr {
			t.Errorf("GobEncode: when state %v for err got %v, wantErr %t", tc.state, err, tc.wantErr)
		}
	}
}

func compareBoundedQuantiles(bq1, bq2 *BoundedQuantiles) bool {
	return bq1.l0Sensitivity == bq2.l0Sensitivity &&
		bq1.lInfSensitivity == bq2.lInfSensitivity &&
		bq1.lower == bq2.lower &&
		bq1.upper == bq2.upper &&
		bq1.treeHeight == bq2.treeHeight &&
		bq1.branchingFactor == bq2.branchingFactor &&
		bq1.numLeaves == bq2.numLeaves &&
		bq1.leftmostLeafIndex == bq2.leftmostLeafIndex &&
		bq1.Noise == bq2.Noise &&
		bq1.noiseKind == bq2.noiseKind &&
		reflect.DeepEqual(bq1.tree, bq2.tree) &&
		reflect.DeepEqual(bq1.noisedTree, bq2.noisedTree) &&
		bq1.state == bq2.state
}
