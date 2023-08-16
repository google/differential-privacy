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

	"github.com/google/go-cmp/cmp"
)

func TestNewPreAggSelectPartition(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		opt     *PreAggSelectPartitionOptions
		want    *PreAggSelectPartition
		wantErr bool
	}{
		{"MaxPartitionsContributed is not set",
			&PreAggSelectPartitionOptions{
				Epsilon: ln3,
				Delta:   tenten,
			},
			nil,
			true},
		{"Epsilon is not set",
			&PreAggSelectPartitionOptions{
				Delta:                    tenten,
				MaxPartitionsContributed: 1,
			},
			nil,
			true},
		{"Negative Epsilon",
			&PreAggSelectPartitionOptions{
				Epsilon:                  -1,
				Delta:                    tenten,
				MaxPartitionsContributed: 1,
			},
			nil,
			true},
		{"Delta is not set",
			&PreAggSelectPartitionOptions{
				Epsilon:                  ln3,
				MaxPartitionsContributed: 1,
			},
			nil,
			true},
		{"Negative delta",
			&PreAggSelectPartitionOptions{
				Epsilon:                  ln3,
				Delta:                    -1,
				MaxPartitionsContributed: 1,
			},
			nil,
			true},
		{"Negative preThreshold",
			&PreAggSelectPartitionOptions{
				Epsilon:                  ln3,
				Delta:                    tenten,
				MaxPartitionsContributed: 1,
				PreThreshold:             -1,
			},
			nil,
			true},
		{"Valid 0 preThreshold",
			&PreAggSelectPartitionOptions{
				Epsilon:                  ln3,
				Delta:                    tenten,
				PreThreshold:             0,
				MaxPartitionsContributed: 1,
			},
			&PreAggSelectPartition{
				epsilon:       1.0986122886681096,
				delta:         1e-10,
				preThreshold:  1, // PreThreshold defaults to 1.
				l0Sensitivity: 1,
				idCount:       0,
				state:         defaultState,
			},
			false},
		{"Valid unset preThreshold",
			&PreAggSelectPartitionOptions{
				Epsilon:                  ln3,
				Delta:                    tenten,
				PreThreshold:             1,
				MaxPartitionsContributed: 1,
			},
			&PreAggSelectPartition{
				epsilon:       1.0986122886681096,
				delta:         1e-10,
				preThreshold:  1, // PreThreshold defaults to 1.
				l0Sensitivity: 1,
				idCount:       0,
				state:         defaultState},
			false},
	} {
		c, err := NewPreAggSelectPartition(tc.opt)
		if (err != nil) != tc.wantErr {
			t.Errorf("With %s, got=%v error, wantErr: %t", tc.desc, err, tc.wantErr)
		}
		if !reflect.DeepEqual(c, tc.want) {
			t.Errorf("NewPreAggSelectPartition: when %s got %+v, want %+v", tc.desc, c, tc.want)
		}
	}
}

func approxEqual(a, b float64) bool {
	maxMagnitude := math.Max(math.Abs(a), math.Abs(b))
	if math.IsInf(maxMagnitude, +1) {
		return a == b
	}
	return math.Abs(a-b) <= 1e-6*maxMagnitude
}

func comparePreAggSelectPartitionSelection(s1, s2 *PreAggSelectPartition) bool {
	return s1.epsilon == s2.epsilon &&
		s1.delta == s2.delta &&
		s1.l0Sensitivity == s2.l0Sensitivity &&
		s1.idCount == s2.idCount &&
		s1.state == s2.state &&
		s1.preThreshold == s2.preThreshold
}

// Tests that serialization for PreAggSelectPartition works as expected.
func TestPreAggSelectPartitionSerialization(t *testing.T) {
	for _, tc := range []struct {
		desc string
		opts *PreAggSelectPartitionOptions
	}{
		{"default options", &PreAggSelectPartitionOptions{
			Epsilon:                  ln3,
			Delta:                    1e-5,
			MaxPartitionsContributed: 1,
		}},
		{"non-default options", &PreAggSelectPartitionOptions{
			Epsilon:                  ln3,
			Delta:                    1e-5,
			MaxPartitionsContributed: 5,
			PreThreshold:             10,
		}},
	} {
		s, err := NewPreAggSelectPartition(tc.opts)
		if err != nil {
			t.Fatalf("Couldn't initialize s: %v", err)
		}
		sUnchanged, err := NewPreAggSelectPartition(tc.opts)
		if err != nil {
			t.Fatalf("Couldn't initialize sUnchanged: %v", err)
		}
		bytes, err := encode(s)
		if err != nil {
			t.Fatalf("encode failed: %v", err)
		}
		sUnmarshalled := new(PreAggSelectPartition)
		if err := decode(sUnmarshalled, bytes); err != nil {
			t.Fatalf("decode failed: %v", err)
		}
		// Check that encoding -> decoding is the identity function.
		if diff := cmp.Diff(sUnchanged, sUnmarshalled, cmp.Comparer(comparePreAggSelectPartitionSelection)); diff != "" {
			t.Errorf("With %s, aggregation changed after encode()->decode(). Diff: %s", tc.desc, diff)
		}
		// Check that the original PreAggSelectPartition has its state set to serialized after serialization.
		if s.state != serialized {
			t.Errorf("PreAggSelectPartition %v should have its state set to serialized after being serialized", s)
		}
	}
}

// Tests that GobEncode() returns errors correctly with different PreAggSelectPartition aggregation states.
func TestPreAggSelectPartitionSerializationStateChecks(t *testing.T) {
	for _, tc := range []struct {
		state   aggregationState
		wantErr bool
	}{
		{defaultState, false},
		{merged, true},
		{serialized, false},
		{resultReturned, true},
	} {
		options := &PreAggSelectPartitionOptions{Epsilon: 1, Delta: 0.1, MaxPartitionsContributed: 1, PreThreshold: 2}
		s, err := NewPreAggSelectPartition(options)
		if err != nil {
			t.Fatalf("Couldn't initialize s: %v", err)
		}
		s.state = tc.state

		if _, err := s.GobEncode(); (err != nil) != tc.wantErr {
			t.Errorf("GobEncode: when state %v for err got %v, wantErr %t", tc.state, err, tc.wantErr)
		}
	}
}

func TestPreAggSelectPartitionIncrementBy(t *testing.T) {
	s := getTestPreAggSelectPartition(t)
	const want = 10
	s.IncrementBy(want)
	if s.idCount != want {
		t.Errorf("IncrementBy: after incrementing by %d, got %d", want, s.idCount)
	}
}

func TestPreAggSelectPartitionIncrementBy_NegativeValues(t *testing.T) {
	s := getTestPreAggSelectPartition(t)
	const want = -2
	s.IncrementBy(want)
	if s.idCount != want {
		t.Errorf("IncrementBy: after incrementing by %d, got %d", want, s.idCount)
	}
}

func TestPreAggSelectPartitionKeepPartitionProbability(t *testing.T) {
	for _, tc := range []struct {
		name          string
		l0Sensitivity int64
		epsilon       float64
		delta         float64
		// want maps idCount to the desired keep partition probability.
		want map[int64]float64
	}{
		{
			name:          "ε != 0",
			l0Sensitivity: 1,
			epsilon:       math.Log(2),
			delta:         0.1,
			want: map[int64]float64{
				0:             0,
				1:             0.1,
				2:             0.3,
				3:             0.7,
				4:             0.9,
				5:             1,
				6:             1,
				math.MaxInt64: 1,
			},
		},
		{
			name:          "ε != 0 with non-trivial l0Sensitivity",
			l0Sensitivity: 2,
			epsilon:       2 * math.Log(2),
			delta:         0.2,
			want: map[int64]float64{
				0:             0,
				1:             0.1,
				2:             0.3,
				3:             0.7,
				4:             0.9,
				5:             1,
				6:             1,
				math.MaxInt64: 1,
			},
		},
		{
			name:          "ε != 0, values chosen so that floor in nCr formula matters",
			l0Sensitivity: 1,
			epsilon:       math.Log(1.5),
			delta:         0.1,
			want: map[int64]float64{
				0:             0,
				1:             0.1,
				2:             0.25,
				3:             0.475,
				4:             0.716666666667,
				5:             0.877777777778,
				6:             0.985185185185,
				7:             1,
				8:             1,
				math.MaxInt64: 1,
			},
		},
		{
			name:          "Precision test: High ε value, low δ value.",
			l0Sensitivity: 1,
			epsilon:       50,
			delta:         1e-200,
			want: map[int64]float64{
				0:             0,
				1:             1e-200,
				2:             5.184706e-179,
				3:             2.688117e-157,
				4:             1.393710e-135,
				5:             7.225974e-114,
				6:             3.746455e-92,
				7:             1.942426e-70,
				8:             1.007091e-48,
				9:             5.221470e-27,
				10:            2.707178e-05,
				11:            1,
				math.MaxInt64: 1,
			},
		},
		{
			name:          "Maximal ε handling.",
			l0Sensitivity: 1,
			epsilon:       math.MaxFloat64,
			delta:         1e-200,
			want: map[int64]float64{
				0:             0,
				1:             1e-200,
				2:             1,
				math.MaxInt64: 1,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for privacyIDCount, wantProbability := range tc.want {
				gotProbability, err := keepPartitionProbability(privacyIDCount, tc.l0Sensitivity, tc.epsilon, tc.delta)
				if err != nil {
					t.Fatalf("Couldn't compute keepPartitionProbability: %v", err)
				}
				if !approxEqual(gotProbability, wantProbability) {
					t.Errorf("keepPartitionProbability(%d, %d, %e, %e) = %e, want: %e",
						privacyIDCount, tc.l0Sensitivity, tc.epsilon, tc.delta, gotProbability, wantProbability)
				}
			}
		})
	}
}

func TestSumExpPowers(t *testing.T) {
	for _, tc := range []struct {
		name      string
		epsilon   float64
		minPower  int64
		numPowers int64
		want      float64
	}{
		{
			name:      "One value",
			epsilon:   1,
			minPower:  1,
			numPowers: 1,
			want:      2.718281828459045,
		},
		{
			name:      "Three values",
			epsilon:   1,
			minPower:  0,
			numPowers: 3,
			want:      11.107337927389695,
		},
		{
			name:      "Negative minPower",
			epsilon:   1,
			minPower:  -2,
			numPowers: 2,
			want:      0.5032147244080551,
		},
		{
			name:      "Non-integer epsilon",
			epsilon:   math.Log(3),
			minPower:  0,
			numPowers: 3,
			want:      13,
		},
		{
			name:      "exp(-epsilon) = 0",
			epsilon:   1e-100,
			minPower:  0,
			numPowers: 100,
			want:      100,
		},
		{
			name:      "large epsilon, positive powers",
			epsilon:   math.MaxFloat64,
			minPower:  1,
			numPowers: 5,
			want:      math.Inf(1),
		},
		{
			name:      "large epsilon, negative powers",
			epsilon:   math.MaxFloat64,
			minPower:  -5,
			numPowers: 3,
			want:      0,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := sumExpPowers(tc.epsilon, tc.minPower, tc.numPowers)
			if err != nil {
				t.Fatalf("Couldn't compute sumExpPowers: %v", err)
			}
			if !approxEqual(got, tc.want) {
				t.Errorf("sumExpPowers(%g, %d, %d) = %g, want %g",
					tc.epsilon, tc.minPower, tc.numPowers, got, tc.want)
			}
		})
	}
}

func TestPreAggSelectPartition(t *testing.T) {
	for _, tc := range []struct {
		name                string
		privacyIDCount      int64
		opts                *PreAggSelectPartitionOptions
		wantSelectionRate   float64
		numTrials           int
		tolerance           float64
		retriesForFlakiness int
	}{
		{
			name:           "Should never select partition",
			privacyIDCount: 0,
			opts: &PreAggSelectPartitionOptions{
				Epsilon:                  math.Log(2),
				Delta:                    0.3,
				MaxPartitionsContributed: 1,
			},
			wantSelectionRate: 0,
			numTrials:         1_000,
			tolerance:         0,
		},
		{
			// This test is non-deterministic. The binomial distribution with
			// parameters (100,000, 0.3) yields a value in the interval (29017, 30989)
			// with probability at least 1 - 1e-11. To verify, run the following
			// script in python3:
			//     from scipy.stats import binom
			//     binom.interval(1-1e-11, 100_000, 0.3)
			// Dividing the interval endpoints by 100,000, we see that the average is
			// within 0.3 +/- 0.01 with high probability. Running this test has a
			// 1e-11 flakiness rate, so we retry up to 2 times upon failure to drive
			// the flakiness rate down to the a truly negligeable math.Pow(1e-11, 3) =
			// 1e-33 flakiness rate.
			name:           "Should sometimes select partition",
			privacyIDCount: 1,
			opts: &PreAggSelectPartitionOptions{
				Epsilon:                  math.Log(2),
				Delta:                    0.3,
				MaxPartitionsContributed: 1,
			},
			wantSelectionRate:   0.3,
			numTrials:           100_000,
			tolerance:           0.01,
			retriesForFlakiness: 2,
		},
		{
			// This test is non-deterministic. The binomial distribution with
			// parameters (100,000, 0.3) yields a value in the interval (29017, 30989)
			// with probability at least 1 - 1e-11. To verify, run the following
			// script in python3:
			//     from scipy.stats import binom
			//     binom.interval(1-1e-11, 100_000, 0.3)
			// Dividing the interval endpoints by 100,000, we see that the average is
			// within 0.3 +/- 0.01 with high probability. Running this test has a
			// 1e-11 flakiness rate, so we retry up to 2 times upon failure to drive
			// the flakiness rate down to the a truly negligeable math.Pow(1e-11, 3) =
			// 1e-33 flakiness rate.
			name:           "Should sometimes select partition database-level privacy",
			privacyIDCount: 1,
			opts: &PreAggSelectPartitionOptions{
				Epsilon:                  math.Log(2),
				Delta:                    0.6,
				MaxPartitionsContributed: 2,
			},
			wantSelectionRate:   0.3,
			numTrials:           100_000,
			tolerance:           0.01,
			retriesForFlakiness: 2,
		},
		{
			name:           "Should always select partition",
			privacyIDCount: 4,
			opts: &PreAggSelectPartitionOptions{
				Epsilon:                  math.Log(2),
				Delta:                    0.3,
				MaxPartitionsContributed: 1,
			},
			wantSelectionRate: 1,
			numTrials:         1_000,
			tolerance:         0,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for testAttempt := 0; testAttempt <= tc.retriesForFlakiness; testAttempt++ {
				var selections int
				for trial := 0; trial < tc.numTrials; trial++ {
					s, err := NewPreAggSelectPartition(tc.opts)
					if err != nil {
						t.Fatalf("Couldn't initialize s: %v", err)
					}
					s.IncrementBy(tc.privacyIDCount)
					should, err := s.ShouldKeepPartition()
					if err != nil {
						t.Fatalf("Couldn't compute ShouldKeepPartition: %v", err)
					}
					if should {
						selections++
					}
				}
				gotSelectionRate := float64(selections) / float64(tc.numTrials)
				if math.Abs(tc.wantSelectionRate-gotSelectionRate) <= tc.tolerance {
					return
				}
				if testAttempt == tc.retriesForFlakiness {
					t.Errorf("Failed on attempt %d: wantSelectionRate: %v, gotSelectionRate: %v", testAttempt, tc.wantSelectionRate, gotSelectionRate)
				} else {
					t.Logf("Failed on attempt %d: wantSelectionRate: %v, gotSelectionRate: %v", testAttempt, tc.wantSelectionRate, gotSelectionRate)
				}
			}
		})
	}
}

// Tests that an idCount smaller than the prethreshold deterministically returns false.
func TestPreAggSelectPartition_CountSmallerThanPreThresholdReturnsFalse(t *testing.T) {
	for _, tc := range []struct {
		name                     string
		maxPartitionsContributed int64 // > 3 leads to gaussian thresholding being used
	}{
		{"magic_partition_selection", 1},
		{"gaussian_thresholding", 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := getTestPreAggSelectPartition(t)
			s.preThreshold = 10
			s.l0Sensitivity = tc.maxPartitionsContributed
			s.IncrementBy(9)
			should, err := s.ShouldKeepPartition()
			if err != nil {
				t.Fatalf("Couldn't compute ShouldKeepPartition: %v", err)
			}
			if should {
				t.Errorf("ShouldKeepPartition returned true for a count smaller than the pre-threshold")
			}
		})
	}
}

// Tests that an idCount greater than the pre-threshold deterministically returns true for a large enough delta & epsilon.
func TestPreAggSelectPartition_CountGreaterThanPreThresholdReturnsTrue(t *testing.T) {
	for _, tc := range []struct {
		name                     string
		maxPartitionsContributed int64 // > 3 leads to gaussian thresholding being used
	}{
		{"magic_partition_selection", 1},
		{"gaussian_thresholding", 4},
	} {
		s := getTestPreAggSelectPartition(t)
		s.preThreshold = 10
		s.l0Sensitivity = tc.maxPartitionsContributed
		s.IncrementBy(11)
		should, err := s.ShouldKeepPartition()
		if err != nil {
			t.Fatalf("Couldn't compute ShouldKeepPartition: %v", err)
		}
		if !should {
			t.Errorf("ShouldKeepPartition returned false for a count larger than the pre-threshold")
		}
	}
}

func TestMergePreAggSelectPartition(t *testing.T) {
	wantFinalS1 := &PreAggSelectPartition{
		epsilon:       0.1,
		delta:         0.2,
		preThreshold:  2,
		l0Sensitivity: 1,
		idCount:       8,
		state:         defaultState}
	options1 := &PreAggSelectPartitionOptions{Epsilon: 0.1, Delta: 0.2, MaxPartitionsContributed: 1, PreThreshold: 2}
	s1, err := NewPreAggSelectPartition(options1)
	if err != nil {
		t.Fatalf("Couldn't initialize s1: %v", err)
	}
	options2 := &PreAggSelectPartitionOptions{Epsilon: 0.1, Delta: 0.2, MaxPartitionsContributed: 1, PreThreshold: 2}
	s2, err := NewPreAggSelectPartition(options2)
	if err != nil {
		t.Fatalf("Couldn't initialize s2: %v", err)
	}
	s1.IncrementBy(5)
	s2.IncrementBy(3)
	err = s1.Merge(s2)
	if err != nil {
		t.Fatalf("Couldn't merge s1 and s2: %v", err)
	}

	if !reflect.DeepEqual(wantFinalS1, s1) {
		t.Errorf("s1: want %v, got %v", wantFinalS1, s1)
	}
	if s2.state != merged {
		t.Errorf("want s2.state = merged, got %v", s2.state)
	}
}

func TestCheckMergePreAggSelectPartition(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		s1      *PreAggSelectPartition
		s2      *PreAggSelectPartition
		wantErr bool
	}{
		{
			desc:    "Compatible PreAggSelectPartitions",
			s1:      &PreAggSelectPartition{epsilon: 0.1, delta: 0.2, l0Sensitivity: 1, idCount: 1},
			s2:      &PreAggSelectPartition{epsilon: 0.1, delta: 0.2, l0Sensitivity: 1, idCount: 2},
			wantErr: false,
		},
		{
			desc:    "Parameter disagreement: ε",
			s1:      &PreAggSelectPartition{epsilon: 0.2, delta: 0.2, l0Sensitivity: 1, idCount: 1},
			s2:      &PreAggSelectPartition{epsilon: 0.1, delta: 0.2, l0Sensitivity: 1, idCount: 2},
			wantErr: true,
		},
		{
			desc:    "Parameter disagreement: δ",
			s1:      &PreAggSelectPartition{epsilon: 0.1, delta: 0.3, l0Sensitivity: 1},
			s2:      &PreAggSelectPartition{epsilon: 0.1, delta: 0.2, l0Sensitivity: 1},
			wantErr: true,
		},
		{
			desc:    "Parameter disagreement: l0Sensitivity",
			s1:      &PreAggSelectPartition{epsilon: 0.1, delta: 0.2, l0Sensitivity: 1, idCount: 1},
			s2:      &PreAggSelectPartition{epsilon: 0.1, delta: 0.2, l0Sensitivity: 2, idCount: 2},
			wantErr: true,
		},
		{
			desc:    "Parameter disagreement: preThreshold",
			s1:      &PreAggSelectPartition{epsilon: 0.1, delta: 0.2, l0Sensitivity: 1, preThreshold: 1},
			s2:      &PreAggSelectPartition{epsilon: 0.1, delta: 0.2, l0Sensitivity: 1, preThreshold: 2},
			wantErr: true,
		},
	} {
		if err := checkMergePreAggSelectPartition(tc.s1, tc.s2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: when %s for err got %v, wantErr %t", tc.desc, err, tc.wantErr)
		}
	}
}

// Tests that checkMergePreAggSelectPartition() returns errors correctly with different PreAggSelectPartition aggregation states.
func TestPreAggSelectPartitionCheckMergeStateChecks(t *testing.T) {
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
		options1 := &PreAggSelectPartitionOptions{Epsilon: 1, Delta: 0.1, MaxPartitionsContributed: 1, PreThreshold: 2}
		s1, err := NewPreAggSelectPartition(options1)
		if err != nil {
			t.Fatalf("Couldn't initialize s1: %v", err)
		}
		options2 := &PreAggSelectPartitionOptions{Epsilon: 1, Delta: 0.1, MaxPartitionsContributed: 1, PreThreshold: 2}
		s2, err := NewPreAggSelectPartition(options2)
		if err != nil {
			t.Fatalf("Couldn't initialize s2: %v", err)
		}

		s1.state = tc.state1
		s2.state = tc.state2

		if err := checkMergePreAggSelectPartition(s1, s2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: when states [%v, %v] for err got %v, wantErr %t", tc.state1, tc.state2, err, tc.wantErr)
		}
	}
}

func TestPreAggSelectPartitionResultSetsStateCorrectly(t *testing.T) {
	options := &PreAggSelectPartitionOptions{Epsilon: 1, Delta: 0.1, MaxPartitionsContributed: 1, PreThreshold: 2}
	s, err := NewPreAggSelectPartition(options)
	if err != nil {
		t.Fatalf("Couldn't initialize s: %v", err)
	}
	_, err = s.ShouldKeepPartition()
	if err != nil {
		t.Fatalf("Couldn't compute ShouldKeepPartition: %v", err)
	}

	if s.state != resultReturned {
		t.Errorf("PreAggSelectPartition should have its state set to ResultReturned, got %v, want ResultReturned", s.state)
	}
}

func getTestPreAggSelectPartition(t *testing.T) *PreAggSelectPartition {
	t.Helper()
	s, err := NewPreAggSelectPartition(&PreAggSelectPartitionOptions{
		Epsilon:                  100,
		Delta:                    1 - 1e-5,
		MaxPartitionsContributed: 1,
	})
	if err != nil {
		t.Fatalf("Couldn't initialize s: %v", err)
	}
	return s
}
