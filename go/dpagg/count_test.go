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

	"github.com/google/differential-privacy/go/v3/noise"
	"github.com/google/differential-privacy/go/v3/stattestutils"
	"github.com/google/go-cmp/cmp"
)

func TestNewCount(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		opt     *CountOptions
		want    *Count
		wantErr bool
	}{
		{"MaxPartitionsContributed is not set",
			&CountOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Noise:                        noNoise{},
				maxContributionsPerPartition: 2,
			},
			nil,
			true},
		{"maxContributionsPerPartition is not set",
			&CountOptions{
				Epsilon:                  ln3,
				Delta:                    0,
				MaxPartitionsContributed: 1,
				Noise:                    noise.Laplace(),
			},
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				Noise:           noise.Laplace(),
				noiseKind:       noise.LaplaceNoise,
				count:           0,
				state:           defaultState,
			},
			false},
		{"Noise is not set",
			&CountOptions{
				Epsilon:                      ln3,
				Delta:                        0,
				MaxPartitionsContributed:     1,
				maxContributionsPerPartition: 2,
			},
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 2,
				Noise:           noise.Laplace(),
				noiseKind:       noise.LaplaceNoise,
				count:           0,
				state:           defaultState,
			},
			false},
		{"Epsilon is not set",
			&CountOptions{
				Delta:                        0,
				MaxPartitionsContributed:     1,
				maxContributionsPerPartition: 1,
				Noise:                        noise.Laplace(),
			},
			nil,
			true},
		{"Negative Epsilon",
			&CountOptions{
				Epsilon:                      -1,
				Delta:                        0,
				MaxPartitionsContributed:     1,
				maxContributionsPerPartition: 1,
				Noise:                        noise.Laplace(),
			},
			nil,
			true},
		{"Delta is not set with Gaussian noise",
			&CountOptions{
				Epsilon:                      ln3,
				MaxPartitionsContributed:     1,
				maxContributionsPerPartition: 1,
				Noise:                        noise.Gaussian(),
			},
			nil,
			true},
		{"Negative delta",
			&CountOptions{
				Epsilon:                      ln3,
				Delta:                        -1,
				MaxPartitionsContributed:     1,
				maxContributionsPerPartition: 1,
				Noise:                        noise.Laplace(),
			},
			nil,
			true},
	} {
		c, err := NewCount(tc.opt)
		if (err != nil) != tc.wantErr {
			t.Errorf("With %s, got=%v error, wantErr: %t", tc.desc, err, tc.wantErr)
		}
		if !reflect.DeepEqual(c, tc.want) {
			t.Errorf("NewCount: when %s got %+v, want %+v", tc.desc, c, tc.want)
		}
	}
}

func compareCount(c1, c2 *Count) bool {
	return c1.epsilon == c2.epsilon &&
		c1.delta == c2.delta &&
		c1.l0Sensitivity == c2.l0Sensitivity &&
		c1.lInfSensitivity == c2.lInfSensitivity &&
		c1.Noise == c2.Noise &&
		c1.noiseKind == c2.noiseKind &&
		c1.count == c2.count &&
		c1.state == c2.state
}

// Tests that serialization for Count works as expected.
func TestCountSerialization(t *testing.T) {
	for _, tc := range []struct {
		desc string
		opts *CountOptions
	}{
		{"default options", &CountOptions{
			Epsilon:                  ln3,
			Delta:                    0,
			MaxPartitionsContributed: 1,
		}},
		{"non-default options", &CountOptions{
			Epsilon:                  ln3,
			Delta:                    1e-5,
			MaxPartitionsContributed: 5,
			Noise:                    noise.Gaussian(),
		}},
	} {
		c, err := NewCount(tc.opts)
		if err != nil {
			t.Fatalf("Couldn't initialize c: %v", err)
		}
		cUnchanged, err := NewCount(tc.opts)
		if err != nil {
			t.Fatalf("Couldn't initialize cUnchanged: %v", err)
		}
		bytes, err := encode(c)
		if err != nil {
			t.Fatalf("encode(Count) error: %v", err)
		}
		cUnmarshalled := new(Count)
		if err := decode(cUnmarshalled, bytes); err != nil {
			t.Fatalf("decode(Count) error: %v", err)
		}
		// Check that encoding -> decoding is the identity function.
		if !cmp.Equal(cUnchanged, cUnmarshalled, cmp.Comparer(compareCount)) {
			t.Errorf("decode(encode(_)): when %s got %+v, want %+v", tc.desc, cUnmarshalled, cUnchanged)
		}
		if c.state != serialized {
			t.Errorf("Count should have its state set to Serialized, got %v, want Serialized", c.state)
		}
	}
}

// Tests that GobEncode() returns errors correctly with different Count aggregation states.
func TestCountSerializationStateChecks(t *testing.T) {
	for _, tc := range []struct {
		state   aggregationState
		wantErr bool
	}{
		{defaultState, false},
		{merged, true},
		{serialized, false},
		{resultReturned, true},
	} {
		c := getNoiselessCount(t)
		c.state = tc.state

		if _, err := c.GobEncode(); (err != nil) != tc.wantErr {
			t.Errorf("GobEncode: when state %v for err got %v, wantErr %t", tc.state, err, tc.wantErr)
		}
	}
}

func getNoiselessCount(t *testing.T) *Count {
	t.Helper()
	c, err := NewCount(&CountOptions{
		Epsilon:                  ln3,
		Delta:                    tenten,
		MaxPartitionsContributed: 1,
		Noise:                    noNoise{},
	})
	if err != nil {
		t.Fatalf("Couldn't get noiseless count: %v", err)
	}
	return c
}

func TestCountIncrement(t *testing.T) {
	c := getNoiselessCount(t)
	const want = 4
	c.Increment()
	c.Increment()
	c.Increment()
	c.Increment()

	if c.count != want {
		t.Errorf("IncrementBy: after adding %d got %d", want, c.count)
	}
}

func TestCountIncrementBy(t *testing.T) {
	c := getNoiselessCount(t)
	const want = 4
	c.IncrementBy(want)
	if c.count != want {
		t.Errorf("IncrementBy: after adding %d got %d", want, c.count)
	}
}

func TestCountIncrementBy_NegativeValues(t *testing.T) {
	c := getNoiselessCount(t)
	const want = -2
	c.IncrementBy(want)
	if c.count != want {
		t.Errorf("IncrementBy: after adding %d got %d", want, c.count)
	}
}

func TestCountMerge(t *testing.T) {
	c1 := getNoiselessCount(t)
	c2 := getNoiselessCount(t)
	c1.Increment()
	c1.Increment()
	c1.Increment()
	c1.Increment()
	c2.Increment()
	err := c1.Merge(c2)
	if err != nil {
		t.Fatalf("Couldn't merge c1 and c2: %v", err)
	}
	got, err := c1.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}
	const want = 5
	if got != want {
		t.Errorf("Merge: when merging 2 instances of Count got %d, want %d", got, want)
	}
	if c2.state != merged {
		t.Errorf("Merge: when merging 2 instances of Count for c2.state got %v, want Merged", c2.state)
	}
}

// Tests that checkMergeCount() checks the compatibility of two counts for merge correctly.
func TestCountCheckMergeCompatibility(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		opt1    *CountOptions
		opt2    *CountOptions
		wantErr bool
	}{
		{"same options, all fields filled",
			&CountOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Noise:                        noise.Gaussian(),
				maxContributionsPerPartition: 2,
			},
			&CountOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				MaxPartitionsContributed:     1,
				Noise:                        noise.Gaussian(),
				maxContributionsPerPartition: 2,
			},
			false},
		{"same options, only required fields filled",
			&CountOptions{
				Epsilon:                  ln3,
				MaxPartitionsContributed: 1,
			},
			&CountOptions{
				Epsilon:                  ln3,
				MaxPartitionsContributed: 1,
			},
			false},
		{"different epsilon",
			&CountOptions{
				Epsilon:                  ln3,
				MaxPartitionsContributed: 1,
			},
			&CountOptions{
				Epsilon:                  2,
				MaxPartitionsContributed: 1,
			},
			true},
		{"different delta",
			&CountOptions{
				Epsilon:                  ln3,
				Delta:                    tenten,
				Noise:                    noise.Gaussian(),
				MaxPartitionsContributed: 1,
			},
			&CountOptions{
				Epsilon:                  ln3,
				Delta:                    tenfive,
				Noise:                    noise.Gaussian(),
				MaxPartitionsContributed: 1,
			},
			true},
		{"different MaxPartitionsContributed",
			&CountOptions{
				Epsilon:                  ln3,
				MaxPartitionsContributed: 1,
			},
			&CountOptions{
				Epsilon:                  ln3,
				MaxPartitionsContributed: 2,
			},
			true},
		{"different maxContributionsPerPartition",
			&CountOptions{
				Epsilon:                      ln3,
				maxContributionsPerPartition: 2,
				MaxPartitionsContributed:     1,
			},
			&CountOptions{
				Epsilon:                      ln3,
				maxContributionsPerPartition: 5,
				MaxPartitionsContributed:     1,
			},
			true},
		{"different noise",
			&CountOptions{
				Epsilon:                  ln3,
				Delta:                    tenten,
				Noise:                    noise.Gaussian(),
				MaxPartitionsContributed: 1,
			},
			&CountOptions{
				Epsilon:                  ln3,
				Noise:                    noise.Laplace(),
				MaxPartitionsContributed: 1,
			},
			true},
	} {
		c1, err := NewCount(tc.opt1)
		if err != nil {
			t.Fatalf("Couldn't initialize c1: %v", err)
		}
		c2, err := NewCount(tc.opt2)
		if err != nil {
			t.Fatalf("Couldn't initialize c2: %v", err)
		}

		if err := checkMergeCount(c1, c2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: when %s for err got %v, wantErr %t", tc.desc, err, tc.wantErr)
		}
	}
}

// Tests that checkMergeCount() returns errors correctly with different Count aggregation states.
func TestCountCheckMergeStateChecks(t *testing.T) {
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
		c1 := getNoiselessCount(t)
		c2 := getNoiselessCount(t)

		c1.state = tc.state1
		c2.state = tc.state2

		if err := checkMergeCount(c1, c2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: when states [%v, %v] for err got %v, wantErr %t", tc.state1, tc.state2, err, tc.wantErr)
		}
	}
}

func TestCountResultSetsStateCorrectly(t *testing.T) {
	c := getNoiselessCount(t)
	_, err := c.Result()
	if err != nil {
		t.Fatalf("Couldn't compute dp result: %v", err)
	}

	if c.state != resultReturned {
		t.Errorf("Count should have its state set to ResultReturned, got %v, want ResultReturned", c.state)
	}
}

func TestCountThresholdedResult(t *testing.T) {
	// ThresholdedResult outputs the result when it is greater than the threshold (5.00001 using noNoise)
	c1 := getNoiselessCount(t)
	for i := 0; i < 10; i++ {
		c1.Increment()
	}
	got, err := c1.ThresholdedResult(tenten)
	if err != nil {
		t.Fatalf("Couldn't compute thresholded dp result: %v", err)
	}
	if got == nil || *got != 10 {
		t.Errorf("ThresholdedResult(%f): after 10 entries got %v, want 10", tenten, got)
	}

	// ThresholdedResult outputs nil when it is less than the threshold
	c2 := getNoiselessCount(t)
	c2.Increment()
	c2.Increment()
	got, err = c2.ThresholdedResult(tenten)
	if err != nil {
		t.Fatalf("Couldn't compute thresholded dp result: %v", err)
	}
	if got != nil {
		t.Errorf("ThresholdedResult(%f): after 2 entries got %v, want nil", tenten, got)
	}

	// Edge case when noisy result is 5 and threshold is 5.00001, ThresholdedResult outputs nil.
	c3 := getNoiselessCount(t)
	for i := 0; i < 5; i++ {
		c3.Increment()
	}
	got, err = c3.ThresholdedResult(tenten)
	if err != nil {
		t.Fatalf("Couldn't compute thresholded dp result: %v", err)
	}
	if got != nil {
		t.Errorf("ThresholdedResult(%f): after 5 entries got %v, want nil", tenten, got)
	}
}

// Tests that a count smaller than the pre-threshold deterministically returns nil.
func TestCount_CountSmallerThanPreThresholdReturnsNil(t *testing.T) {
	// With a pre-threshold of 10, a count of 9 should not be kept.
	//
	// We use tiny threshold delta, which results in a DP threshold of 1 (after ceiling). A count is
	// kept if it is greater than preThreshold + DPThreshold = 9 + 1 = 10.
	//
	// We use a tiny epsilon, which means the noisy count is greater than 10 with close to 1/2 probability.
	//
	// We run the test 50 times, which means it should fail with probability ~1-2^-50 if
	// pre-thresholding doesn't drop counts smaller than the pre-threshold.
	for i := 0; i < 50; i++ {
		c, err := NewCount(&CountOptions{Epsilon: 1e-5, MaxPartitionsContributed: 1, Noise: noise.Laplace()})
		if err != nil {
			t.Fatalf("Couldn't create Count: %v", err)
		}
		c.IncrementBy(9)
		res, err := c.PreThresholdedResult(10, 1-1e-10)
		if err != nil {
			t.Fatalf("Couldn't compute PreThresholdedResult: %v", err)
		}
		if res != nil {
			t.Errorf("PreThresholdedResult returned a result (%d) for a count smaller than pre-threshold", *res)
		}
	}
}

// Tests that a count greater than the pre-threshold + DP threshold deterministically returns a non-nil result.
func TestCount_CountGreaterThanPreThresholdReturnsResult(t *testing.T) {
	// NoiselessCount has a DP threshold of 5.001. Using a Pre-Threshold of 10 means
	// counts larger than 10 - 1 + 5.001 =14.001 should be deterministically kept.
	c := getNoiselessCount(t)
	c.IncrementBy(15)
	res, err := c.PreThresholdedResult(10, 1e-5)
	if err != nil {
		t.Fatalf("Couldn't compute PreThresholdedResult: %v", err)
	}
	if res == nil {
		t.Errorf("PreThresholdedResult returned nil for a count greater than the pre-threshold")
	}
}

type mockNoiseCount struct {
	t *testing.T
	noise.Noise
}

// AddNoiseInt64 checks that the parameters passed are the ones we expect.
func (mn mockNoiseCount) AddNoiseInt64(x, l0, lInf int64, eps, del float64) (int64, error) {
	if x != 10 && x != 0 { // AddNoiseInt64 is initially called with a placeholder value of 0, so we don't want to fail when that happens
		mn.t.Errorf("AddNoiseInt64: for parameter x got %d, want %d", x, 10)
	}
	if l0 != 3 {
		mn.t.Errorf("AddNoiseInt64: for parameter l0Sensitivity got %d, want %d", l0, 3)
	}
	if lInf != 2 {
		mn.t.Errorf("AddNoiseInt64: for parameter lInfSensitivity got %d, want %d", lInf, 2)
	}
	if !ApproxEqual(eps, ln3) {
		mn.t.Errorf("AddNoiseInt64: for parameter epsilon got %f, want %f", eps, ln3)
	}
	if !ApproxEqual(del, tenten) {
		mn.t.Errorf("AddNoiseInt64: for parameter delta got %f, want %f", del, tenten)
	}
	return 0, nil // ignored
}

// Threshold checks that the parameters passed are the ones we expect.
func (mn mockNoiseCount) Threshold(l0 int64, lInf, eps, del, thresholdDelta float64) (float64, error) {
	if !ApproxEqual(thresholdDelta, 20.0) {
		mn.t.Errorf("Threshold: for parameter thresholdDelta got %f, want %f", thresholdDelta, 10.0)
	}
	if l0 != 3 {
		mn.t.Errorf("Threshold: for parameter l0Sensitivity got %d, want %d", l0, 1)
	}
	if !ApproxEqual(lInf, 2.0) {
		mn.t.Errorf("Threshold: for parameter l0Sensitivity got %f, want %f", lInf, 2.0)
	}
	if !ApproxEqual(eps, ln3) {
		mn.t.Errorf("Threshold: for parameter epsilon got %f, want %f", eps, ln3)
	}
	if !ApproxEqual(del, tenten) {
		mn.t.Errorf("Threshold: for parameter delta got %f, want %f", del, tenten)
	}
	return 0, nil // ignored
}

func getMockCount(t *testing.T) *Count {
	t.Helper()
	c, err := NewCount(&CountOptions{
		Epsilon:                      ln3,
		Delta:                        tenten,
		MaxPartitionsContributed:     3,
		maxContributionsPerPartition: 2,
		Noise:                        mockNoiseCount{t: t},
	})
	if err != nil {
		t.Fatalf("Couldn't get mock count: %v", err)
	}
	return c
}

func TestCountNoiseIsCorrectlyCalled(t *testing.T) {
	count := getMockCount(t)
	for i := 0; i < 10; i++ {
		count.Increment()
	}
	count.Result() // will fail if parameters are wrong
}

func TestThresholdIsCorrectlyCalledForCount(t *testing.T) {
	count := getMockCount(t)
	for i := 0; i < 10; i++ {
		count.Increment()
	}
	count.ThresholdedResult(20) // will fail if parameters are wrong
}

func TestCountEquallyInitialized(t *testing.T) {
	for _, tc := range []struct {
		desc   string
		count1 *Count
		count2 *Count
		equal  bool
	}{
		{
			"equal parameters",
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				state:           defaultState},
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				state:           defaultState},
			true,
		},
		{
			"different epsilon",
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				count:           0,
				state:           defaultState},
			&Count{
				epsilon:         1,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				state:           defaultState},
			false,
		},
		{
			"different delta",
			&Count{
				epsilon:         ln3,
				delta:           0.5,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.GaussianNoise,
				count:           0,
				state:           defaultState},
			&Count{
				epsilon:         ln3,
				delta:           0.6,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.GaussianNoise},
			false,
		},
		{
			"different l0Sensitivity",
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				count:           0,
				state:           defaultState},
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   2,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				state:           defaultState},
			false,
		},
		{
			"different lInfSensitivity",
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				count:           0,
				state:           defaultState},
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 2,
				noiseKind:       noise.LaplaceNoise,
				state:           defaultState},
			false,
		},
		{
			"different noiseKind",
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				count:           0,
				state:           defaultState},
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.GaussianNoise,
				state:           defaultState},
			false,
		},
		{
			"different state",
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise,
				count:           0,
				state:           defaultState,
			},
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.GaussianNoise,
				state:           merged,
			},
			false,
		},
	} {
		if countEquallyInitialized(tc.count1, tc.count2) != tc.equal {
			t.Errorf("countEquallyInitialized: when %s got %t, want %t", tc.desc, !tc.equal, tc.equal)
		}
	}
}

func TestCountIsUnbiased(t *testing.T) {
	const numberOfSamples = 100000
	for _, tc := range []struct {
		opt      *CountOptions
		rawCount int64
		variance float64
	}{
		{
			opt: &CountOptions{
				Epsilon:                  ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 1,
				Noise:                    noise.Gaussian(),
			},
			rawCount: 0,
			variance: 11.9, // approximated via a simulation
		}, {
			opt: &CountOptions{
				Epsilon:                  2.0 * ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 1,
				Noise:                    noise.Gaussian(),
			},
			rawCount: 0,
			variance: 3.5, // approximated via a simulation
		}, {
			opt: &CountOptions{
				Epsilon:                  ln3,
				Delta:                    0.01,
				MaxPartitionsContributed: 1,
				Noise:                    noise.Gaussian(),
			},
			rawCount: 0,
			variance: 3.2, // approximated via a simulation
		}, {
			opt: &CountOptions{
				Epsilon:                  ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 25,
				Noise:                    noise.Gaussian(),
			},
			rawCount: 0,
			variance: 295.0, // approximated via a simulation
		}, {
			opt: &CountOptions{
				Epsilon:                  ln3,
				Delta:                    0.00001,
				MaxPartitionsContributed: 1,
				Noise:                    noise.Gaussian(),
			},
			rawCount: 3380636,
			variance: 11.9, // approximated via a simulation
		}, {
			opt: &CountOptions{
				Epsilon:                  ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 1,
				Noise:                    noise.Laplace(),
			},
			rawCount: 0,
			variance: 1.8, // approximated via a simulation
		}, {
			opt: &CountOptions{
				Epsilon:                  2.0 * ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 1,
				Noise:                    noise.Laplace(),
			},
			rawCount: 0,
			variance: 0.5, // approximated via a simulation
		}, {
			opt: &CountOptions{
				Epsilon:                  ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 25,
				Noise:                    noise.Laplace(),
			},
			rawCount: 0,
			variance: 1035.0, // approximated via a simulation
		}, {
			opt: &CountOptions{
				Epsilon:                  ln3,
				Delta:                    0.0,
				MaxPartitionsContributed: 1,
				Noise:                    noise.Laplace(),
			},
			rawCount: 3380636,
			variance: 1.8, // approximated via a simulation
		},
	} {
		countSamples := make([]float64, numberOfSamples)
		for i := 0; i < numberOfSamples; i++ {
			count, err := NewCount(tc.opt)
			if err != nil {
				t.Fatalf("Couldn't initialize count: %v", err)
			}
			count.IncrementBy(tc.rawCount)
			intSample, err := count.Result()
			if err != nil {
				t.Fatalf("Couldn't compute dp result: %v", err)
			}
			countSamples[i] = float64(intSample)
		}
		sampleMean := stattestutils.SampleMean(countSamples)
		// Assuming that count is unbiased, each sample should have a mean of tc.rawCount
		// and a variance of tc.variance. The resulting sampleMean is approximately Gaussian
		// distributed with the same mean and a variance of tc.variance / numberOfSamples.
		//
		// The tolerance is set to the 99.9995% quantile of the anticipated distribution
		// of sampleMean. Thus, the test falsely rejects with a probability of 10⁻⁵.
		tolerance := 4.41717 * math.Sqrt(tc.variance/float64(numberOfSamples))

		if math.Abs(sampleMean-float64(tc.rawCount)) > tolerance {
			t.Errorf("Got mean = %f, want %f (parameters %+v)", sampleMean, float64(tc.rawCount), tc)
		}
	}
}
