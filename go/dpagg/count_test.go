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

func TestNewCount(t *testing.T) {
	for _, tc := range []struct {
		desc string
		opt  *CountOptions
		want *Count
	}{
		{"MaxPartitionsContributed is not set",
			&CountOptions{
				Epsilon:                      ln3,
				Delta:                        tenten,
				Noise:                        noNoise{},
				maxContributionsPerPartition: 2,
			},
			&Count{
				epsilon:         ln3,
				delta:           tenten,
				l0Sensitivity:   1,
				lInfSensitivity: 2,
				noise:           noNoise{},
				count:           0,
				resultReturned:  false,
			}},
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
				noise:           noise.Laplace(),
				noiseKind:       noise.LaplaceNoise,
				count:           0,
				resultReturned:  false,
			}},
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
				noise:           noise.Laplace(),
				noiseKind:       noise.LaplaceNoise,
				count:           0,
				resultReturned:  false,
			}},
	} {
		got := NewCount(tc.opt)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("NewCount: when %s got %v, want %v", tc.desc, got, tc.want)
		}
	}
}

func compareCount(c1, c2 *Count) bool {
	return c1.epsilon == c2.epsilon &&
		c1.delta == c2.delta &&
		c1.l0Sensitivity == c2.l0Sensitivity &&
		c1.lInfSensitivity == c2.lInfSensitivity &&
		c1.noise == c2.noise &&
		c1.noiseKind == c2.noiseKind &&
		c1.count == c2.count &&
		c1.resultReturned == c2.resultReturned
}

// Tests that serialization for Count works as expected.
func TestCountSerialization(t *testing.T) {
	for _, tc := range []struct {
		desc string
		opts *CountOptions
	}{
		{"default options", &CountOptions{
			Epsilon: ln3,
			Delta:   0,
		}},
		{"non-default options", &CountOptions{
			Epsilon:                  ln3,
			Delta:                    1e-5,
			MaxPartitionsContributed: 5,
			Noise:                    noise.Gaussian(),
		}},
	} {
		c, cUnchanged := NewCount(tc.opts), NewCount(tc.opts)
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
			t.Errorf("decode(encode(_)): when %s got %v, want %v", tc.desc, cUnmarshalled, c)
		}
		// Check that the original Count has its resultReturned set to true after serialization.
		if !c.resultReturned {
			t.Errorf("Count %v should have its resultReturned set to true after being serialized", c)
		}
	}
}

func getNoiselessCount() *Count {
	return NewCount(&CountOptions{
		Epsilon:                  ln3,
		Delta:                    tenten,
		MaxPartitionsContributed: 1,
		Noise:                    noNoise{},
	})
}

func TestCountIncrement(t *testing.T) {
	count := getNoiselessCount()
	count.Increment()
	count.Increment()
	count.Increment()
	count.Increment()
	got := count.Result()
	const want = 4
	if got != want {
		t.Errorf("Increment: after adding %d values got %d, want %d", want, got, want)
	}
}

func TestCountIncrementBy(t *testing.T) {
	count := getNoiselessCount()
	count.IncrementBy(4)
	got := count.Result()
	const want = 4
	if got != want {
		t.Errorf("IncrementBy: after adding %d got %d, want %d", want, got, want)
	}
}

func TestCountMerge(t *testing.T) {
	c1 := getNoiselessCount()
	c2 := getNoiselessCount()
	c1.Increment()
	c1.Increment()
	c1.Increment()
	c1.Increment()
	c2.Increment()
	c1.Merge(c2)
	got := c1.Result()
	const want = 5
	if got != want {
		t.Errorf("Merge: when merging 2 instances of Count got %d, want %d", got, want)
	}
	if !c2.resultReturned {
		t.Errorf("Merge: when merging 2 instances of Count for c2.resultReturned got false, want true")
	}
}

func TestCountCheckMerge(t *testing.T) {
	for _, tc := range []struct {
		desc          string
		opt1          *CountOptions
		opt2          *CountOptions
		returnResult1 bool
		returnResult2 bool
		wantErr       bool
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
			false,
			false,
			false},
		{"same options, only required fields filled",
			&CountOptions{
				Epsilon: ln3,
			},
			&CountOptions{
				Epsilon: ln3,
			},
			false,
			false,
			false},
		{"same options, first result returned",
			&CountOptions{
				Epsilon: ln3,
			},
			&CountOptions{
				Epsilon: ln3,
			},
			true,
			false,
			true},
		{"same options, second result returned",
			&CountOptions{
				Epsilon: ln3,
			},
			&CountOptions{
				Epsilon: ln3,
			},
			false,
			true,
			true},
		{"different epsilon",
			&CountOptions{
				Epsilon: ln3,
			},
			&CountOptions{
				Epsilon: 2,
			},
			false,
			false,
			true},
		{"different delta",
			&CountOptions{
				Epsilon: ln3,
				Delta:   tenten,
				Noise:   noise.Gaussian(),
			},
			&CountOptions{
				Epsilon: ln3,
				Delta:   tenfive,
				Noise:   noise.Gaussian(),
			},
			false,
			false,
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
			false,
			false,
			true},
		{"different maxContributionsPerPartition",
			&CountOptions{
				Epsilon:                      ln3,
				maxContributionsPerPartition: 2,
			},
			&CountOptions{
				Epsilon:                      ln3,
				maxContributionsPerPartition: 5,
			},
			false,
			false,
			true},
		{"different noise",
			&CountOptions{
				Epsilon: ln3,
				Delta:   tenten,
				Noise:   noise.Gaussian(),
			},
			&CountOptions{
				Epsilon: ln3,
				Noise:   noise.Laplace(),
			},
			false,
			false,
			true},
	} {
		c1 := NewCount(tc.opt1)
		c2 := NewCount(tc.opt2)

		if tc.returnResult1 {
			c1.Result()
		}
		if tc.returnResult2 {
			c2.Result()
		}

		if err := checkMergeCount(c1, c2); (err != nil) != tc.wantErr {
			t.Errorf("CheckMerge: when %v for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCountThresholdedResult(t *testing.T) {
	// ThresholdedResult outputs the result when it is greater than the threshold (5 using noNoise)
	c1 := getNoiselessCount()
	for i := 0; i < 10; i++ {
		c1.Increment()
	}
	got := c1.ThresholdedResult(tenten)
	if got == nil || *got != 10 {
		t.Errorf("ThresholdedResult(%f): when 10 addings got %v, want 10", tenten, got)
	}

	// ThresholdedResult outputs nil when it is less than the threshold
	c2 := getNoiselessCount()
	c2.Increment()
	c2.Increment()
	got = c2.ThresholdedResult(tenten)
	if got != nil {
		t.Errorf("ThresholdedResult(%f): when 2 addings got %v, want nil", tenten, got)
	}
}

type mockNoiseCount struct {
	t *testing.T
	noise.Noise
}

// AddNoiseInt64 checks that the parameters passed are the ones we expect.
func (mn mockNoiseCount) AddNoiseInt64(x, l0, lInf int64, eps, del float64) int64 {
	if x != 10 && x != 0 { // AddNoiseInt64 is initially called with a dummy value of 0, so we don't want to fail when that happens
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
	return 0 // ignored
}

// Threshold checks that the parameters passed are the ones we expect.
func (mn mockNoiseCount) Threshold(l0 int64, lInf, eps, del, thresholdDelta float64) float64 {
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
	return 0 // ignored
}

func getMockCount(t *testing.T) *Count {
	return NewCount(&CountOptions{
		Epsilon:                      ln3,
		Delta:                        tenten,
		MaxPartitionsContributed:     3,
		maxContributionsPerPartition: 2,
		Noise:                        mockNoiseCount{t: t},
	})
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
			},
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise},
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
				resultReturned:  false},
			&Count{
				epsilon:         1,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise},
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
				resultReturned:  false},
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
				resultReturned:  false},
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   2,
				lInfSensitivity: 1,
				noiseKind:       noise.LaplaceNoise},
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
				resultReturned:  false},
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 2,
				noiseKind:       noise.LaplaceNoise},
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
				resultReturned:  false},
			&Count{
				epsilon:         ln3,
				delta:           0,
				l0Sensitivity:   1,
				lInfSensitivity: 1,
				noiseKind:       noise.GaussianNoise},
			false,
		},
	} {
		if countEquallyInitialized(tc.count1, tc.count2) != tc.equal {
			t.Errorf("countEquallyInitialized: when %v got %t, want %t", tc.desc, !tc.equal, tc.equal)
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
		countSamples := make(stat.IntSlice, numberOfSamples)
		for i := 0; i < numberOfSamples; i++ {
			count := NewCount(tc.opt)
			count.IncrementBy(tc.rawCount)
			countSamples[i] = count.Result()
		}
		sampleMean := stat.Mean(countSamples)
		// Assuming that count is unbiased, each sample should have a mean of tc.rawCount
		// and a variance of tc.variance. The resulting sampleMean is approximately Gaussian
		// distributed with the same mean and a variance of tc.variance / numberOfSamples.
		//
		// The tolerance is set to the 99.9995% quantile of the anticipated distribution
		// of sampleMean. Thus, the test falsely rejects with a probability of 10⁻⁵.
		tolerance := 4.41717 * math.Sqrt(tc.variance/float64(numberOfSamples))

		if math.Abs(sampleMean-float64(tc.rawCount)) > tolerance {
			t.Errorf("got mean = %f, want %f (parameters %+v)", sampleMean, float64(tc.rawCount), tc)
		}
	}
}
