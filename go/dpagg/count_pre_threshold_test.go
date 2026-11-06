//
// Copyright 2026 Yanzhi Xin
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
	"fmt"
	"testing"

	"github.com/google/differential-privacy/go/v4/noise"
)

func TestCountPreThresholdedResultConsumesSuppressedCount(t *testing.T) {
	for _, tc := range []struct {
		name string
		use  func(*Count) error
	}{
		{"result", func(c *Count) error { _, err := c.Result(); return err }},
		{"thresholded_result", func(c *Count) error { _, err := c.ThresholdedResult(0.1); return err }},
		{"pre_thresholded_result", func(c *Count) error { _, err := c.PreThresholdedResult(10, 0.1); return err }},
		{"lower_pre_threshold", func(c *Count) error { _, err := c.PreThresholdedResult(1, 0.1); return err }},
		{"increment", func(c *Count) error { return c.Increment() }},
		{"increment_by", func(c *Count) error { return c.IncrementBy(10) }},
		{"merge_into", func(c *Count) error { return c.Merge(getNoiselessCount(t)) }},
		{"merge_from", func(c *Count) error { return getNoiselessCount(t).Merge(c) }},
		{"serialize", func(c *Count) error { _, err := c.GobEncode(); return err }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := getNoiselessCount(t)
			if err := c.IncrementBy(9); err != nil {
				t.Fatal(err)
			}
			if got, err := c.PreThresholdedResult(10, 0.1); got != nil || err != nil {
				t.Fatalf("PreThresholdedResult(10, 0.1) = (%v, %v), want (nil, nil)", got, err)
			}
			if err := tc.use(c); err == nil {
				t.Error("operation succeeded after PreThresholdedResult suppressed the count")
			}
		})
	}
}

func TestCountPreThresholdedResultRejectsConsumedCount(t *testing.T) {
	for _, tc := range []struct {
		name    string
		consume func(*Count) error
	}{
		{"result", func(c *Count) error { _, err := c.Result(); return err }},
		{"thresholded_result", func(c *Count) error { _, err := c.ThresholdedResult(0.1); return err }},
		{"pre_thresholded_result", func(c *Count) error { _, err := c.PreThresholdedResult(1, 0.1); return err }},
		{"dp_threshold_suppression", func(c *Count) error { _, err := c.PreThresholdedResult(9, 0.1); return err }},
		{"merge", func(c *Count) error { return getNoiselessCount(t).Merge(c) }},
		{"serialize", func(c *Count) error { _, err := c.GobEncode(); return err }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := getNoiselessCount(t)
			if err := c.IncrementBy(9); err != nil {
				t.Fatal(err)
			}
			if err := tc.consume(c); err != nil {
				t.Fatalf("could not consume count: %v", err)
			}
			if got, err := c.PreThresholdedResult(10, 0.1); got != nil || err == nil {
				t.Errorf("PreThresholdedResult after consumption = (%v, %v), want nil and an error", got, err)
			}
		})
	}
}

type countPreThresholdNoise struct {
	noNoise
	addCalls       int
	thresholdCalls int
	intervalCalls  int
	thresholdErr   error
}

func (n *countPreThresholdNoise) AddNoiseInt64(x, _, _ int64, _, _ float64) (int64, error) {
	n.addCalls++
	return x, nil
}

func (n *countPreThresholdNoise) Threshold(_ int64, _, _, _, _ float64) (float64, error) {
	n.thresholdCalls++
	return 5.00001, n.thresholdErr
}

func (n *countPreThresholdNoise) ComputeConfidenceIntervalInt64(x, _, _ int64, _, _, _ float64) (noise.ConfidenceInterval, error) {
	n.intervalCalls++
	return noise.ConfidenceInterval{LowerBound: float64(x), UpperBound: float64(x)}, nil
}

func TestCountPreThresholdedResultConfidenceInterval(t *testing.T) {
	for _, tc := range []struct {
		name         string
		count        int64
		wantNil      bool
		wantInterval bool
	}{
		{"pre_threshold_suppression", 9, true, false},
		{"dp_threshold_suppression", 14, true, true},
		{"released_result", 15, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := getNoiselessCount(t)
			n := &countPreThresholdNoise{}
			c.Noise = n
			if err := c.IncrementBy(tc.count); err != nil {
				t.Fatal(err)
			}
			got, err := c.PreThresholdedResult(10, 0.1)
			if err != nil || (got == nil) != tc.wantNil {
				t.Fatalf("PreThresholdedResult = (%v, %v), wantNil %t", got, err, tc.wantNil)
			}
			ci, err := c.ComputeConfidenceInterval(0.1)
			if (err == nil) != tc.wantInterval {
				t.Fatalf("ComputeConfidenceInterval error = %v, wantInterval %t", err, tc.wantInterval)
			}
			wantCalls := 0
			if tc.wantInterval {
				wantCalls = 1
				if ci.LowerBound != float64(tc.count) || ci.UpperBound != float64(tc.count) {
					t.Errorf("confidence interval = %v, want [%d, %d]", ci, tc.count, tc.count)
				}
			}
			if n.addCalls != wantCalls || n.thresholdCalls != wantCalls || n.intervalCalls != wantCalls {
				t.Errorf("noise calls = (%d, %d, %d), want (%d, %d, %d)", n.addCalls, n.thresholdCalls, n.intervalCalls, wantCalls, wantCalls, wantCalls)
			}
		})
	}
}

func TestCountPreThresholdedResultInvalidArgumentsDoNotConsume(t *testing.T) {
	for _, tc := range []struct {
		name         string
		preThreshold int64
		thresholdErr error
	}{
		{"negative_pre_threshold", -1, nil},
		{"noise_threshold_error", 10, fmt.Errorf("invalid threshold delta")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := getNoiselessCount(t)
			c.Noise = &countPreThresholdNoise{thresholdErr: tc.thresholdErr}
			if err := c.IncrementBy(15); err != nil {
				t.Fatal(err)
			}
			if got, err := c.PreThresholdedResult(tc.preThreshold, 0.1); got != nil || err == nil {
				t.Fatalf("PreThresholdedResult = (%v, %v), want nil and an error", got, err)
			}
			if got, err := c.Result(); got != 15 || err != nil {
				t.Errorf("Result after invalid threshold = (%d, %v), want (15, nil)", got, err)
			}
		})
	}
}

func TestCountPreThresholdedResultDefaultPreThreshold(t *testing.T) {
	for _, preThreshold := range []int64{0, 1} {
		t.Run(fmt.Sprint(preThreshold), func(t *testing.T) {
			c := getNoiselessCount(t)
			if err := c.IncrementBy(6); err != nil {
				t.Fatal(err)
			}
			if got, err := c.PreThresholdedResult(preThreshold, 0.1); err != nil || got == nil || *got != 6 {
				t.Errorf("PreThresholdedResult(%d, 0.1) = (%v, %v), want 6 and nil", preThreshold, got, err)
			}
		})
	}
}
