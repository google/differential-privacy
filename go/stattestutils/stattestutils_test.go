// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package stattestutils

import (
	"math"
	"testing"
)

func TestSampleMean(t *testing.T) {
	for _, tc := range []struct {
		input    []float64
		wantMean float64
	}{
		{
			input:    []float64{},
			wantMean: 0,
		},
		{
			input:    []float64{100.123},
			wantMean: 100.123,
		},
		{
			input:    []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			wantMean: 5,
		},
	} {
		output := SampleMean(tc.input)
		if math.Abs(output-tc.wantMean) > 10e-10 {
			t.Errorf("got sampleMean(%v)=%f, want %f", tc.input, output, tc.wantMean)
		}
	}
}

func TestSampleVariance(t *testing.T) {
	for _, tc := range []struct {
		input        []float64
		wantVariance float64
	}{
		{
			input:        []float64{},
			wantVariance: 0,
		},
		{
			input:        []float64{100.123},
			wantVariance: 0,
		},
		{
			input:        []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			wantVariance: 10,
		},
	} {
		output := SampleVariance(tc.input)
		if math.Abs(output-tc.wantVariance) > 10e-10 {
			t.Errorf("got sampleVariance(%v)=%f, want %f", tc.input, output, tc.wantVariance)
		}
	}
}
