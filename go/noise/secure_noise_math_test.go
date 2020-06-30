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

package noise

import (
	"math"
	"testing"
)

func TestCeilPowerOfTwoInputIsNotInDomain(t *testing.T) {
	for _, x := range []float64{
		0.0,
		-1.0,
		math.Inf(-1),
		math.Inf(1),
		math.NaN(),
		math.MaxFloat64,
		math.Pow(2.001, 1023.0),
	} {
		if got := ceilPowerOfTwo(x); !math.IsNaN(got) {
			t.Errorf("ceilPowerOfTwo(%f) = %f, want NaN", x, got)
		}
	}
}

func TestCeilPowerOfTwoInputIsPowerOfTwo(t *testing.T) {
	// Verify that CeilPowerOfTwo returns its input if the input is a power
	// of 2. The test is done exhaustively for all possible exponents of a
	// float64 value.
	for exponent := -1022.0; exponent <= 1023; exponent++ {
		x := math.Pow(2.0, exponent)
		got := ceilPowerOfTwo(x)
		want := x
		if got != want {
			t.Errorf("ceilPowerOfTwo(%f) = %f, want %f", x, got, want)
		}
	}
}

func TestCeilPowerOfTwoInputIsNotPowerOfTwo(t *testing.T) {
	// Verify that CeilPowerOfTwo returns the next power of two for inputs
	// that are different from a power of 2. The test is done exhaustively
	// for all possible exponents of a float64 value.
	for exponent := -1022.0; exponent <= -1.0; exponent++ {
		x := math.Pow(2.001, exponent)
		got := ceilPowerOfTwo(x)
		want := math.Pow(2.0, exponent)
		if got != want {
			t.Errorf("ceilPowerOfTwo(%f) = %f, want %f", x, got, want)
		}
	}
	x := 0.99
	got := ceilPowerOfTwo(x)
	want := 1.0
	if got != want {
		t.Errorf("ceilPowerOfTwo(%f) = %f, want %f", x, got, want)
	}
	for exponent := 1.0; exponent <= 1022.0; exponent++ {
		x := math.Pow(2.001, exponent)
		got := ceilPowerOfTwo(x)
		want := math.Pow(2.0, exponent+1)
		if got != want {
			t.Errorf("ceilPowerOfTwo(%f) = %f, want %f", x, got, want)
		}
	}
}

func TestRoundToMultipleOfPowerOfTwoXIsAMultiple(t *testing.T) {
	// Verify that RoundToMultipleOfPowerOfTwo returns x if x is a
	// multiple of granularity.
	for _, tc := range []struct {
		x           float64
		granularity float64
		want        float64
	}{
		{
			x:           0.0,
			granularity: 0.5,
			want:        0.0,
		},
		{
			x:           0.125,
			granularity: 0.125,
			want:        0.125,
		},
		{
			x:           -0.125,
			granularity: 0.125,
			want:        -0.125,
		},
		{
			x:           16512.0,
			granularity: 1.0,
			want:        16512.0,
		},
		{
			x:           -16512.0,
			granularity: 1.0,
			want:        -16512.0,
		},
		{
			x:           3936.0,
			granularity: 32.0,
			want:        3936.0,
		},
		{
			x:           -3936.0,
			granularity: 32.0,
			want:        -3936.0,
		},
		{
			x:           7.9990234375,
			granularity: 0.0009765625,
			want:        7.9990234375,
		},
		{
			x:           -7.9990234375,
			granularity: 0.0009765625,
			want:        -7.9990234375,
		},
		{
			x:           float64(math.MaxInt64),
			granularity: 0.125,
			want:        float64(math.MaxInt64),
		},
		{
			x:           float64(math.MinInt64),
			granularity: 0.125,
			want:        float64(math.MinInt64),
		},
	} {
		got := roundToMultipleOfPowerOfTwo(tc.x, tc.granularity)
		if got != tc.want {
			t.Errorf(
				"roundToMultipleOfPowerOfTwo(%f, %f) = %f, want %f",
				tc.x,
				tc.granularity,
				got,
				tc.want,
			)
		}
	}
}

func TestRoundToMultipleOfPowerOfTwoXIsNotAMultiple(t *testing.T) {
	// Verify that RoundToMultipleOfPowerOfTwo returns the next closest
	// multiple of granularity if x is not already a multiple.
	for _, tc := range []struct {
		x           float64
		granularity float64
		want        float64
	}{
		{
			x:           0.124,
			granularity: 0.125,
			want:        0.125,
		},
		{
			x:           -0.124,
			granularity: 0.125,
			want:        -0.125,
		},
		{
			x:           0.126,
			granularity: 0.125,
			want:        0.125,
		},
		{
			x:           -0.126,
			granularity: 0.125,
			want:        -0.125,
		},
		{
			x:           16512.499,
			granularity: 1.0,
			want:        16512.0,
		},
		{
			x:           -16512.499,
			granularity: 1.0,
			want:        -16512.0,
		},
		{
			x:           16511.501,
			granularity: 1.0,
			want:        16512.0,
		},
		{
			x:           -16511.501,
			granularity: 1.0,
			want:        -16512.0,
		},
		{
			x:           3920.3257,
			granularity: 32.0,
			want:        3936.0,
		},
		{
			x:           -3920.3257,
			granularity: 32.0,
			want:        -3936.0,
		},
		{
			x:           3951.7654,
			granularity: 32.0,
			want:        3936.0,
		},
		{
			x:           -3951.7654,
			granularity: 32.0,
			want:        -3936.0,
		},
		{
			x:           7.9990232355,
			granularity: 0.0009765625,
			want:        7.9990234375,
		},
		{
			x:           -7.9990232355,
			granularity: 0.0009765625,
			want:        -7.9990234375,
		},
		{
			x:           7.9990514315,
			granularity: 0.0009765625,
			want:        7.9990234375,
		},
		{
			x:           -7.9990514315,
			granularity: 0.0009765625,
			want:        -7.9990234375,
		},
	} {
		got := roundToMultipleOfPowerOfTwo(tc.x, tc.granularity)
		if got != tc.want {
			t.Errorf(
				"roundToMultipleOfPowerOfTwo(%f, %f) = %f, want %f",
				tc.x,
				tc.granularity,
				got,
				tc.want,
			)
		}
	}
}
