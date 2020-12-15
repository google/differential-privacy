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

func TestRoundToMultipleGranularityIsOne(t *testing.T) {
	// Verify that RoundToMultiple returns x if granularity is 1
	for _, x := range []int64{0, 1, -1, 2, -2, 648391, -648391} {
		got := roundToMultiple(x, 1)
		if got != x {
			t.Errorf(
				"roundToMultipleOfPowerOfTwo(%d, 1) = %d, want %d",
				x,
				got,
				x,
			)
		}
	}
}

func TestRoundToMultipleXIsEven(t *testing.T) {
	for _, tc := range []struct {
		x           int64
		granularity int64
		want        int64
	}{
		{
			x:           0,
			granularity: 4,
			want:        0,
		},
		{
			x:           1,
			granularity: 4,
			want:        0,
		},
		{
			x:           2,
			granularity: 4,
			want:        4,
		},
		{
			x:           3,
			granularity: 4,
			want:        4,
		},
		{
			x:           4,
			granularity: 4,
			want:        4,
		},
		{
			x:           -1,
			granularity: 4,
			want:        0,
		},
		{
			x:           -2,
			granularity: 4,
			want:        0,
		},
		{
			x:           -3,
			granularity: 4,
			want:        -4,
		},
		{
			x:           -4,
			granularity: 4,
			want:        -4,
		},
		{
			x:           648389,
			granularity: 4,
			want:        648388,
		},
		{
			x:           648390,
			granularity: 4,
			want:        648392,
		},
		{
			x:           648391,
			granularity: 4,
			want:        648392,
		},
		{
			x:           648392,
			granularity: 4,
			want:        648392,
		},
		{
			x:           -648389,
			granularity: 4,
			want:        -648388,
		},
		{
			x:           -648390,
			granularity: 4,
			want:        -648388,
		},
		{
			x:           -648391,
			granularity: 4,
			want:        -648392,
		},
		{
			x:           -648392,
			granularity: 4,
			want:        -648392,
		},
	} {
		got := roundToMultiple(tc.x, tc.granularity)
		if got != tc.want {
			t.Errorf(
				"roundToMultiple(%d, %d) = %d, want %d",
				tc.x,
				tc.granularity,
				got,
				tc.want,
			)
		}
	}
}

func TestRoundToMultipleXIsOdd(t *testing.T) {
	for _, tc := range []struct {
		x           int64
		granularity int64
		want        int64
	}{
		{
			x:           0,
			granularity: 3,
			want:        0,
		},
		{
			x:           1,
			granularity: 3,
			want:        0,
		},
		{
			x:           2,
			granularity: 3,
			want:        3,
		},
		{
			x:           3,
			granularity: 3,
			want:        3,
		},
		{
			x:           -1,
			granularity: 3,
			want:        0,
		},
		{
			x:           -2,
			granularity: 3,
			want:        -3,
		},
		{
			x:           -3,
			granularity: 3,
			want:        -3,
		},
		{
			x:           648391,
			granularity: 3,
			want:        648390,
		},
		{
			x:           648392,
			granularity: 3,
			want:        648393,
		},
		{
			x:           648393,
			granularity: 3,
			want:        648393,
		},
		{
			x:           -648391,
			granularity: 3,
			want:        -648390,
		},
		{
			x:           -648392,
			granularity: 3,
			want:        -648393,
		},
		{
			x:           -648393,
			granularity: 3,
			want:        -648393,
		},
	} {
		got := roundToMultiple(tc.x, tc.granularity)
		if got != tc.want {
			t.Errorf(
				"roundToMultiple(%d, %d) = %d, want %d",
				tc.x,
				tc.granularity,
				got,
				tc.want,
			)
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

func TestNextLargerFloat64InputRepresentableAsFloat64(t *testing.T) {
	// Verify that nextLargerFloat64 returns a float64 of the same value.
	for _, x := range []int64{
		0,
		1,
		-1,
		// Smallest positive float64 for which next float64 is a distance of 2 away.
		9007199254740992,
		// Largest negative float64 for which previous float64 is a distance of 2 away.
		-9007199254740992,
		// Arbitrary representable number.
		8646911284551352320,
		-8646911284551352320,
		// Largest int64 value accurately representable as a float64.
		math.MaxInt64 - 1023,
		// Smallest int64 value accurately representable as a float64.
		math.MinInt64,
	} {
		got := int64(nextLargerFloat64(x))
		if got != x {
			t.Errorf("nextLargerFloat64(%d) = %d, want %d", x, got, x)
		}
	}
}

func TestNextLargerFloat64InputNotRepresentableAsFloat64(t *testing.T) {
	// Verify that nextLargerFloat64 computes the smallest float64 value that is larger than
	// or equal to the provided int64 value.
	for _, tc := range []struct {
		n    int64
		want int64
	}{
		// Smallest positive int64 value not representable as a float64.
		{n: 9007199254740993, want: 9007199254740994},
		// Largest negative int64 value not representable as a float64.
		{n: -9007199254740993, want: -9007199254740992},
		// Testing non-representable int64 values that lie between arbitrary float64 gap.
		{n: 8646911284551352321, want: 8646911284551353344},
		{n: 8646911284551353343, want: 8646911284551353344},
		{n: -8646911284551352321, want: -8646911284551352320},
		{n: -8646911284551353343, want: -8646911284551352320},
	} {
		got := int64(nextLargerFloat64(tc.n))
		if got != tc.want {
			t.Errorf("nextLargerFloat64(%d) = %d, want %d", tc.n, got, tc.want)
		}
	}

	// Casting math.MaxInt64 to float64 should result in the next larger float64, i.e 2^63.
	// This equality has to be done in float64.
	got := nextLargerFloat64(math.MaxInt64)
	want := float64(1 << 63)
	if got != want {
		t.Errorf("nextLargerFloat64(%d) = %f, want %f", math.MaxInt64, got, want)
	}
}

func TestNextSmallerFloat64InputRepresentableAsFloat64(t *testing.T) {
	// Verify that nextSmallerFloat64 returns a float64 of the same value.
	for _, x := range []int64{
		0,
		1,
		-1,
		// Smallest positive float64 for which next float64 is a distance of 2 away.
		9007199254740992,
		// Largest negative float64 for which previous float64 is a distance of 2 away.
		-9007199254740992,
		// Arbitrary representable number.
		8646911284551352320,
		-8646911284551352320,
		// Largest int64 value accurately representable as a float64.
		math.MaxInt64 - 1023,
		// Smallest int64 value accurately representable as a float64.
		math.MinInt64,
	} {
		got := int64(nextSmallerFloat64(x))
		if got != x {
			t.Errorf("nextSmallerFloat64(%d) = %d, want %d", x, got, x)
		}
	}
}

func TestNextSmallerFloat64InputNotRepresentableAsFloat64(t *testing.T) {
	// Verify that nextSmallerFloat64 computes the largest float64 value that is smaller than
	// or equal to the provided int64 value.
	for _, tc := range []struct {
		n    int64
		want int64
	}{
		// Smallest positive int64 value not representable as a float64.
		{n: 9007199254740993, want: 9007199254740992},
		// Largest negative int64 value not representable as a float64.
		{n: -9007199254740993, want: -9007199254740994},
		// Testing non-representable int64 values that lie between arbitrary float64 gap.
		{n: 8646911284551352321, want: 8646911284551352320},
		{n: 8646911284551353343, want: 8646911284551352320},
		{n: -8646911284551352321, want: -8646911284551353344},
		{n: -8646911284551353343, want: -8646911284551353344},
	} {
		got := int64(nextSmallerFloat64(tc.n))
		if got != tc.want {
			t.Errorf("nextSmallerFloat64(%d) = %d, want %d", tc.n, got, tc.want)
		}
	}
}
