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

package checks

import (
	"math"
	"testing"
)

func TestVerifyName(t *testing.T) {
	tests := []struct {
		defaultName string
		nameSlice   []string
		want        string
		wantErr     bool
	}{
		{"Epsilon", []string{}, "Epsilon", false},
		{"Epsilon", []string{"AggregationEpsilon"}, "AggregationEpsilon", false},
		{"Epsilon", []string{"AggregationEpsilon", "PartitionSelectionEpsilon"}, "", true},
	}
	for _, tc := range tests {
		got, err := verifyName(tc.defaultName, tc.nameSlice)
		if (err != nil) != tc.wantErr {
			t.Errorf("verifyName(%v, %v) returned an unexpected error: %v", tc.defaultName, tc.nameSlice, err)
		}
		if got != tc.want {
			t.Errorf("verifyName(%v, %v) = %v, want: %v", tc.defaultName, tc.nameSlice, got, tc.want)
		}
	}
}

func TestCheckEpsilonVeryStrict(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		epsilon float64
		wantErr bool
	}{
		{"epsilon < 2⁻⁵⁰",
			math.Exp2(-51.0),
			true},
		{"epsilon == 2⁻⁵⁰",
			math.Exp2(-50.0),
			false},
		{"negative epsilon",
			-2,
			true},
		{"zero epsilon",
			0,
			true},
		{"epsilon is NaN",
			math.NaN(),
			true},
		{"epsilon is negative infinity",
			math.Inf(-1),
			true},
		{"epsilon is positive infinity",
			math.Inf(1),
			true},
		{"epsilon is infinity",
			math.Inf(0),
			true},
		{"positive epsilon",
			50,
			false},
	} {
		if err := CheckEpsilonVeryStrict(tc.epsilon); (err != nil) != tc.wantErr {
			t.Errorf("CheckEpsilonVeryStrict: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckEpsilonStrict(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		epsilon float64
		wantErr bool
	}{
		{"negative epsilon",
			-2,
			true},
		{"zero epsilon",
			0,
			true},
		{"epsilon is NaN",
			math.NaN(),
			true},
		{"epsilon is negative infinity",
			math.Inf(-1),
			true},
		{"epsilon is positive infinity",
			math.Inf(1),
			true},
		{"epsilon is infinity",
			math.Inf(0),
			true},
		{"positive epsilon",
			50,
			false},
	} {
		if err := CheckEpsilonStrict(tc.epsilon); (err != nil) != tc.wantErr {
			t.Errorf("CheckEpsilonStrict: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckEpsilon(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		epsilon float64
		wantErr bool
	}{
		{"negative epsilon",
			-2,
			true},
		{"zero epsilon",
			0,
			false},
		{"epsilon is NaN",
			math.NaN(),
			true},
		{"epsilon is negative infinity",
			math.Inf(-1),
			true},
		{"epsilon is positive infinity",
			math.Inf(1),
			true},
		{"epsilon is infinity",
			math.Inf(0),
			true},
		{"positive epsilon",
			50,
			false},
	} {
		if err := CheckEpsilon(tc.epsilon); (err != nil) != tc.wantErr {
			t.Errorf("CheckEpsilon: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckDelta(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		delta   float64
		wantErr bool
	}{
		{"negative delta",
			-2,
			true},
		{"zero delta",
			0,
			false},
		{"delta == 1",
			1,
			true},
		{"0 <= delta < 1",
			0.5,
			false},
		{"delta > 1",
			2,
			true},
		{"delta is NaN",
			math.NaN(),
			true},
		{"delta is negative infinity",
			math.Inf(-1),
			true},
		{"delta is positive infinity",
			math.Inf(1),
			true},
		{"delta is infinity",
			math.Inf(0),
			true},
	} {
		if err := CheckDelta(tc.delta); (err != nil) != tc.wantErr {
			t.Errorf("CheckDelta: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckDeltaStrict(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		delta   float64
		wantErr bool
	}{
		{"negative delta",
			-2,
			true},
		{"zero delta",
			0,
			true},
		{"delta == 1",
			1,
			true},
		{"0 < delta < 1",
			0.5,
			false},
		{"delta > 1",
			2,
			true},
		{"delta is NaN",
			math.NaN(),
			true},
		{"delta is negative infinity",
			math.Inf(-1),
			true},
		{"delta is positive infinity",
			math.Inf(1),
			true},
		{"delta is infinity",
			math.Inf(0),
			true},
	} {
		if err := CheckDeltaStrict(tc.delta); (err != nil) != tc.wantErr {
			t.Errorf("CheckDeltaStrict: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckNoDelta(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		delta   float64
		wantErr bool
	}{
		{"negative delta",
			-2,
			true},
		{"positive delta",
			10,
			true},
		{"zero delta",
			0,
			false},
	} {
		if err := CheckNoDelta(tc.delta); (err != nil) != tc.wantErr {
			t.Errorf("CheckNoDelta: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckL0Sensitivity(t *testing.T) {
	for _, tc := range []struct {
		desc          string
		l0Sensitivity int64
		wantErr       bool
	}{
		{"negative l0 sensitivity",
			-2,
			true},
		{"zero l0 sensitivity",
			0,
			true},
		{"l0 sensitivity == 10",
			10,
			false},
	} {
		if err := CheckL0Sensitivity(tc.l0Sensitivity); (err != nil) != tc.wantErr {
			t.Errorf("CheckL0Sensitivity: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckLInfSensitivity(t *testing.T) {
	for _, tc := range []struct {
		desc            string
		lInfSensitivity float64
		wantErr         bool
	}{
		{"negative lInf sensitivity",
			-2,
			true},
		{"zero lInf sensitivity",
			0,
			true},
		{"lInf sensitivity is negative infinity",
			math.Inf(-1),
			true},
		{"lInf sensitivity is positive infinity",
			math.Inf(1),
			true},
		{"lInf sensitivity is infinity",
			math.Inf(0),
			true},
		{"lInf sensitivity is NaN",
			math.Inf(-1),
			true},
		{"lInf sensitivity == 10",
			10,
			false},
	} {
		if err := CheckLInfSensitivity(tc.lInfSensitivity); (err != nil) != tc.wantErr {
			t.Errorf("CheckLInfSensitivity: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckBoundsInt64(t *testing.T) {
	for _, tc := range []struct {
		desc         string
		lower, upper int64
		wantErr      bool
	}{
		{"lower is min int64",
			math.MinInt64,
			-1,
			true,
		},
		{"upper is min int64",
			-2,
			math.MinInt64,
			true,
		},
		{"both bounds are min int64",
			math.MinInt64,
			math.MinInt64,
			true,
		},
		{"lower > upper",
			5,
			1,
			true,
		},
		{"lower == upper",
			1,
			1,
			false,
		},
		{"lower < upper",
			1,
			4,
			false,
		},
	} {
		if err := CheckBoundsInt64(tc.lower, tc.upper); (err != nil) != tc.wantErr {
			t.Errorf("CheckBoundsInt64: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckBoundsFloat64(t *testing.T) {
	for _, tc := range []struct {
		desc         string
		lower, upper float64
		wantErr      bool
	}{
		{"lower is NaN",
			math.NaN(),
			-1,
			true,
		},
		{"upper is NaN",
			-2,
			math.NaN(),
			true,
		},
		{"both bounds are NaN",
			math.NaN(),
			math.NaN(),
			true,
		},
		{"lower is positive infinity",
			math.Inf(1),
			-1,
			true,
		},
		{"upper is positive infinity",
			-2,
			math.Inf(1),
			true,
		},
		{"both bounds are positive infinity",
			math.Inf(1),
			math.Inf(1),
			true,
		},
		{"lower is negative infinity",
			math.Inf(-1),
			-1,
			true,
		},
		{"upper is negative infinity",
			-2,
			math.Inf(-1),
			true,
		},
		{"both bounds are negative infinity",
			math.Inf(-1),
			math.Inf(-1),
			true,
		},
		{"lower is  infinity",
			math.Inf(0),
			-1,
			true,
		},
		{"upper is infinity",
			-2,
			math.Inf(0),
			true,
		},
		{"both bounds are infinity",
			math.Inf(0),
			math.Inf(0),
			true,
		},
		{"lower > upper",
			5,
			1,
			true,
		},
		{"lower == upper",
			1,
			1,
			false,
		},
		{"lower < upper",
			1,
			4,
			false,
		},
	} {
		if err := CheckBoundsFloat64(tc.lower, tc.upper); (err != nil) != tc.wantErr {
			t.Errorf("CheckBoundsFloat64: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckBoundsFloat64AsInt64(t *testing.T) {
	for _, tc := range []struct {
		desc         string
		lower, upper float64
		wantErr      bool
	}{
		{"lower is NaN",
			math.NaN(),
			-1,
			true,
		},
		{"upper is NaN",
			-2,
			math.NaN(),
			true,
		},
		{"both bounds are NaN",
			math.NaN(),
			math.NaN(),
			true,
		},
		{"lower is max float64",
			math.MaxFloat64,
			-1,
			true,
		},
		{"lower is -max float64",
			-math.MaxFloat64,
			-1,
			true,
		},
		{"upper is max float64",
			1,
			math.MaxFloat64,
			true,
		},
		{"upper is -max float64",
			1,
			-math.MaxFloat64,
			true,
		},
		{"both bounds are max float64",
			math.MaxFloat64,
			math.MaxFloat64,
			true,
		},
		{"both bounds are -max float64",
			-math.MaxFloat64,
			-math.MaxFloat64,
			true,
		},
		{"ordinary, lower > upper",
			5,
			1,
			true,
		},
		{"ordinary, lower == upper",
			1,
			1,
			false,
		},
		{"ordinary, lower < upper",
			1,
			4,
			false,
		},
	} {
		if err := CheckBoundsFloat64AsInt64(tc.lower, tc.upper); (err != nil) != tc.wantErr {
			t.Errorf("CheckBoundsFloat64AsInt64: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckAlpha(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		alpha   float64
		wantErr bool
	}{
		{"negative alpha", -2, true},
		{"alpha larger than 1", 10, true},
		{"positive infinity alpha", math.Inf(1), true},
		{"negative infinity alpha", math.Inf(-1), true},
		{"infinity alpha", math.Inf(0), true},
		{"NaN alpha", math.NaN(), true},
		{"arbitrary alpha", 0.758464984, false},
	} {
		if err := CheckAlpha(tc.alpha); (err != nil) != tc.wantErr {
			t.Errorf("CheckAlpha: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckBoundsNotEqual(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		lower   float64
		upper   float64
		wantErr bool
	}{
		{"equal int bounds", -2, -2, true},
		{"unequal int bounds", 10, 15, false},
		{"equal float bounds", -2.0, -2.0, true},
		{"unequal float bounds", 10.0, 10.01, false},
	} {
		if err := CheckBoundsNotEqual(tc.lower, tc.upper); (err != nil) != tc.wantErr {
			t.Errorf("CheckBoundsNotEqual: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckTreeHeight(t *testing.T) {
	for _, tc := range []struct {
		desc       string
		treeHeight int
		wantErr    bool
	}{
		{"treeHeight < 1", 0, true},
		{"treeHeight = 1", 1, false},
		{"treeHeight > 1", 2, false},
	} {
		if err := CheckTreeHeight(tc.treeHeight); (err != nil) != tc.wantErr {
			t.Errorf("CheckTreeHeight: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCheckBranchingFactor(t *testing.T) {
	for _, tc := range []struct {
		desc            string
		branchingFactor int
		wantErr         bool
	}{
		{"branchingFactor < 2", 1, true},
		{"branchingFactor = 2", 2, false},
		{"branchingFactor > 2", 3, false},
	} {
		if err := CheckBranchingFactor(tc.branchingFactor); (err != nil) != tc.wantErr {
			t.Errorf("CheckBranchingFactor: when %s for err got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}
