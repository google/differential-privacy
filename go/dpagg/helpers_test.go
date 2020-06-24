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
	"testing"
)

func TestClampFloat64(t *testing.T) {
	for _, tc := range []struct {
		desc         string
		valueToClamp float64
		lower        float64
		upper        float64
		want         float64
		wantErr      bool
	}{
		{
			desc:         "Equal bounds, value is equal to bound",
			valueToClamp: 1,
			lower:        1,
			upper:        1,
			want:         1,
		},
		{
			desc:         "Equal bounds, value is less than bound",
			valueToClamp: -1,
			lower:        1,
			upper:        1,
			want:         1,
		},
		{
			desc:         "Equal bounds, value is greater than bound",
			valueToClamp: 2,
			lower:        1,
			upper:        1,
			want:         1,
		},
		{
			desc:         "Negative bounds, value is inside bounds",
			valueToClamp: -2,
			lower:        -3,
			upper:        -1,
			want:         -2,
		},
		{
			desc:         "Negative bounds, value is equal to upper bound",
			valueToClamp: -1,
			lower:        -3,
			upper:        -1,
			want:         -1,
		},
		{
			desc:         "Negative bounds, value is equal to lower bound",
			valueToClamp: -3,
			lower:        -3,
			upper:        -1,
			want:         -3,
		},
		{
			desc:         "Negative bounds, value is less than lower bound",
			valueToClamp: -4,
			lower:        -3,
			upper:        -1,
			want:         -3,
		},
		{
			desc:         "Negative bounds, value is greater than upper bound",
			valueToClamp: 0,
			lower:        -3,
			upper:        -1,
			want:         -1,
		},
		{
			desc:         "Positive bounds, value is inside bounds",
			valueToClamp: 2,
			lower:        0,
			upper:        3,
			want:         2,
		},
		{
			desc:         "Positive bounds, value is equal to upper bound",
			valueToClamp: 2,
			lower:        0,
			upper:        2,
			want:         2,
		},
		{
			desc:         "Positive bounds, value is equal to lower bound",
			valueToClamp: 0,
			lower:        0,
			upper:        2,
			want:         0,
		},
		{
			desc:         "Positive bounds, value is less than lower bound",
			valueToClamp: -4,
			lower:        0,
			upper:        1,
			want:         0,
		},
		{
			desc:         "Positive bounds, value is greater than upper bound",
			valueToClamp: 4,
			lower:        0,
			upper:        2,
			want:         2,
		},
		{
			desc:         "Incorrect bounds, lower > upper",
			valueToClamp: 4,
			lower:        5,
			upper:        2,
			want:         0,
			wantErr:      true,
		},
	} {
		got, err := ClampFloat64(tc.valueToClamp, tc.lower, tc.upper)
		if (err != nil) != tc.wantErr {
			t.Errorf("With %s, got=%v error, wantErr=%t", tc.desc, err, tc.wantErr)
		}

		if !approxEqual(got, tc.want) {
			t.Errorf("ClampFloat64: when %s got %v, want %v", tc.desc, got, tc.want)
		}
	}
}

func TestClampInt64(t *testing.T) {
	for _, tc := range []struct {
		desc         string
		valueToClamp int64
		lower        int64
		upper        int64
		want         int64
		wantErr      bool
	}{
		{
			desc:         "Equal bounds, value is equal to bound",
			valueToClamp: 1,
			lower:        1,
			upper:        1,
			want:         1,
		},
		{
			desc:         "Equal bounds, value is less than bound",
			valueToClamp: -1,
			lower:        1,
			upper:        1,
			want:         1,
		},
		{
			desc:         "Equal bounds, value is greater than bound",
			valueToClamp: 2,
			lower:        1,
			upper:        1,
			want:         1,
		},
		{
			desc:         "Negative bounds, value is inside bounds",
			valueToClamp: -2,
			lower:        -3,
			upper:        -1,
			want:         -2,
		},
		{
			desc:         "Negative bounds, value is equal to upper bound",
			valueToClamp: -1,
			lower:        -3,
			upper:        -1,
			want:         -1,
		},
		{
			desc:         "Negative bounds, value is equal to lower bound",
			valueToClamp: -3,
			lower:        -3,
			upper:        -1,
			want:         -3,
		},
		{
			desc:         "Negative bounds, value is less than lower bound",
			valueToClamp: -4,
			lower:        -3,
			upper:        -1,
			want:         -3,
		},
		{
			desc:         "Negative bounds, value is greater than upper bound",
			valueToClamp: 0,
			lower:        -3,
			upper:        -1,
			want:         -1,
		},
		{
			desc:         "Positive bounds, value is inside bounds",
			valueToClamp: 2,
			lower:        0,
			upper:        3,
			want:         2,
		},
		{
			desc:         "Positive bounds, value is equal to upper bound",
			valueToClamp: 2,
			lower:        0,
			upper:        2,
			want:         2,
		},
		{
			desc:         "Positive bounds, value is equal to lower bound",
			valueToClamp: 0,
			lower:        0,
			upper:        2,
			want:         0,
		},
		{
			desc:         "Positive bounds, value is less than lower bound",
			valueToClamp: -4,
			lower:        0,
			upper:        1,
			want:         0,
		},
		{
			desc:         "Positive bounds, value is greater than upper bound",
			valueToClamp: 4,
			lower:        0,
			upper:        2,
			want:         2,
		},
		{
			desc:         "Incorrect bounds, lower > upper",
			valueToClamp: 4,
			lower:        5,
			upper:        2,
			want:         0,
			wantErr:      true,
		},
	} {
		got, err := ClampInt64(tc.valueToClamp, tc.lower, tc.upper)
		if (err != nil) != tc.wantErr {
			t.Errorf("With %s, got=%v error, wantErr=%t", tc.desc, err, tc.wantErr)
		}
		if got != tc.want {
			t.Errorf("ClampFloat64: when %s got %v, want %v", tc.desc, got, tc.want)
		}
	}
}
