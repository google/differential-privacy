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

package pbeam

import (
	"testing"

	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/testing/ptest"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

// Tests for the test helpers.

func TestApproxEqualsKVInt64(t *testing.T) {
	tolerance := 1.0
	for _, tc := range []struct {
		desc    string
		values1 []pairII64
		values2 []pairII64
		wantErr bool
	}{
		{"same values",
			[]pairII64{
				{35, 0},
				{99, 1},
			},
			[]pairII64{
				{35, 0},
				{99, 1},
			},
			false,
		},
		{"approximately equal values",
			[]pairII64{
				{35, 0}, // Equal to -1 within a tolerance of 1.0
				{99, 1}, // Equal to 2 within a tolerance of 1.0
			},
			[]pairII64{
				{35, -1},
				{99, 2},
			},
			false,
		},
		{"sufficiently different values",
			[]pairII64{
				{35, 0},
				{99, 1},
			},
			[]pairII64{
				{35, 10},
				{99, 7},
			},
			true,
		},
	} {
		p, s, col1, col2 := ptest.CreateList2(tc.values1, tc.values2)
		col1KV := beam.ParDo(s, pairII64ToKV, col1)
		col2KV := beam.ParDo(s, pairII64ToKV, col2)
		if err := approxEqualsKVInt64(s, col1KV, col2KV, tolerance); err != nil {
			t.Fatalf("TestApproxEqualsKVInt64: %v", err)
		}
		if err := ptest.Run(p); (err != nil) != tc.wantErr {
			t.Errorf("TestApproxEqualsKVInt64 failed for %s: got=%v, wantErr=%v", tc.desc, err, tc.wantErr)
		}
	}
}

func TestApproxEqualsKVFloat64(t *testing.T) {
	tolerance := 0.1
	for _, tc := range []struct {
		desc    string
		values1 []pairIF64
		values2 []pairIF64
		wantErr bool
	}{
		{"same values",
			[]pairIF64{
				{35, 0.1},
				{99, 1.0},
			},
			[]pairIF64{
				{35, 0.1},
				{99, 1.00},
			},
			false,
		},
		{"approximately equal values",
			[]pairIF64{
				{35, 0.1}, // Equal to 0.2 within a tolerance of 0.1
				{99, 1.0}, // Equal to 1.05 within a tolerance of 0.1
			},
			[]pairIF64{
				{35, 0.2},
				{99, 1.05},
			},
			false,
		},
		{"sufficiently different values",
			[]pairIF64{
				{35, 0.1},
				{99, 1.0},
			},
			[]pairIF64{
				{35, 10.3},
				{99, 7.6},
			},
			true,
		},
	} {
		p, s, col1, col2 := ptest.CreateList2(tc.values1, tc.values2)
		col1KV := beam.ParDo(s, pairIFToKV, col1)
		col2KV := beam.ParDo(s, pairIFToKV, col2)
		if err := approxEqualsKVFloat64(s, col1KV, col2KV, tolerance); err != nil {
			t.Fatalf("TestApproxEqualsKVFloat64: %v", err)
		}
		if err := ptest.Run(p); (err != nil) != tc.wantErr {
			t.Errorf("TestApproxEqualsKVFloat64 failed for %s: got=%v, wantErr=%v", tc.desc, err, tc.wantErr)
		}
	}
}

func assertFloat64PtrHasApproxValue(t *testing.T, got *float64, wantValue, tolerance float64) {
	if got == nil {
		t.Errorf("got <nil>, want: %g", wantValue)
	} else if diff := cmp.Diff(*got, wantValue, cmpopts.EquateApprox(0, tolerance)); diff != "" {
		t.Errorf("got %g, want %g", *got, wantValue)
	}
}
