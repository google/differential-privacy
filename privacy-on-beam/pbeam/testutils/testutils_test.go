//
// Copyright 2021 Google LLC
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

package testutils

import (
	"testing"

	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/testing/ptest"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

// Tests for the test helpers.

func TestMain(m *testing.M) {
	ptest.Main(m)
}

func TestApproxEqualsKVInt64(t *testing.T) {
	tolerance := 1.0
	for _, tc := range []struct {
		desc    string
		values1 []PairII64
		values2 []PairII64
		wantErr bool
	}{
		{"same values",
			[]PairII64{
				{35, 0},
				{99, 1},
			},
			[]PairII64{
				{35, 0},
				{99, 1},
			},
			false,
		},
		{"approximately equal values",
			[]PairII64{
				{35, 0}, // Equal to -1 within a tolerance of 1.0
				{99, 1}, // Equal to 2 within a tolerance of 1.0
			},
			[]PairII64{
				{35, -1},
				{99, 2},
			},
			false,
		},
		{"sufficiently different values",
			[]PairII64{
				{35, 0},
				{99, 1},
			},
			[]PairII64{
				{35, 10},
				{99, 7},
			},
			true,
		},
	} {
		p, s, col1, col2 := ptest.CreateList2(tc.values1, tc.values2)
		col1KV := beam.ParDo(s, PairII64ToKV, col1)
		col2KV := beam.ParDo(s, PairII64ToKV, col2)
		if err := ApproxEqualsKVInt64(s, col1KV, col2KV, tolerance); err != nil {
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
		values1 []PairIF64
		values2 []PairIF64
		wantErr bool
	}{
		{"same values",
			[]PairIF64{
				{35, 0.1},
				{99, 1.0},
			},
			[]PairIF64{
				{35, 0.1},
				{99, 1.00},
			},
			false,
		},
		{"approximately equal values",
			[]PairIF64{
				{35, 0.1}, // Equal to 0.2 within a tolerance of 0.1
				{99, 1.0}, // Equal to 1.05 within a tolerance of 0.1
			},
			[]PairIF64{
				{35, 0.2},
				{99, 1.05},
			},
			false,
		},
		{"sufficiently different values",
			[]PairIF64{
				{35, 0.1},
				{99, 1.0},
			},
			[]PairIF64{
				{35, 10.3},
				{99, 7.6},
			},
			true,
		},
	} {
		p, s, col1, col2 := ptest.CreateList2(tc.values1, tc.values2)
		col1KV := beam.ParDo(s, PairIFToKV, col1)
		col2KV := beam.ParDo(s, PairIFToKV, col2)
		if err := ApproxEqualsKVFloat64(s, col1KV, col2KV, tolerance); err != nil {
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

func TestCheckNumPartitionsFn(t *testing.T) {
	for _, tc := range []struct {
		desc           string
		numPartitions  int
		wantPartitions int
		wantErr        bool
	}{
		{"same number of partitions",
			5,
			5,
			false,
		},
		{"different number of partitions",
			5,
			6,
			true,
		},
		{"got and want zero number of partitions",
			0,
			0,
			false,
		},
		{"got zero number of partitions want non-zero number of partitions",
			0,
			5,
			true,
		},
	} {
		partitions := make([]int, tc.numPartitions)
		p, s, col := ptest.CreateList(partitions)

		CheckNumPartitions(s, col, tc.wantPartitions)
		if err := ptest.Run(p); (err != nil) != tc.wantErr {
			t.Errorf("With %s, got error=%v, wantErr=%t", tc.desc, err, tc.wantErr)
		}
	}
}
