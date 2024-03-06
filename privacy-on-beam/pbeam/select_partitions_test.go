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

	"github.com/google/differential-privacy/go/v3/dpagg"
	"github.com/google/differential-privacy/privacy-on-beam/v3/pbeam/testutils"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/testing/ptest"
)

// Checks that SelectPartitions is performing a random partition selection
// for PrivatePCollection<V> inputs.
func TestSelectPartitionsIsNonDeterministicV(t *testing.T) {
	for _, tc := range []struct {
		name          string
		epsilon       float64
		delta         float64
		numPartitions int
	}{
		{
			epsilon: 1,
			delta:   0.3, // yields a 30% chance of emitting any particular partition.
			// 143 distinct partitions implies that some (but not all) partitions are
			// emitted with high probability (at least 1 - 1e-20).
			numPartitions: 143,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Build up {ID, Value} pairs such that 1 privacy unit contributes to
			// each of the tc.numPartitions partitions:
			//    {0,0}, {1,1}, ..., {numPartitions-1,numPartitions-1}
			var (
				pairs []testutils.PairII
			)
			for i := 0; i < tc.numPartitions; i++ {
				pairs = append(pairs, testutils.PairII{i, i})
			}
			p, s, col := ptest.CreateList(pairs)
			col = beam.ParDo(s, testutils.PairToKV, col)

			// Run SelectPartitions on pairs
			pcol := MakePrivate(s, col, privacySpec(t,
				PrivacySpecParams{
					PartitionSelectionEpsilon: tc.epsilon,
					PartitionSelectionDelta:   tc.delta,
				}))
			got := SelectPartitions(s, pcol, PartitionSelectionParams{MaxPartitionsContributed: 1})

			// Validate that partitions are selected randomly (i.e., some emitted and some dropped).
			testutils.CheckSomePartitionsAreDropped(s, got, tc.numPartitions)
			if err := ptest.Run(p); err != nil {
				t.Errorf("%v", err)
			}
		})
	}
}

// Checks that SelectPartitions is performing a random partition selection
// for PrivatePCollection<K,V> inputs.
func TestSelectPartitionsIsNonDeterministicKV(t *testing.T) {
	for _, tc := range []struct {
		name          string
		epsilon       float64
		delta         float64
		numPartitions int
	}{
		{
			epsilon: 1,
			delta:   0.3, // yields a 30% chance of emitting any particular partition.
			// 143 distinct partitions implies that some (but not all) partitions are
			// emitted with high probability (at least 1 - 1e-20).
			numPartitions: 143,
		},
		{
			epsilon: 1,
			delta:   0.3, // yields a 30% chance of emitting any particular partition.
			// 143 distinct partitions implies that some (but not all) partitions are
			// emitted with high probability (at least 1 - 1e-20).
			numPartitions: 143,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Build up {ID, Partition, Value} pairs such that 1 privacy unit contributes to
			// each of the tc.numPartitions partitions:
			//    {0,0,0}, {1,1,0}, ..., {numPartitions-1,numPartitions-1,0}
			var (
				triples []testutils.TripleWithIntValue
			)
			for i := 0; i < tc.numPartitions; i++ {
				triples = append(triples, testutils.TripleWithIntValue{i, i, 0})
			}
			p, s, col := ptest.CreateList(triples)
			col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

			// Run SelectPartitions on triples
			pcol := MakePrivate(s, col, privacySpec(t,
				PrivacySpecParams{
					PartitionSelectionEpsilon: tc.epsilon,
					PartitionSelectionDelta:   tc.delta,
				}))
			pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
			got := SelectPartitions(s, pcol, PartitionSelectionParams{MaxPartitionsContributed: 1})

			// Validate that partitions are selected randomly (i.e., some emitted and some dropped).
			testutils.CheckSomePartitionsAreDropped(s, got, tc.numPartitions)
			if err := ptest.Run(p); err != nil {
				t.Errorf("%v", err)
			}
		})
	}
}

// Checks that SelectPartitions bounds cross-partition contributions correctly
// for PrivatePCollection<V> inputs.
func TestSelectPartitionsBoundsCrossPartitionContributionsV(t *testing.T) {
	// Create 10 partitions with a single privacy ID contributing to each.
	var pairs []testutils.PairII
	for i := 0; i < 10; i++ {
		pairs = append(pairs, testutils.MakePairsWithFixedV(1, i)...)
	}
	p, s, col := ptest.CreateList(pairs)
	col = beam.ParDo(s, testutils.PairToKV, col)

	// ε=50, δ=~1 and l0Sensitivity=1 gives a threshold of 2.
	epsilon, delta, l0Sensitivity := 50.0, dpagg.LargestRepresentableDelta, 1
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	got := SelectPartitions(s, pcol, PartitionSelectionParams{MaxPartitionsContributed: int64(l0Sensitivity)})
	// With a max contribution of 1, only 1 partition should be outputted.
	testutils.CheckNumPartitions(s, got, 1)
	if err := ptest.Run(p); err != nil {
		t.Errorf("Did not bound cross partition contributions correctly for PrivatePCollection<V> inputs: %v", err)
	}
}

// Checks that SelectPartitions bounds cross-partition contributions correctly
// for PrivatePCollection<K,V> inputs.
func TestSelectPartitionsBoundsCrossPartitionContributionsKV(t *testing.T) {
	// Create 10 partitions with a single privacy ID contributing to each.
	var triples []testutils.TripleWithIntValue
	for i := 0; i < 10; i++ {
		triples = append(triples, testutils.MakeTripleWithIntValue(1, i, 0)...)
	}
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

	// ε=50, δ=~1 and l0Sensitivity=1 gives a threshold of 2.
	epsilon, delta, l0Sensitivity := 50.0, dpagg.LargestRepresentableDelta, 1
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
	got := SelectPartitions(s, pcol, PartitionSelectionParams{MaxPartitionsContributed: int64(l0Sensitivity)})
	// With a max contribution of 1, only 1 partition should be outputted.
	testutils.CheckNumPartitions(s, got, 1)
	if err := ptest.Run(p); err != nil {
		t.Errorf("Did not bound cross partition contributions correctly for PrivatePCollection<K,V> inputs: %v", err)
	}
}

func TestSelectPartitionsPrethresholding(t *testing.T) {
	// Create two partitions, one with 4 users and the other with 5 users.
	triples := testutils.MakeTripleWithIntValue(4, 0, 0)
	triples = append(triples, testutils.MakeTripleWithIntValueStartingFromKey(4, 5, 1, 0)...)
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

	// We set very large epsilon & delta, and a pre-threshold of 5, so the partition
	// with 5 users should be kept and the one with 4 users should not be kept.
	epsilon, delta, preThreshold, l0Sensitivity := 1e9, dpagg.LargestRepresentableDelta, int64(5), int64(1)
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
			PreThreshold:              preThreshold,
		}))
	pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
	got := SelectPartitions(s, pcol, PartitionSelectionParams{MaxPartitionsContributed: l0Sensitivity})

	// Assert
	testutils.CheckNumPartitions(s, got, 1)
	if err := ptest.Run(p); err != nil {
		t.Errorf("Expected only a single partition to be kept with pre-thresholding:  %v", err)
	}
}
