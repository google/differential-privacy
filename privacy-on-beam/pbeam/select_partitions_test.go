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
	"fmt"
	"reflect"
	"testing"

	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/testing/ptest"
	"github.com/apache/beam/sdks/go/pkg/beam/transforms/stats"
)

func init() {
	beam.RegisterType(reflect.TypeOf((*gotExpectedNumPartitionsFn)(nil)))
}

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
				pairs []pairII
			)
			for i := 0; i < tc.numPartitions; i++ {
				pairs = append(pairs, pairII{i, i})
			}
			p, s, col := ptest.CreateList(pairs)
			col = beam.ParDo(s, pairToKV, col)

			// Run SelectPartitions on pairs
			pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
			got := SelectPartitions(s, pcol, SelectPartitionsParams{MaxPartitionsContributed: 1})

			// Validate that partitions are selected randomly (i.e., some emitted and some dropped).
			checkSomePartitionsAreDropped(s, got, tc.numPartitions)
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
				triples []tripleWithIntValue
			)
			for i := 0; i < tc.numPartitions; i++ {
				triples = append(triples, tripleWithIntValue{i, i, 0})
			}
			p, s, col := ptest.CreateList(triples)
			col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

			// Run SelectPartitions on triples
			pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
			pcol = ParDo(s, tripleWithIntValueToKV, pcol)
			got := SelectPartitions(s, pcol, SelectPartitionsParams{MaxPartitionsContributed: 1})

			// Validate that partitions are selected randomly (i.e., some emitted and some dropped).
			checkSomePartitionsAreDropped(s, got, tc.numPartitions)
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
	var pairs []pairII
	for i := 0; i < 10; i++ {
		pairs = append(pairs, makePairsWithFixedV(1, i)...)
	}
	p, s, col := ptest.CreateList(pairs)
	col = beam.ParDo(s, pairToKV, col)

	// ε=50, δ=0.01 and l1Sensitivity=3 gives a threshold of 1.
	epsilon, delta, l1Sensitivity := 50.0, 0.01, 3
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	got := SelectPartitions(s, pcol, SelectPartitionsParams{MaxPartitionsContributed: int64(l1Sensitivity)})
	// With a max contribution of 3, only 3 partitions should be outputted.
	checkNumPartitions(s, got, 3)
	if err := ptest.Run(p); err != nil {
		t.Errorf("Did not bound cross partition contributions correctly for PrivatePCollection<V> inputs: %v", err)
	}
}

// Checks that SelectPartitions bounds cross-partition contributions correctly
// for PrivatePCollection<K,V> inputs.
func TestSelectPartitionsBoundsCrossPartitionContributionsKV(t *testing.T) {
	// Create 10 partitions with a single privacy ID contributing to each.
	var triples []tripleWithIntValue
	for i := 0; i < 10; i++ {
		triples = append(triples, makeTripleWithIntValue(1, i, 0)...)
	}
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

	// ε=50, δ=0.01 and l1Sensitivity=3 gives a threshold of 1.
	epsilon, delta, l1Sensitivity := 50.0, 0.01, 3
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	pcol = ParDo(s, tripleWithIntValueToKV, pcol)
	got := SelectPartitions(s, pcol, SelectPartitionsParams{MaxPartitionsContributed: int64(l1Sensitivity)})
	// With a max contribution of 3, only 3 partitions should be outputted.
	checkNumPartitions(s, got, 3)
	if err := ptest.Run(p); err != nil {
		t.Errorf("Did not bound cross partition contributions correctly for PrivatePCollection<K,V> inputs: %v", err)
	}
}

func checkNumPartitions(s beam.Scope, col beam.PCollection, expected int) {
	ones := beam.ParDo(s, oneFn, col)
	numPartitions := stats.Sum(s, ones)
	beam.ParDo0(s, &gotExpectedNumPartitionsFn{Expected: expected}, numPartitions)
}

type gotExpectedNumPartitionsFn struct {
	Expected int
}

func (fn *gotExpectedNumPartitionsFn) ProcessElement(i int) error {
	if i != fn.Expected {
		return fmt.Errorf("got %d emitted partitions, want %d", i, fn.Expected)
	}
	return nil
}
