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
	"bytes"
	"encoding/base64"
	"reflect"
	"testing"

	"flag"
	"github.com/google/differential-privacy/privacy-on-beam/v3/pbeam/testutils"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/testing/ptest"
)

func TestPartitionMapFnAddInput(t *testing.T) {
	fn := newPartitionMapFn(beam.EncodedType{reflect.TypeOf("")})
	fn.Setup()
	var partitionBuf bytes.Buffer
	if err := fn.partitionEnc.Encode("partition_key", &partitionBuf); err != nil {
		t.Fatalf("Couldn't encode 'partition_key'")
	}

	accum := fn.CreateAccumulator()
	fn.AddInput(accum, "partition_key")

	pk := base64.StdEncoding.EncodeToString(partitionBuf.Bytes())
	if !accum[pk] {
		t.Errorf("'partition_key' was not found in partition map")
	}
}

func TestPartitionMapFnMergeAccumulators(t *testing.T) {
	fn := newPartitionMapFn(beam.EncodedType{reflect.TypeOf("")})
	fn.Setup()
	var partitionBuf bytes.Buffer
	if err := fn.partitionEnc.Encode("partition_key_1", &partitionBuf); err != nil {
		t.Fatalf("Couldn't encode 'partition_key_1'")
	}
	pk1 := base64.StdEncoding.EncodeToString(partitionBuf.Bytes())
	partitionBuf.Reset()
	if err := fn.partitionEnc.Encode("partition_key_2", &partitionBuf); err != nil {
		t.Fatalf("Couldn't encode 'partition_key_2'")
	}
	pk2 := base64.StdEncoding.EncodeToString(partitionBuf.Bytes())

	accum1 := fn.CreateAccumulator()
	fn.AddInput(accum1, "partition_key_1")
	accum2 := fn.CreateAccumulator()
	fn.AddInput(accum2, "partition_key_2")
	merged := fn.MergeAccumulators(accum1, accum2)

	if !merged[pk1] {
		t.Errorf("'partition_key_1' was not found in partition map")
	}
	if !merged[pk2] {
		t.Errorf("'partition_key_2' was not found in partition map")
	}
}

// Checks that elements with non-public partitions are dropped.
// This function is used for count and distinct_id.
func TestDropNonPublicPartitionsVFn(t *testing.T) {
	pairs := testutils.ConcatenatePairs(
		testutils.MakePairsWithFixedV(7, 0),
		testutils.MakePairsWithFixedVStartingFromKey(7, 10, 1),
		testutils.MakePairsWithFixedVStartingFromKey(17, 83, 2),
		testutils.MakePairsWithFixedVStartingFromKey(100, 10, 3),
	)

	// Keep partitions 0, 2;
	// drop partitions 1, 3.
	result := testutils.ConcatenatePairs(
		testutils.MakePairsWithFixedV(7, 0),
		testutils.MakePairsWithFixedVStartingFromKey(17, 83, 2),
	)

	p, s, col, want := ptest.CreateList2(pairs, result)
	want = beam.ParDo(s, testutils.PairToKV, want)
	col = beam.ParDo(s, testutils.PairToKV, col)
	partitions := []int{0, 2}

	partitionsCol := beam.CreateList(s, partitions)
	epsilon := 50.0
	pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
	got := dropNonPublicPartitionsVFn(s, partitionsCol, pcol)
	testutils.EqualsKVInt(t, s, got, want)
	if err := ptest.Run(p); err != nil {
		t.Errorf("DropNonPublicPartitionsVFn did not drop non public partitions as expected: %v", err)
	}
}

// TestDropNonPublicPartitionsKVFn checks that int elements with non-public partitions
// are dropped (tests function used for sum and mean).
func TestDropNonPublicPartitionsKVFn(t *testing.T) {
	triples := testutils.ConcatenateTriplesWithIntValue(
		testutils.MakeTripleWithIntValueStartingFromKey(0, 7, 0, 0),
		testutils.MakeTripleWithIntValueStartingFromKey(7, 3, 1, 0),
		testutils.MakeTripleWithIntValueStartingFromKey(10, 90, 2, 0),
		testutils.MakeTripleWithIntValueStartingFromKey(100, 100, 11, 0),
		testutils.MakeTripleWithIntValueStartingFromKey(200, 5, 12, 0))
	// Keep partitions 0, 2.
	// Drop partitions 1, 33, 100.
	result := testutils.ConcatenateTriplesWithIntValue(
		testutils.MakeTripleWithIntValueStartingFromKey(0, 7, 0, 0),
		testutils.MakeTripleWithIntValueStartingFromKey(10, 90, 2, 0))

	p, s, col, col2 := ptest.CreateList2(triples, result)
	// Doesn't matter that the values 3, 4, 5, 6, 9, 10
	// are in the partitions PCollection because we are
	// just dropping the values that are in our original PCollection
	// that are not in public partitions.
	partitionsCol := beam.CreateList(s, []int{0, 2, 3, 4, 5, 6, 9, 10})
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)
	col2 = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col2)
	epsilon := 50.0

	pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
	pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
	idT, _ := beam.ValidateKVType(pcol.col)

	got := dropNonPublicPartitionsKVFn(s, partitionsCol, pcol, idT)
	got = beam.SwapKV(s, got)

	pcol2 := MakePrivate(s, col2, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
	pcol2 = ParDo(s, testutils.TripleWithIntValueToKV, pcol2)
	want := pcol2.col
	want = beam.SwapKV(s, want)

	testutils.EqualsKVInt(t, s, got, want)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestDropNonPublicPartitionsKVFn did not drop non public partitions as expected: %v", err)
	}
}

// Check that float elements with non-public partitions
// are dropped (tests function used for sum and mean).
func TestDropNonPublicPartitionsFloat(t *testing.T) {
	// In this test, we check  that non-public partitions
	// are dropped. This function is used for sum and mean.
	// Used example values from the mean test.
	triples := testutils.ConcatenateTriplesWithFloatValue(
		testutils.MakeTripleWithFloatValue(7, 0, 2.0),
		testutils.MakeTripleWithFloatValueStartingFromKey(7, 100, 1, 1.3),
		testutils.MakeTripleWithFloatValueStartingFromKey(107, 150, 1, 2.5),
	)
	// Keep partition 0.
	// drop partition 1.
	result := testutils.ConcatenateTriplesWithFloatValue(
		testutils.MakeTripleWithFloatValue(7, 0, 2.0))

	p, s, col, col2 := ptest.CreateList2(triples, result)

	// Doesn't matter that the values 2, 3, 4, 5, 6, 7 are in the partitions PCollection.
	// We are just dropping the values that are in our original PCollection that are not in
	// public partitions.
	partitionsCol := beam.CreateList(s, []int{0, 2, 3, 4, 5, 6, 7})
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)
	col2 = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col2)
	epsilon := 50.0

	pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	idT, _ := beam.ValidateKVType(pcol.col)

	got := dropNonPublicPartitionsKVFn(s, partitionsCol, pcol, idT)
	got = beam.SwapKV(s, got)

	pcol2 := MakePrivate(s, col2, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
	pcol2 = ParDo(s, testutils.TripleWithFloatValueToKV, pcol2)
	want := pcol2.col
	want = beam.SwapKV(s, want)

	testutils.EqualsKVInt(t, s, got, want)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestDropNonPublicPartitionsFloat did not drop non public partitions as expected: %v", err)
	}
}

// TODO: Remove once the enable_sharded_public_partitions flag is gone.
func TestDropNonPublicPartitionsFloatShardedImpl(t *testing.T) {
	flag.Set("enable_sharded_public_partitions", "true")

	// In this test, we check  that non-public partitions
	// are dropped. This function is used for sum and mean.
	// Used example values from the mean test.
	triples := testutils.ConcatenateTriplesWithFloatValue(
		testutils.MakeTripleWithFloatValue(7, 0, 2.0),
		testutils.MakeTripleWithFloatValueStartingFromKey(7, 100, 1, 1.3),
		testutils.MakeTripleWithFloatValueStartingFromKey(107, 150, 1, 2.5),
	)
	// Keep partition 0.
	// drop partition 1.
	result := testutils.ConcatenateTriplesWithFloatValue(
		testutils.MakeTripleWithFloatValue(7, 0, 2.0))

	p, s, col, col2 := ptest.CreateList2(triples, result)

	// Doesn't matter that the values 2, 3, 4, 5, 6, 7 are in the partitions PCollection.
	// We are just dropping the values that are in our original PCollection that are not in
	// public partitions.
	partitionsCol := beam.CreateList(s, []int{0, 2, 3, 4, 5, 6, 7})
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)
	col2 = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col2)
	epsilon := 50.0

	pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	idT, _ := beam.ValidateKVType(pcol.col)

	got := dropNonPublicPartitionsKVFn(s, partitionsCol, pcol, idT)
	got = beam.SwapKV(s, got)

	pcol2 := MakePrivate(s, col2, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
	pcol2 = ParDo(s, testutils.TripleWithFloatValueToKV, pcol2)
	want := pcol2.col
	want = beam.SwapKV(s, want)

	testutils.EqualsKVInt(t, s, got, want)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestDropNonPublicPartitionsFloat did not drop non public partitions as expected: %v", err)
	}
	flag.Set("enable_sharded_public_partitions", "false")
}
