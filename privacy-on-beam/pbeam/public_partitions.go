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

// This file contains logic for handling public partitions.

package pbeam

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"reflect"

	"github.com/google/differential-privacy/privacy-on-beam/v3/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/core/typex"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/register"
)

func init() {
	register.Combiner2[pMap, beam.W](&partitionMapFn{})

	register.DoFn2x3[beam.U, kv.Pair, beam.W, kv.Pair, error](&encodeIDVFn{})
	register.DoFn2x3[beam.W, kv.Pair, beam.U, kv.Pair, error](&decodeIDVFn{})
	register.DoFn3x1[beam.U, beam.W, func(beam.U, beam.W), error](&prunePartitionsInMemoryVFn{})
	register.Emitter2[beam.U, beam.W]()
	register.DoFn3x0[beam.U, kv.Pair, func(beam.U, kv.Pair)](&prunePartitionsInMemoryKVFn{})
	register.Emitter2[beam.U, kv.Pair]()

	register.Function1x2[beam.W, beam.W, int64](addZeroValuesToPublicPartitionsInt64)
	register.Function1x2[beam.W, beam.W, float64](addZeroValuesToPublicPartitionsFloat64)
	register.Function1x2[beam.W, beam.W, []float64](addEmptySliceToPublicPartitionsFloat64)
	register.Function4x1[beam.U, kv.Pair, func(*pMap) bool, func(beam.U, kv.Pair), error](prunePartitionsKV)
	register.Function4x0[beam.V, func(*int64) bool, func(*beam.U) bool, func(beam.U, beam.V)](mergePublicValues)
	register.Iter1[int64]()
	register.Iter1[beam.U]()
	register.Emitter2[beam.U, beam.V]()
	register.Function4x0[beam.W, func(*beam.V) bool, func(*beam.V) bool, func(beam.W, beam.V)](mergeResultWithEmptyPublicPartitionsFn)
	register.Iter1[beam.V]()
	register.Emitter2[beam.W, beam.V]()
}

// newAddZeroValuesToPublicPartitionsFn turns a PCollection<V> into PCollection<V,0>.
func newAddZeroValuesToPublicPartitionsFn(vKind reflect.Kind) (any, error) {
	switch vKind {
	case reflect.Int64:
		return addZeroValuesToPublicPartitionsInt64, nil
	case reflect.Float64:
		return addZeroValuesToPublicPartitionsFloat64, nil
	default:
		return nil, fmt.Errorf("vKind(%v) should be int64 or float64", vKind)
	}
}

func addZeroValuesToPublicPartitionsInt64(partition beam.W) (k beam.W, v int64) {
	return partition, 0
}

func addZeroValuesToPublicPartitionsFloat64(partition beam.W) (k beam.W, v float64) {
	return partition, 0
}

func addEmptySliceToPublicPartitionsFloat64(partition beam.W) (k beam.W, v []float64) {
	return partition, []float64{}
}

// pMap holds a set of partition keys for quick lookup as a map from string to bool.
// Key is the base64 string representation of encoded partition key.
// Value is set to true if partition key exists.
type pMap map[string]bool

// dropNonPublicPartitions returns the PCollection with the non-public partitions dropped if public partitions are
// specified. Returns the input PCollection otherwise.
func dropNonPublicPartitions(s beam.Scope, pcol PrivatePCollection, publicPartitions any, partitionType reflect.Type) (beam.PCollection, error) {
	// Obtain type information from the underlying PCollection<K,V>.
	idT, _ := beam.ValidateKVType(pcol.col)

	// If PublicPartitions is not specified, return the input collection.
	if publicPartitions == nil {
		return pcol.col, nil
	}

	// Drop non-public partitions, if public partitions are specified as a PCollection.
	if publicPartitionscCol, ok := publicPartitions.(beam.PCollection); ok {
		// Data is <PrivacyKey, PartitionKey, Value>
		if pcol.codec != nil {
			return dropNonPublicPartitionsKVFn(s, publicPartitionscCol, pcol, idT), nil
		}
		// Data is <PrivacyKey, PartitionKey>
		return dropNonPublicPartitionsVFn(s, publicPartitionscCol, pcol), nil
	}

	// Drop non-public partitions, public partitions are specified as slice/array (i.e., in-memory).
	// Convert PublicPartitions to map for quick lookup.
	partitionEnc := beam.NewElementEncoder(partitionType)
	partitionMap := pMap{}
	for i := 0; i < reflect.ValueOf(publicPartitions).Len(); i++ {
		partitionKey := reflect.ValueOf(publicPartitions).Index(i).Interface()
		var partitionBuf bytes.Buffer
		if err := partitionEnc.Encode(partitionKey, &partitionBuf); err != nil {
			return pcol.col, fmt.Errorf("couldn't encode partition %v: %v", partitionKey, err)
		}
		partitionMap[base64.StdEncoding.EncodeToString(partitionBuf.Bytes())] = true
	}
	// Data is <PrivacyKey, PartitionKey, Value>
	if pcol.codec != nil {
		return beam.ParDo(s, newPrunePartitionsInMemoryKVFn(partitionMap), pcol.col), nil
	}
	// Data is <PrivacyKey, PartitionKey>
	partitionEncodedType := beam.EncodedType{partitionType}
	return beam.ParDo(s, newPrunePartitionsInMemoryVFn(partitionEncodedType, partitionMap), pcol.col), nil
}

// mergePublicValues merges the public partitions with the values for a PrivatePCollection
// after a CoGroupByKey. Only outputs a <privacyKey, v> pair (where v is value in the case
// of Count & DistinctPrivacyID, and kv.Pair for other aggregations) if the value is in
// the public partitions, i.e., the PCollection that is passed to the CoGroupByKey first.
func mergePublicValues(value beam.V, isKnown func(*int64) bool, privacyKeys func(*beam.U) bool, emit func(beam.U, beam.V)) {
	var ignoredZero int64
	if isKnown(&ignoredZero) {
		var privacyKey beam.U
		for privacyKeys(&privacyKey) {
			emit(privacyKey, value)
		}
	}
}

// dropNonPublicPartitionsVFn drops partitions not specified in
// PublicPartitions from pcol. It can be used for aggregations on V values,
// e.g. Count and DistinctPrivacyID.
//
// We drop values that are not in the publicPartitions PCollection as follows:
//  1. Transform publicPartitions from <V> to <V, int64(0)> (0 is a placeholder value)
//  2. Swap pcol.col from <PrivacyKey, V> to <V, PrivacyKey>
//  3. Do a CoGroupByKey on the output of 1 and 2.
//  4. From the output of 3, only output <PrivacyKey, V> if there is an input
//     from 1 using the mergePublicValues.
//
// Returns a PCollection<PrivacyKey, Value> only for values present in
// publicPartitions.
func dropNonPublicPartitionsVFn(s beam.Scope, publicPartitions beam.PCollection, pcol PrivatePCollection) beam.PCollection {
	publicPartitionsWithZeros := beam.ParDo(s, addZeroValuesToPublicPartitionsInt64, publicPartitions)
	groupedByValue := beam.CoGroupByKey(s, publicPartitionsWithZeros, beam.SwapKV(s, pcol.col))
	return beam.ParDo(s, mergePublicValues, groupedByValue)
}

// dropNonPublicPartitionsKVFn drops partitions not specified in
// PublicPartitions from pcol. It can be used for aggregations on <K,V> pairs,
// e.g. SumPerKey and MeanPerKey.
//
// We drop values that are not in the publicPartitions PCollection as follows:
//  1. Transform publicPartitions from <PartitionKey> to <PartitionKey, int64(0)> (0 is a placeholder value)
//  2. Transform pcol.col from <PrivacyKey, <PartitionKey, Value>> to <PartitionKey, <PrivacyKey, Value>>
//  3. Do a CoGroupByKey on the output of 1 and 2.
//  4. From the output of 3, only output <PartitionKey, <PrivacyKey, Value>> if there
//     is an input from 1 using mergePublicValues.
//  5. Transform output of 4 from <PartitionKey, <PrivacyKey, Value>> to <PrivacyKey, <PartitionKey, Value>>
//
// This works great for smaller partitions, but we run into performance bottlenecks in
// steps 3 & 4 in case some partitions have a huge number of user contributions.
//
// Returns a PCollection<PrivacyKey, <PartitionKey, Value>> only for values present in
// publicPartitions.
func dropNonPublicPartitionsKVFn(s beam.Scope, publicPartitions beam.PCollection, pcol PrivatePCollection, idType typex.FullType) beam.PCollection {
	publicPartitionsWithZeros := beam.ParDo(s, addZeroValuesToPublicPartitionsInt64, publicPartitions)
	encodedIDV := beam.ParDo(s, newEncodeIDVFn(idType, pcol.codec), pcol.col, beam.TypeDefinition{Var: beam.WType, T: pcol.codec.KType.T})
	groupedByValue := beam.CoGroupByKey(s, publicPartitionsWithZeros, encodedIDV)
	merged := beam.SwapKV(s, beam.ParDo(s, mergePublicValues, groupedByValue))
	decodeFn := newDecodeIDVFn(pcol.codec.KType, kv.NewCodec(idType.Type(), pcol.codec.VType.T))
	return beam.ParDo(s, decodeFn, merged, beam.TypeDefinition{Var: beam.UType, T: idType.Type()})
}

// encodeIDVFn takes a PCollection<ID,kv.Pair{K,V}> as input, and returns a
// PCollection<K, kv.Pair{ID,V}>; where ID and V have been coded, and K has been
// decoded.
type encodeIDVFn struct {
	IDType         beam.EncodedType    // Type information of the privacy ID
	idEnc          beam.ElementEncoder // Encoder for privacy ID, set during Setup() according to IDType
	InputPairCodec *kv.Codec           // Codec for the input kv.Pair{K,V}
}

func newEncodeIDVFn(idType typex.FullType, kvCodec *kv.Codec) *encodeIDVFn {
	return &encodeIDVFn{
		IDType:         beam.EncodedType{T: idType.Type()},
		InputPairCodec: kvCodec,
	}
}

func (fn *encodeIDVFn) Setup() error {
	fn.idEnc = beam.NewElementEncoder(fn.IDType.T)
	return fn.InputPairCodec.Setup()
}

func (fn *encodeIDVFn) ProcessElement(id beam.U, pair kv.Pair) (beam.W, kv.Pair, error) {
	var idBuf bytes.Buffer
	if err := fn.idEnc.Encode(id, &idBuf); err != nil {
		return nil, kv.Pair{}, fmt.Errorf("pbeam.encodeIDVFn.ProcessElement: couldn't encode ID %v: %w", id, err)
	}
	k, _, err := fn.InputPairCodec.Decode(pair)
	return k, kv.Pair{idBuf.Bytes(), pair.V}, err
}

// decodeIDVFn is the reverse operation of encodeIDVFn. It takes a PCollection<K, kv.Pair{ID,V}>
// as input, and returns a PCollection<ID, kv.Pair{K,V}>; where K and V has been coded, and ID
// has been decoded.
type decodeIDVFn struct {
	KType          beam.EncodedType    // Type information of the partition key K
	kEnc           beam.ElementEncoder // Encoder for partition key, set during Setup() according to KType
	InputPairCodec *kv.Codec           // Codec for the input kv.Pair{ID,V}
}

func newDecodeIDVFn(kType beam.EncodedType, idvCodec *kv.Codec) *decodeIDVFn {
	return &decodeIDVFn{
		KType:          kType,
		InputPairCodec: idvCodec,
	}
}

func (fn *decodeIDVFn) Setup() error {
	fn.kEnc = beam.NewElementEncoder(fn.KType.T)
	return fn.InputPairCodec.Setup()
}

func (fn *decodeIDVFn) ProcessElement(k beam.W, pair kv.Pair) (beam.U, kv.Pair, error) {
	var kBuf bytes.Buffer
	if err := fn.kEnc.Encode(k, &kBuf); err != nil {
		return nil, kv.Pair{}, fmt.Errorf("pbeam.decodeIDVFn.ProcessElement: couldn't encode K %v: %w", k, err)
	}
	id, _, err := fn.InputPairCodec.Decode(pair)
	return id, kv.Pair{kBuf.Bytes(), pair.V}, err // pair.V is the V in PCollection<K, kv.Pair{ID,V}>
}

// partitionMapFn makes a map consisting of public partitions.
type partitionMapFn struct {
	PartitionType beam.EncodedType
	partitionEnc  beam.ElementEncoder
}

func newPartitionMapFn(partitionType beam.EncodedType) *partitionMapFn {
	return &partitionMapFn{PartitionType: partitionType}
}

// Setup is our "constructor"
func (fn *partitionMapFn) Setup() {
	fn.partitionEnc = beam.NewElementEncoder(fn.PartitionType.T)
}

// CreateAccumulator creates a new accumulator for the appropriate data type
func (fn *partitionMapFn) CreateAccumulator() pMap {
	return make(pMap)
}

// AddInput adds the public partition key to the map
func (fn *partitionMapFn) AddInput(p pMap, partitionKey beam.W) (pMap, error) {
	var partitionBuf bytes.Buffer
	if err := fn.partitionEnc.Encode(partitionKey, &partitionBuf); err != nil {
		return p, fmt.Errorf("pbeam.PartitionsMapFn.AddInput: couldn't encode partition key %v: %w", partitionKey, err)
	}
	p[base64.StdEncoding.EncodeToString(partitionBuf.Bytes())] = true
	return p, nil
}

// MergeAccumulators adds the keys from a to b
func (fn *partitionMapFn) MergeAccumulators(a, b pMap) pMap {
	for k := range a {
		b[k] = true
	}
	return b
}

type prunePartitionsInMemoryVFn struct {
	PartitionType beam.EncodedType
	partitionEnc  beam.ElementEncoder
	PartitionMap  pMap
}

func newPrunePartitionsInMemoryVFn(partitionType beam.EncodedType, partitionMap pMap) *prunePartitionsInMemoryVFn {
	return &prunePartitionsInMemoryVFn{PartitionType: partitionType, PartitionMap: partitionMap}
}

func (fn *prunePartitionsInMemoryVFn) Setup() {
	fn.partitionEnc = beam.NewElementEncoder(fn.PartitionType.T)
}

func (fn *prunePartitionsInMemoryVFn) ProcessElement(id beam.U, partitionKey beam.W, emit func(beam.U, beam.W)) error {
	var partitionBuf bytes.Buffer
	if err := fn.partitionEnc.Encode(partitionKey, &partitionBuf); err != nil {
		return fmt.Errorf("pbeam.prunePartitionsInMemoryVFn.ProcessElement: couldn't encode partition %v: %w", partitionKey, err)
	}
	if fn.PartitionMap[base64.StdEncoding.EncodeToString(partitionBuf.Bytes())] {
		emit(id, partitionKey)
	}
	return nil
}

type prunePartitionsInMemoryKVFn struct {
	PartitionMap pMap
}

func newPrunePartitionsInMemoryKVFn(partitionMap pMap) *prunePartitionsInMemoryKVFn {
	return &prunePartitionsInMemoryKVFn{PartitionMap: partitionMap}
}

func (fn *prunePartitionsInMemoryKVFn) ProcessElement(id beam.U, pair kv.Pair, emit func(beam.U, kv.Pair)) {
	// Partition Key in a kv.Pair is already encoded, we just convert it to base64 encoding.
	if fn.PartitionMap[base64.StdEncoding.EncodeToString(pair.K)] {
		emit(id, pair)
	}
}

// prunePartitionsFn takes a PCollection<ID, kv.Pair{K,V}> as input, and returns a
// PCollection<ID, kv.Pair{K,V}>, where non-public partitions have been dropped.
// Used for sum and mean.
func prunePartitionsKV(id beam.U, pair kv.Pair, partitionsIter func(*pMap) bool, emit func(beam.U, kv.Pair)) error {
	var partitionMap pMap
	partitionsIter(&partitionMap)
	var err error
	if partitionMap == nil {
		return err
	}
	// Partition Key in a kv.Pair is already encoded, we just convert it to base64 encoding.
	if partitionMap[base64.StdEncoding.EncodeToString(pair.K)] {
		emit(id, pair)
	}
	return nil
}

func mergeResultWithEmptyPublicPartitionsFn(k beam.W, resultIter, publicPartitionsIter func(*beam.V) bool, emit func(beam.W, beam.V)) {
	var v beam.V
	if resultIter(&v) {
		emit(k, v)
	} else {
		if publicPartitionsIter(&v) {
			emit(k, v)
		}
	}
}

// checkPublicPartitions returns an error if publicPartitions parameter of an aggregation
// is not valid.
func checkPublicPartitions(publicPartitions any, partitionType reflect.Type) error {
	if publicPartitions != nil {
		if reflect.TypeOf(publicPartitions) != reflect.TypeOf(beam.PCollection{}) &&
			reflect.ValueOf(publicPartitions).Kind() != reflect.Slice &&
			reflect.ValueOf(publicPartitions).Kind() != reflect.Array {
			return fmt.Errorf("PublicPartitions=%+v needs to be a beam.PCollection, slice or array", reflect.TypeOf(publicPartitions))
		}
		publicPartitionsCol, isPCollection := publicPartitions.(beam.PCollection)
		if isPCollection && (!publicPartitionsCol.IsValid() || partitionType != publicPartitionsCol.Type().Type()) {
			return fmt.Errorf("PublicPartitions=%+v needs to be a valid beam.PCollection with the same type as the partition key (+%v)", publicPartitions, partitionType)
		}
		if !isPCollection && reflect.TypeOf(publicPartitions).Elem() != partitionType {
			return fmt.Errorf("PublicPartitions=%+v needs to be a slice or an array whose elements are the same type as the partition key (%+v)", publicPartitions, partitionType)
		}
	}
	return nil
}
