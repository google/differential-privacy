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

	"github.com/google/differential-privacy/privacy-on-beam/v2/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/core/typex"
)

func init() {
	beam.RegisterType(reflect.TypeOf((*partitionMapFn)(nil)).Elem())
	beam.RegisterType(reflect.TypeOf((*prunePartitionsInMemoryVFn)(nil)).Elem())
	beam.RegisterType(reflect.TypeOf((*prunePartitionsInMemoryKVFn)(nil)).Elem())
	beam.RegisterType(reflect.TypeOf((*emitPartitionsNotInTheDataFn)(nil)).Elem())

	beam.RegisterFunction(addZeroValuesToPublicPartitionsInt64Fn)
	beam.RegisterFunction(addZeroValuesToPublicPartitionsFloat64Fn)
	beam.RegisterFunction(addEmptySliceToPublicPartitionsFloat64Fn)
	beam.RegisterFunction(prunePartitionsKVFn)
	beam.RegisterFunction(mergePublicValuesFn)
}

// newAddZeroValuesToPublicPartitionsFn turns a PCollection<V> into PCollection<V,0>.
func newAddZeroValuesToPublicPartitionsFn(vKind reflect.Kind) (any, error) {
	switch vKind {
	case reflect.Int64:
		return addZeroValuesToPublicPartitionsInt64Fn, nil
	case reflect.Float64:
		return addZeroValuesToPublicPartitionsFloat64Fn, nil
	default:
		return nil, fmt.Errorf("vKind(%v) should be int64 or float64", vKind)
	}
}

func addZeroValuesToPublicPartitionsInt64Fn(partition beam.X) (k beam.X, v int64) {
	return partition, 0
}

func addZeroValuesToPublicPartitionsFloat64Fn(partition beam.X) (k beam.X, v float64) {
	return partition, 0
}

func addEmptySliceToPublicPartitionsFloat64Fn(partition beam.X) (k beam.X, v []float64) {
	return partition, []float64{}
}

// pMap holds a set of partition keys for quick lookup as a map from string to bool.
// Key is the base64 string representation of encoded partition key.
// Value is set to true if partition key exists.
type pMap map[string]bool

// dropNonPublicPartitions returns the PCollection with the non-public partitions dropped if public partitions are
// specified. Returns the input PCollection otherwise.
func dropNonPublicPartitions(s beam.Scope, pcol PrivatePCollection, publicPartitions any, partitionType reflect.Type) (beam.PCollection, error) {
	// If PublicPartitions is not specified, return the input collection.
	if publicPartitions == nil {
		return pcol.col, nil
	}

	// Drop non-public partitions, if public partitions are specified as a PCollection.
	if publicPartitionscCol, ok := publicPartitions.(beam.PCollection); ok {
		partitionEncodedType := beam.EncodedType{partitionType}
		// Data is <PrivacyKey, PartitionKey, Value>
		if pcol.codec != nil {
			return dropNonPublicPartitionsKVFn(s, publicPartitionscCol, pcol, partitionEncodedType), nil
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

// dropNonPublicPartitionsKVFn drops partitions not specified in PublicPartitions from pcol. It can be used for aggregations on <K,V> pairs, e.g. sum and mean.
func dropNonPublicPartitionsKVFn(s beam.Scope, publicPartitions beam.PCollection, pcol PrivatePCollection, partitionEncodedType beam.EncodedType) beam.PCollection {
	partitionMap := beam.Combine(s, newPartitionMapFn(partitionEncodedType), publicPartitions)
	return beam.ParDo(s, prunePartitionsKVFn, pcol.col, beam.SideInput{Input: partitionMap})
}

// mergePublicValuesFn merges the public partitions with the values for Count
// and DistinctPrivacyId after a CoGroupByKey. Only outputs a <privacyKey,
// value> pair if the value is in the public partitions, i.e., the PCollection
// that is passed to the CoGroupByKey first.
func mergePublicValuesFn(value beam.X, isKnown func(*int64) bool, privacyKeys func(*beam.W) bool, emit func(beam.W, beam.X)) {
	var ignoredZero int64
	if isKnown(&ignoredZero) {
		var privacyKey beam.W
		for privacyKeys(&privacyKey) {
			emit(privacyKey, value)
		}
	}
}

// dropNonPublicPartitionsVFn drops partitions not specified in
// PublicPartitions from pcol. It can be used for aggregations on V values,
// e.g. count and distinctid.
//
// We drop values that are not in the publicPartitions PCollection as follows:
//  1. Transform publicPartitions from <V> to <V, int64(0)> (0 is a placeholder value)
//  2. Swap pcol.col from <PrivacyKey, V> to <V, PrivacyKey>
//  3. Do a CoGroupByKey on the output of 1 and 2.
//  4. From the output of 3, only output <PrivacyKey, V> if there is an input
//     from 1 using the mergePublicValuesFn.
//
// Returns a PCollection<PrivacyKey, Value> only for values present in
// publicPartitions.
func dropNonPublicPartitionsVFn(s beam.Scope, publicPartitions beam.PCollection, pcol PrivatePCollection) beam.PCollection {
	publicPartitionsWithZeros := beam.ParDo(s, addZeroValuesToPublicPartitionsInt64Fn, publicPartitions)
	groupedByValue := beam.CoGroupByKey(s, publicPartitionsWithZeros, beam.SwapKV(s, pcol.col))
	return beam.ParDo(s, mergePublicValuesFn, groupedByValue)
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
func (fn *partitionMapFn) AddInput(p pMap, partitionKey beam.X) (pMap, error) {
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

// ExtractOutput returns the completed partition map
func (fn *partitionMapFn) ExtractOutput(p pMap) pMap {
	return p
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

func (fn *prunePartitionsInMemoryVFn) ProcessElement(id beam.X, partitionKey beam.V, emit func(beam.X, beam.V)) error {
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

func (fn *prunePartitionsInMemoryKVFn) ProcessElement(id beam.X, pair kv.Pair, emit func(beam.X, kv.Pair)) {
	// Partition Key in a kv.Pair is already encoded, we just convert it to base64 encoding.
	if fn.PartitionMap[base64.StdEncoding.EncodeToString(pair.K)] {
		emit(id, pair)
	}
}

// prunePartitionsFn takes a PCollection<ID, kv.Pair{K,V}> as input, and returns a
// PCollection<ID, kv.Pair{K,V}>, where non-public partitions have been dropped.
// Used for sum and mean.
func prunePartitionsKVFn(id beam.X, pair kv.Pair, partitionsIter func(*pMap) bool, emit func(beam.X, kv.Pair)) error {
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

// emitPartitionsNotInTheDataFn emits partitions that are public but not found in the data.
type emitPartitionsNotInTheDataFn struct {
	PartitionType beam.EncodedType
	partitionEnc  beam.ElementEncoder
}

func newEmitPartitionsNotInTheDataFn(partitionType typex.FullType) *emitPartitionsNotInTheDataFn {
	return &emitPartitionsNotInTheDataFn{
		PartitionType: beam.EncodedType{partitionType.Type()},
	}
}

func (fn *emitPartitionsNotInTheDataFn) Setup() {
	fn.partitionEnc = beam.NewElementEncoder(fn.PartitionType.T)
}

func (fn *emitPartitionsNotInTheDataFn) ProcessElement(partitionKey beam.X, value beam.V, partitionsIter func(*pMap) bool, emit func(beam.X, beam.V)) error {
	var partitionBuf bytes.Buffer
	if err := fn.partitionEnc.Encode(partitionKey, &partitionBuf); err != nil {
		return fmt.Errorf("pbeam.emitPartitionsNotInTheDataFn.ProcessElement: couldn't encode partition %v: %w", partitionKey, err)
	}
	var partitionsInDataMap pMap
	partitionsIter(&partitionsInDataMap)
	// If partitionsInDataMap is nil, partitionsInDataMap is empty, so none of the partitions are in the data, which means we need to emit all of them.
	// Similarly, if a partition is not in partitionsInDataMap, it means that the partition is not in the data, so we need to emit it.
	//
	// Partition Key in a kv.Pair is already encoded, we just convert it to base64 encoding.
	if partitionsInDataMap == nil || !partitionsInDataMap[base64.StdEncoding.EncodeToString(partitionBuf.Bytes())] {
		emit(partitionKey, value)
	}
	return nil
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
