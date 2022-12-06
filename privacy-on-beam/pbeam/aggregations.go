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

// This file contains methods & ParDos used by multiple DP aggregations.

package pbeam

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"reflect"

	"github.com/google/differential-privacy/go/v2/checks"
	"github.com/google/differential-privacy/go/v2/dpagg"
	"github.com/google/differential-privacy/go/v2/noise"
	"github.com/google/differential-privacy/privacy-on-beam/v2/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/core/typex"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/top"
)

type pMap map[string]bool

func init() {
	beam.RegisterType(reflect.TypeOf((*boundedSumInt64Fn)(nil)))
	beam.RegisterType(reflect.TypeOf((*boundedSumFloat64Fn)(nil)))
	beam.RegisterType(reflect.TypeOf((*decodePairInt64Fn)(nil)))
	beam.RegisterType(reflect.TypeOf((*decodePairFloat64Fn)(nil)))
	beam.RegisterType(reflect.TypeOf((*dropValuesFn)(nil)))
	beam.RegisterType(reflect.TypeOf((*encodeKVFn)(nil)))
	beam.RegisterType(reflect.TypeOf((*encodeIDKFn)(nil)))
	beam.RegisterType(reflect.TypeOf((*decodeIDKFn)(nil)))
	beam.RegisterType(reflect.TypeOf((*expandValuesCombineFn)(nil)))
	beam.RegisterType(reflect.TypeOf((*expandFloat64ValuesCombineFn)(nil)))
	beam.RegisterType(reflect.TypeOf((*decodePairArrayFloat64Fn)(nil)))
	beam.RegisterType(reflect.TypeOf((*partitionsMapFn)(nil)).Elem())
	beam.RegisterType(reflect.TypeOf((*prunePartitionsInMemoryVFn)(nil)).Elem())
	beam.RegisterType(reflect.TypeOf((*prunePartitionsInMemoryKVFn)(nil)).Elem())
	beam.RegisterType(reflect.TypeOf((*pMap)(nil)).Elem())
	beam.RegisterType(reflect.TypeOf((*emitPartitionsNotInTheDataFn)(nil)).Elem())

	beam.RegisterFunction(randBool)
	beam.RegisterFunction(clampNegativePartitionsInt64Fn)
	beam.RegisterFunction(clampNegativePartitionsFloat64Fn)
	beam.RegisterFunction(addZeroValuesToPublicPartitionsInt64Fn)
	beam.RegisterFunction(addZeroValuesToPublicPartitionsFloat64Fn)
	beam.RegisterFunction(addEmptySliceToPublicPartitionsFloat64Fn)
	beam.RegisterFunction(dropThresholdedPartitionsInt64Fn)
	beam.RegisterFunction(dropThresholdedPartitionsFloat64Fn)
	beam.RegisterFunction(dropThresholdedPartitionsFloat64SliceFn)
	beam.RegisterFunction(dereferenceValueToInt64Fn)
	beam.RegisterFunction(dereferenceValueToFloat64Fn)
	beam.RegisterFunction(prunePartitionsKVFn)
	beam.RegisterFunction(mergePublicValuesFn)
	// TODO: add tests to make sure we don't forget anything here
}

// randBool returns a uniformly random boolean. The randomness used here is not
// cryptographically secure, and using this with top.LargestPerKey doesn't
// necessarily result in a uniformly random permutation: the distribution of
// the permutation depends on the exact sorting algorithm used by Beam and the
// order in which the input values are processed within the pipeline.
//
// The fact that the resulting permutation is not necessarily uniformly random is
// not a problem, since all we require from this function to satisfy DP properties
// is that it doesn't depend on the data. More specifically, in order to satisfy DP
// properties, a privacy unit's data should not influence another privacy unit's
// permutation of contributions. We assume that the order Beam processes the
// input values for a privacy unit is independent of other privacy units'
// inputs, in which case this requirement is satisfied.
func randBool(_, _ beam.V) bool {
	return rand.Uint32()%2 == 0
}

// boundContributions takes a PCollection<K,V> as input, and for each key, selects and returns
// at most contributionLimit records with this key. The selection is "mostly random":
// the records returned are selected randomly, but the randomness isn't secure.
// This is fine to use in the cross-partition bounding stage or in the per-partition bounding stage,
// since the privacy guarantee doesn't depend on the privacy unit contributions being selected randomly.
//
// In order to do the cross-partition contribution bounding we need:
//  1. the key to be the privacy ID.
//  2. the value to be the partition ID or the pair = {partition ID, aggregated statistic},
//     where aggregated statistic is either array of values which are associated with the given id
//     and partition, or sum/count/etc of these values.
//
// In order to do the per-partition contribution bounding we need:
//  1. the key to be the pair = {privacy ID, partition ID}.
//  2. the value to be just the value which is associated with that {privacy ID, partition ID} pair
//     (there could be multiple entries with the same key).
func boundContributions(s beam.Scope, kvCol beam.PCollection, contributionLimit int64) beam.PCollection {
	s = s.Scope("boundContributions")
	// Transform the PCollection<K,V> into a PCollection<K,[]V>, where
	// there are at most contributionLimit elements per slice, chosen randomly. To
	// do that, the easiest solution seems to be to use the LargestPerKey
	// function (that returns the contributionLimit "largest" elements), except
	// the function used to sort elements is random.
	sampled := top.LargestPerKey(s, kvCol, int(contributionLimit), randBool)
	// Flatten the values for each key to get back a PCollection<K,V>.
	return beam.ParDo(s, flattenValuesFn, sampled)
}

// Given a PCollection<K,[]V>, flattens the second argument to return a PCollection<K,V>.
func flattenValuesFn(key beam.T, values []beam.V, emit func(beam.T, beam.V)) {
	for _, v := range values {
		emit(key, v)
	}
}

// vToInt64Fn converts the second element of a KV<K,int> pair to an int64.
func vToInt64Fn(k beam.T, v int) (beam.T, int64) {
	return k, int64(v)
}

func findRekeyFn(kind reflect.Kind) (any, error) {
	switch kind {
	case reflect.Int64:
		return rekeyInt64Fn, nil
	case reflect.Float64:
		return rekeyFloat64Fn, nil
	default:
		return nil, fmt.Errorf("kind(%v) should be int64 or float64", kind)
	}
}

// pairInt64 contains an encoded value and an int64 metric.
type pairInt64 struct {
	X []byte
	M int64
}

// rekeyInt64Fn transforms a PCollection<kv.Pair<codedK,codedV>,int64> into a
// PCollection<codedK,pairInt64<codedV,int>>.
func rekeyInt64Fn(kv kv.Pair, m int64) ([]byte, pairInt64) {
	return kv.K, pairInt64{kv.V, m}
}

// pairFloat64 contains an encoded value and an float64 metric.
type pairFloat64 struct {
	X []byte
	M float64
}

// rekeyFloat64Fn transforms a PCollection<kv.Pair<codedK,codedV>,float64> into a
// PCollection<codedK,pairFloat64<codedV,int>>.
func rekeyFloat64Fn(kv kv.Pair, m float64) ([]byte, pairFloat64) {
	return kv.K, pairFloat64{kv.V, m}
}

func newDecodePairFn(t reflect.Type, kind reflect.Kind) (any, error) {
	switch kind {
	case reflect.Int64:
		return newDecodePairInt64Fn(t), nil
	case reflect.Float64:
		return newDecodePairFloat64Fn(t), nil
	default:
		return nil, fmt.Errorf("kind(%v) should be int64 or float64", kind)
	}
}

// decodePairInt64Fn transforms a PCollection<pairInt64<codedX,int64>> into a
// PCollection<X,int64>.
type decodePairInt64Fn struct {
	XType beam.EncodedType
	xDec  beam.ElementDecoder
}

func newDecodePairInt64Fn(t reflect.Type) *decodePairInt64Fn {
	return &decodePairInt64Fn{XType: beam.EncodedType{t}}
}

func (fn *decodePairInt64Fn) Setup() {
	fn.xDec = beam.NewElementDecoder(fn.XType.T)
}

func (fn *decodePairInt64Fn) ProcessElement(pair pairInt64) (beam.X, int64, error) {
	x, err := fn.xDec.Decode(bytes.NewBuffer(pair.X))
	if err != nil {
		return nil, 0, fmt.Errorf("pbeam.decodePairInt64Fn.ProcessElement: couldn't decode pair %v: %w", pair, err)
	}
	return x, pair.M, nil
}

// decodePairFloat64Fn transforms a PCollection<pairFloat64<codedX,float64>> into a
// PCollection<X,float64>.
type decodePairFloat64Fn struct {
	XType beam.EncodedType
	xDec  beam.ElementDecoder
}

func newDecodePairFloat64Fn(t reflect.Type) *decodePairFloat64Fn {
	return &decodePairFloat64Fn{XType: beam.EncodedType{t}}
}

func (fn *decodePairFloat64Fn) Setup() {
	fn.xDec = beam.NewElementDecoder(fn.XType.T)
}

func (fn *decodePairFloat64Fn) ProcessElement(pair pairFloat64) (beam.X, float64, error) {
	x, err := fn.xDec.Decode(bytes.NewBuffer(pair.X))
	if err != nil {
		return nil, 0.0, fmt.Errorf("pbeam.decodePairFloat64Fn.ProcessElement: couldn't decode pair %v: %w", pair, err)
	}
	return x, pair.M, nil
}

func newBoundedSumFn(epsilon, delta float64, maxPartitionsContributed int64, lower, upper float64, noiseKind noise.Kind, vKind reflect.Kind, publicPartitions bool, testMode testMode) (any, error) {
	var err, checkErr error
	var bsFn any

	switch vKind {
	case reflect.Int64:
		checkErr = checks.CheckBoundsFloat64AsInt64(lower, upper)
		if checkErr != nil {
			return nil, checkErr
		}
		bsFn, err = newBoundedSumInt64Fn(epsilon, delta, maxPartitionsContributed, int64(lower), int64(upper), noiseKind, publicPartitions, testMode)
	case reflect.Float64:
		checkErr = checks.CheckBoundsFloat64(lower, upper)
		if checkErr != nil {
			return nil, checkErr
		}
		bsFn, err = newBoundedSumFloat64Fn(epsilon, delta, maxPartitionsContributed, lower, upper, noiseKind, publicPartitions, testMode)
	default:
		err = fmt.Errorf("vKind(%v) should be int64 or float64", vKind)
	}

	return bsFn, err
}

type boundedSumAccumInt64 struct {
	BS               *dpagg.BoundedSumInt64
	SP               *dpagg.PreAggSelectPartition
	PublicPartitions bool
}

// boundedSumInt64Fn is a differentially private combineFn for summing values. Do not
// initialize it yourself, use newBoundedSumInt64Fn to create a boundedSumInt64Fn instance.
type boundedSumInt64Fn struct {
	// Privacy spec parameters (set during initial construction).
	NoiseEpsilon              float64
	PartitionSelectionEpsilon float64
	NoiseDelta                float64
	PartitionSelectionDelta   float64
	MaxPartitionsContributed  int64
	Lower                     int64
	Upper                     int64
	NoiseKind                 noise.Kind
	noise                     noise.Noise // Set during Setup phase according to NoiseKind.
	PublicPartitions          bool
	TestMode                  testMode
}

// newBoundedSumInt64Fn returns a boundedSumInt64Fn with the given budget and parameters.
func newBoundedSumInt64Fn(epsilon, delta float64, maxPartitionsContributed, lower, upper int64, noiseKind noise.Kind, publicPartitions bool, testMode testMode) (*boundedSumInt64Fn, error) {
	fn := &boundedSumInt64Fn{
		MaxPartitionsContributed: maxPartitionsContributed,
		Lower:                    lower,
		Upper:                    upper,
		NoiseKind:                noiseKind,
		PublicPartitions:         publicPartitions,
		TestMode:                 testMode,
	}
	if fn.PublicPartitions {
		fn.NoiseEpsilon = epsilon
		fn.NoiseDelta = delta
		return fn, nil
	}
	fn.NoiseEpsilon = epsilon / 2
	fn.PartitionSelectionEpsilon = epsilon - fn.NoiseEpsilon
	switch noiseKind {
	case noise.GaussianNoise:
		fn.NoiseDelta = delta / 2
	case noise.LaplaceNoise:
		fn.NoiseDelta = 0
	default:
		return nil, fmt.Errorf("unknown noise.Kind (%v) is specified. Please specify a valid noise", noiseKind)
	}
	fn.PartitionSelectionDelta = delta - fn.NoiseDelta
	return fn, nil
}

func (fn *boundedSumInt64Fn) Setup() {
	fn.noise = noise.ToNoise(fn.NoiseKind)
	if fn.TestMode.isEnabled() {
		fn.noise = noNoise{}
	}
}

func (fn *boundedSumInt64Fn) CreateAccumulator() (boundedSumAccumInt64, error) {
	if fn.TestMode == noNoiseWithoutContributionBounding {
		fn.Lower = math.MinInt64
		fn.Upper = math.MaxInt64
	}
	var bs *dpagg.BoundedSumInt64
	var err error
	bs, err = dpagg.NewBoundedSumInt64(&dpagg.BoundedSumInt64Options{
		Epsilon:                  fn.NoiseEpsilon,
		Delta:                    fn.NoiseDelta,
		MaxPartitionsContributed: fn.MaxPartitionsContributed,
		Lower:                    fn.Lower,
		Upper:                    fn.Upper,
		Noise:                    fn.noise,
	})
	if err != nil {
		return boundedSumAccumInt64{}, err
	}
	accum := boundedSumAccumInt64{BS: bs, PublicPartitions: fn.PublicPartitions}
	if !fn.PublicPartitions {
		accum.SP, err = dpagg.NewPreAggSelectPartition(&dpagg.PreAggSelectPartitionOptions{
			Epsilon:                  fn.PartitionSelectionEpsilon,
			Delta:                    fn.PartitionSelectionDelta,
			MaxPartitionsContributed: fn.MaxPartitionsContributed,
		})
	}
	return accum, err
}

func (fn *boundedSumInt64Fn) AddInput(a boundedSumAccumInt64, value int64) (boundedSumAccumInt64, error) {
	err := a.BS.Add(value)
	if err != nil {
		return a, err
	}
	if !fn.PublicPartitions {
		err := a.SP.Increment()
		if err != nil {
			return a, err
		}
	}
	return a, nil
}

func (fn *boundedSumInt64Fn) MergeAccumulators(a, b boundedSumAccumInt64) (boundedSumAccumInt64, error) {
	err := a.BS.Merge(b.BS)
	if err != nil {
		return a, err
	}
	if !fn.PublicPartitions {
		err := a.SP.Merge(b.SP)
		if err != nil {
			return a, err
		}
	}
	return a, nil
}

func (fn *boundedSumInt64Fn) ExtractOutput(a boundedSumAccumInt64) (*int64, error) {
	if fn.TestMode.isEnabled() {
		a.BS.Noise = noNoise{}
	}
	var err error
	shouldKeepPartition := fn.TestMode.isEnabled() || a.PublicPartitions // If in test mode or public partitions are specified, we always keep the partition.
	if !shouldKeepPartition {                                            // If not, we need to perform private partition selection.
		shouldKeepPartition, err = a.SP.ShouldKeepPartition()
		if err != nil {
			return nil, err
		}
	}

	if shouldKeepPartition {
		result, err := a.BS.Result()
		return &result, err
	}
	return nil, nil
}

func (fn *boundedSumInt64Fn) String() string {
	return fmt.Sprintf("%#v", fn)
}

type boundedSumAccumFloat64 struct {
	BS               *dpagg.BoundedSumFloat64
	SP               *dpagg.PreAggSelectPartition
	PublicPartitions bool
}

// boundedSumFloat64Fn is a differentially private combineFn for summing values. Do not
// initialize it yourself, use newBoundedSumFloat64Fn to create a boundedSumFloat64Fn instance.
type boundedSumFloat64Fn struct {
	// Privacy spec parameters (set during initial construction).
	NoiseEpsilon              float64
	PartitionSelectionEpsilon float64
	NoiseDelta                float64
	PartitionSelectionDelta   float64
	MaxPartitionsContributed  int64
	Lower                     float64
	Upper                     float64
	NoiseKind                 noise.Kind
	// Noise, set during Setup phase according to NoiseKind.
	noise            noise.Noise
	PublicPartitions bool
	TestMode         testMode
}

// newBoundedSumFloat64Fn returns a boundedSumFloat64Fn with the given budget and parameters.
func newBoundedSumFloat64Fn(epsilon, delta float64, maxPartitionsContributed int64, lower, upper float64, noiseKind noise.Kind, publicPartitions bool, testMode testMode) (*boundedSumFloat64Fn, error) {
	fn := &boundedSumFloat64Fn{
		MaxPartitionsContributed: maxPartitionsContributed,
		Lower:                    lower,
		Upper:                    upper,
		NoiseKind:                noiseKind,
		PublicPartitions:         publicPartitions,
		TestMode:                 testMode,
	}
	if fn.PublicPartitions {
		fn.NoiseEpsilon = epsilon
		fn.NoiseDelta = delta
		return fn, nil
	}
	fn.NoiseEpsilon = epsilon / 2
	fn.PartitionSelectionEpsilon = epsilon - fn.NoiseEpsilon
	switch noiseKind {
	case noise.GaussianNoise:
		fn.NoiseDelta = delta / 2
	case noise.LaplaceNoise:
		fn.NoiseDelta = 0
	default:
		return nil, fmt.Errorf("unknown noise.Kind (%v) is specified. Please specify a valid noise", noiseKind)
	}
	fn.PartitionSelectionDelta = delta - fn.NoiseDelta
	return fn, nil
}

func (fn *boundedSumFloat64Fn) Setup() {
	fn.noise = noise.ToNoise(fn.NoiseKind)
	if fn.TestMode.isEnabled() {
		fn.noise = noNoise{}
	}
}

func (fn *boundedSumFloat64Fn) CreateAccumulator() (boundedSumAccumFloat64, error) {
	if fn.TestMode == noNoiseWithoutContributionBounding {
		fn.Lower = math.Inf(-1)
		fn.Upper = math.Inf(1)
	}
	var bs *dpagg.BoundedSumFloat64
	var err error
	bs, err = dpagg.NewBoundedSumFloat64(&dpagg.BoundedSumFloat64Options{
		Epsilon:                  fn.NoiseEpsilon,
		Delta:                    fn.NoiseDelta,
		MaxPartitionsContributed: fn.MaxPartitionsContributed,
		Lower:                    fn.Lower,
		Upper:                    fn.Upper,
		Noise:                    fn.noise,
	})
	if err != nil {
		return boundedSumAccumFloat64{}, err
	}
	accum := boundedSumAccumFloat64{BS: bs, PublicPartitions: fn.PublicPartitions}
	if !fn.PublicPartitions {
		accum.SP, err = dpagg.NewPreAggSelectPartition(&dpagg.PreAggSelectPartitionOptions{
			Epsilon:                  fn.PartitionSelectionEpsilon,
			Delta:                    fn.PartitionSelectionDelta,
			MaxPartitionsContributed: fn.MaxPartitionsContributed,
		})
	}
	return accum, err
}

func (fn *boundedSumFloat64Fn) AddInput(a boundedSumAccumFloat64, value float64) (boundedSumAccumFloat64, error) {
	var err error
	err = a.BS.Add(value)
	if err != nil {
		return a, err
	}
	if !fn.PublicPartitions {
		err = a.SP.Increment()
	}
	return a, err
}

func (fn *boundedSumFloat64Fn) MergeAccumulators(a, b boundedSumAccumFloat64) (boundedSumAccumFloat64, error) {
	var err error
	err = a.BS.Merge(b.BS)
	if err != nil {
		return a, err
	}
	if !fn.PublicPartitions {
		err = a.SP.Merge(b.SP)
	}
	return a, err
}

func (fn *boundedSumFloat64Fn) ExtractOutput(a boundedSumAccumFloat64) (*float64, error) {
	if fn.TestMode.isEnabled() {
		a.BS.Noise = noNoise{}
	}
	var err error
	shouldKeepPartition := fn.TestMode.isEnabled() || a.PublicPartitions // If in test mode or public partitions are specified, we always keep the partition.
	if !shouldKeepPartition {                                            // If not, we need to perform private partition selection.
		shouldKeepPartition, err = a.SP.ShouldKeepPartition()
		if err != nil {
			return nil, err
		}
	}

	if shouldKeepPartition {
		result, err := a.BS.Result()
		return &result, err
	}
	return nil, nil
}

// findDereferenceValueFn dereferences a *int64 to int64 or *float64 to float64.
func findDereferenceValueFn(kind reflect.Kind) (any, error) {
	switch kind {
	case reflect.Int64:
		return dereferenceValueToInt64Fn, nil
	case reflect.Float64:
		return dereferenceValueToFloat64Fn, nil
	default:
		return nil, fmt.Errorf("kind(%v) should be int64 or float64", kind)
	}
}

func dereferenceValueToInt64Fn(key beam.X, value *int64) (k beam.X, v int64) {
	return key, *value
}

func dereferenceValueToFloat64Fn(key beam.X, value *float64) (k beam.X, v float64) {
	return key, *value
}

func (fn *boundedSumFloat64Fn) String() string {
	return fmt.Sprintf("%#v", fn)
}

func findDropThresholdedPartitionsFn(kind reflect.Kind) (any, error) {
	switch kind {
	case reflect.Int64:
		return dropThresholdedPartitionsInt64Fn, nil
	case reflect.Float64:
		return dropThresholdedPartitionsFloat64Fn, nil
	default:
		return nil, fmt.Errorf("kind(%v) should be int64 or float64", kind)
	}
}

// dropThresholdedPartitionsInt64Fn drops thresholded int partitions, i.e. those
// that have nil r, by emitting only non-thresholded partitions.
func dropThresholdedPartitionsInt64Fn(v beam.V, r *int64, emit func(beam.V, int64)) {
	if r != nil {
		emit(v, *r)
	}
}

// dropThresholdedPartitionsFloat64Fn drops thresholded float partitions, i.e. those
// that have nil r, by emitting only non-thresholded partitions.
func dropThresholdedPartitionsFloat64Fn(v beam.V, r *float64, emit func(beam.V, float64)) {
	if r != nil {
		emit(v, *r)
	}
}

// dropThresholdedPartitionsFloat64SliceFn drops thresholded []float64 partitions, i.e.
// those that have nil r, by emitting only non-thresholded partitions.
func dropThresholdedPartitionsFloat64SliceFn(v beam.V, r []float64, emit func(beam.V, []float64)) {
	if r != nil {
		emit(v, r)
	}
}

func findClampNegativePartitionsFn(kind reflect.Kind) (any, error) {
	switch kind {
	case reflect.Int64:
		return clampNegativePartitionsInt64Fn, nil
	case reflect.Float64:
		return clampNegativePartitionsFloat64Fn, nil
	default:
		return nil, fmt.Errorf("kind(%v) should be int64 or float64", kind)
	}
}

// Clamp negative partitions to zero for int64 partitions, e.g., as a post aggregation step for Count.
func clampNegativePartitionsInt64Fn(v beam.V, r int64) (beam.V, int64) {
	if r < 0 {
		return v, 0
	}
	return v, r
}

// Clamp negative partitions to zero for float64 partitions.
func clampNegativePartitionsFloat64Fn(v beam.V, r float64) (beam.V, float64) {
	if r < 0 {
		return v, 0
	}
	return v, r
}

func convertFloat32ToFloat64Fn(z beam.Z, f float32) (beam.Z, float64) {
	return z, float64(f)
}

func convertFloat64ToFloat64Fn(z beam.Z, f float64) (beam.Z, float64) {
	return z, f
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
		return dropNonPublicPartitionsVFn(s, publicPartitionscCol, pcol, partitionEncodedType), nil
	}

	// Drop non-public partitions, public partitions are specified as slice/array (i.e., in-memory).
	// Convert PublicPartitions to map[byte]bool for quick lookup.
	partitionEnc := beam.NewElementEncoder(partitionType)
	partitionMap := map[string]bool{}
	for i := 0; i < reflect.ValueOf(publicPartitions).Len(); i++ {
		partitionKey := reflect.ValueOf(publicPartitions).Index(i).Interface()
		var partitionBuf bytes.Buffer
		if err := partitionEnc.Encode(partitionKey, &partitionBuf); err != nil {
			return pcol.col, fmt.Errorf("couldn't encode partition %v: %v", partitionKey, err)
		}
		partitionMap[string(partitionBuf.Bytes())] = true
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
	partitionMap := beam.Combine(s, newPartitionsMapFn(partitionEncodedType), publicPartitions)
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
func dropNonPublicPartitionsVFn(s beam.Scope, publicPartitions beam.PCollection, pcol PrivatePCollection, partitionEncodedType beam.EncodedType) beam.PCollection {
	publicPartitionsWithZeros := beam.ParDo(s, addZeroValuesToPublicPartitionsInt64Fn, publicPartitions)
	groupedByValue := beam.CoGroupByKey(s, publicPartitionsWithZeros, beam.SwapKV(s, pcol.col))
	return beam.ParDo(s, mergePublicValuesFn, groupedByValue)
}

type mapAccum struct {
	// Key is the string representation of encoded partition key.
	// Value is always set to true.
	PartitionMap pMap
}

// partitionsMapFn makes a map consisting of public partitions.
type partitionsMapFn struct {
	PartitionType beam.EncodedType
	partitionEnc  beam.ElementEncoder
}

func newPartitionsMapFn(partitionType beam.EncodedType) *partitionsMapFn {
	return &partitionsMapFn{PartitionType: partitionType}
}

// Setup is our "constructor"
func (fn *partitionsMapFn) Setup() {
	fn.partitionEnc = beam.NewElementEncoder(fn.PartitionType.T)
}

// CreateAccumulator creates a new accumulator for the appropriate data type
func (fn *partitionsMapFn) CreateAccumulator() mapAccum {
	return mapAccum{PartitionMap: make(pMap)}
}

// AddInput adds the public partition key to the map
func (fn *partitionsMapFn) AddInput(m mapAccum, partitionKey beam.X) (mapAccum, error) {
	var partitionBuf bytes.Buffer
	if err := fn.partitionEnc.Encode(partitionKey, &partitionBuf); err != nil {
		return m, fmt.Errorf("pbeam.PartitionsMapFn.AddInput: couldn't encode partition key %v: %w", partitionKey, err)
	}
	m.PartitionMap[string(partitionBuf.Bytes())] = true
	return m, nil
}

// MergeAccumulators adds the keys from a to b
func (fn *partitionsMapFn) MergeAccumulators(a, b mapAccum) mapAccum {
	for k := range a.PartitionMap {
		b.PartitionMap[k] = true
	}
	return b
}

// ExtractOutput returns the completed partition map
func (fn *partitionsMapFn) ExtractOutput(m mapAccum) pMap {
	return m.PartitionMap
}

type prunePartitionsInMemoryVFn struct {
	PartitionType beam.EncodedType
	partitionEnc  beam.ElementEncoder
	PartitionMap  map[string]bool
}

func newPrunePartitionsInMemoryVFn(partitionType beam.EncodedType, partitionMap map[string]bool) *prunePartitionsInMemoryVFn {
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

	if fn.PartitionMap[string(partitionBuf.Bytes())] {
		emit(id, partitionKey)
	}
	return nil
}

type prunePartitionsInMemoryKVFn struct {
	PartitionMap map[string]bool
}

func newPrunePartitionsInMemoryKVFn(partitionMap map[string]bool) *prunePartitionsInMemoryKVFn {
	return &prunePartitionsInMemoryKVFn{PartitionMap: partitionMap}
}

func (fn *prunePartitionsInMemoryKVFn) ProcessElement(id beam.X, pair kv.Pair, emit func(beam.X, kv.Pair)) {
	// Parameters in a kv.Pair are already encoded.
	if fn.PartitionMap[string(pair.K)] {
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
	// Parameters in a kv.Pair are already encoded.
	if partitionMap[string(pair.K)] {
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
	if partitionsInDataMap == nil || !partitionsInDataMap[string(partitionBuf.Bytes())] {
		emit(partitionKey, value)
	}
	return nil
}

type dropValuesFn struct {
	Codec *kv.Codec
}

func (fn *dropValuesFn) Setup() {
	fn.Codec.Setup()
}

func (fn *dropValuesFn) ProcessElement(id beam.Z, kv kv.Pair) (beam.Z, beam.W, error) {
	k, _, err := fn.Codec.Decode(kv)
	return id, k, err
}

// encodeKVFn takes a PCollection<kv.Pair{ID,K}, codedV> as input, and returns a
// PCollection<ID, kv.Pair{K,V}>; where K and V have been coded, and ID has been
// decoded.
type encodeKVFn struct {
	InputPairCodec *kv.Codec // Codec for the input kv.Pair{ID,K}
}

func newEncodeKVFn(idkCodec *kv.Codec) *encodeKVFn {
	return &encodeKVFn{InputPairCodec: idkCodec}
}

func (fn *encodeKVFn) Setup() error {
	return fn.InputPairCodec.Setup()
}

func (fn *encodeKVFn) ProcessElement(pair kv.Pair, codedV []byte) (beam.W, kv.Pair, error) {
	id, _, err := fn.InputPairCodec.Decode(pair)
	return id, kv.Pair{pair.V, codedV}, err // pair.V is the K in PCollection<kv.Pair{ID,K}, codedV>
}

// encodeIDKFn takes a PCollection<ID,kv.Pair{K,V}> as input, and returns a
// PCollection<kv.Pair{ID,K},V>; where ID and K have been coded, and V has been
// decoded.
type encodeIDKFn struct {
	IDType         beam.EncodedType    // Type information of the privacy ID
	idEnc          beam.ElementEncoder // Encoder for privacy ID, set during Setup() according to IDType
	InputPairCodec *kv.Codec           // Codec for the input kv.Pair{K,V}
}

func newEncodeIDKFn(idType typex.FullType, kvCodec *kv.Codec) *encodeIDKFn {
	return &encodeIDKFn{
		IDType:         beam.EncodedType{idType.Type()},
		InputPairCodec: kvCodec,
	}
}

func (fn *encodeIDKFn) Setup() error {
	fn.idEnc = beam.NewElementEncoder(fn.IDType.T)
	return fn.InputPairCodec.Setup()
}

func (fn *encodeIDKFn) ProcessElement(id beam.W, pair kv.Pair) (kv.Pair, beam.V, error) {
	var idBuf bytes.Buffer
	if err := fn.idEnc.Encode(id, &idBuf); err != nil {
		return kv.Pair{}, nil, fmt.Errorf("pbeam.encodeIDKFn.ProcessElement: couldn't encode ID %v: %w", id, err)
	}
	_, v, err := fn.InputPairCodec.Decode(pair)
	return kv.Pair{idBuf.Bytes(), pair.K}, v, err
}

// decodeIDKFn is the reverse operation of encodeIDKFn. It takes a PCollection<kv.Pair{ID,K},V>
// as input, and returns a PCollection<ID, kv.Pair{K,V}>; where K and V has been coded, and ID
// has been decoded.
type decodeIDKFn struct {
	VType          beam.EncodedType    // Type information of the value V
	vEnc           beam.ElementEncoder // Encoder for privacy ID, set during Setup() according to VType
	InputPairCodec *kv.Codec           // Codec for the input kv.Pair{ID,K}
}

func newDecodeIDKFn(vType typex.FullType, idkCodec *kv.Codec) *decodeIDKFn {
	return &decodeIDKFn{
		VType:          beam.EncodedType{vType.Type()},
		InputPairCodec: idkCodec,
	}
}

func (fn *decodeIDKFn) Setup() error {
	fn.vEnc = beam.NewElementEncoder(fn.VType.T)
	return fn.InputPairCodec.Setup()
}

func (fn *decodeIDKFn) ProcessElement(pair kv.Pair, v beam.V) (beam.W, kv.Pair, error) {
	var vBuf bytes.Buffer
	if err := fn.vEnc.Encode(v, &vBuf); err != nil {
		return nil, kv.Pair{}, fmt.Errorf("pbeam.decodeIDKFn.ProcessElement: couldn't encode V %v: %w", v, err)
	}
	id, _, err := fn.InputPairCodec.Decode(pair)
	return id, kv.Pair{pair.V, vBuf.Bytes()}, err // pair.V is the K in PCollection<kv.Pair{ID,K},V>
}

// decodePairArrayFloat64Fn transforms a PCollection<pairArrayFloat64<codedX,[]float64>> into a
// PCollection<X,[]float64>.
type decodePairArrayFloat64Fn struct {
	XType beam.EncodedType
	xDec  beam.ElementDecoder
}

func newDecodePairArrayFloat64Fn(t reflect.Type) *decodePairArrayFloat64Fn {
	return &decodePairArrayFloat64Fn{XType: beam.EncodedType{t}}
}

func (fn *decodePairArrayFloat64Fn) Setup() {
	fn.xDec = beam.NewElementDecoder(fn.XType.T)
}

func (fn *decodePairArrayFloat64Fn) ProcessElement(pair pairArrayFloat64) (beam.X, []float64, error) {
	x, err := fn.xDec.Decode(bytes.NewBuffer(pair.X))
	if err != nil {
		return nil, nil, fmt.Errorf("pbeam.decodePairArrayFloat64Fn.ProcessElement: couldn't decode pair %v: %w", pair, err)
	}
	return x, pair.M, nil
}

// findConvertFn gets the correct conversion to float64 function.
func findConvertToFloat64Fn(t typex.FullType) (any, error) {
	switch t.Type().String() {
	case "int":
		return convertIntToFloat64Fn, nil
	case "int8":
		return convertInt8ToFloat64Fn, nil
	case "int16":
		return convertInt16ToFloat64Fn, nil
	case "int32":
		return convertInt32ToFloat64Fn, nil
	case "int64":
		return convertInt64ToFloat64Fn, nil
	case "uint":
		return convertUintToFloat64Fn, nil
	case "uint8":
		return convertUint8ToFloat64Fn, nil
	case "uint16":
		return convertUint16ToFloat64Fn, nil
	case "uint32":
		return convertUint32ToFloat64Fn, nil
	case "uint64":
		return convertUint64ToFloat64Fn, nil
	case "float32":
		return convertFloat32ToFloat64Fn, nil
	case "float64":
		return convertFloat64ToFloat64Fn, nil
	default:
		return nil, fmt.Errorf("unexpected value type of %v", t)
	}
}

func convertIntToFloat64Fn(z beam.Z, i int) (beam.Z, float64) {
	return z, float64(i)
}

func convertInt8ToFloat64Fn(z beam.Z, i int8) (beam.Z, float64) {
	return z, float64(i)
}

func convertInt16ToFloat64Fn(z beam.Z, i int16) (beam.Z, float64) {
	return z, float64(i)
}

func convertInt32ToFloat64Fn(z beam.Z, i int32) (beam.Z, float64) {
	return z, float64(i)
}

func convertInt64ToFloat64Fn(z beam.Z, i int64) (beam.Z, float64) {
	return z, float64(i)
}

func convertUintToFloat64Fn(z beam.Z, i uint) (beam.Z, float64) {
	return z, float64(i)
}

func convertUint8ToFloat64Fn(z beam.Z, i uint8) (beam.Z, float64) {
	return z, float64(i)
}

func convertUint16ToFloat64Fn(z beam.Z, i uint16) (beam.Z, float64) {
	return z, float64(i)
}

func convertUint32ToFloat64Fn(z beam.Z, i uint32) (beam.Z, float64) {
	return z, float64(i)
}

func convertUint64ToFloat64Fn(z beam.Z, i uint64) (beam.Z, float64) {
	return z, float64(i)
}

type expandValuesAccum struct {
	Values [][]byte
}

// expandValuesCombineFn converts a PCollection<K,V> to PCollection<K,[]V> where each value
// corresponding to the same key are collected in a slice. Resulting PCollection has a
// single slice for each key.
type expandValuesCombineFn struct {
	VType beam.EncodedType
	vEnc  beam.ElementEncoder
}

func newExpandValuesCombineFn(vType beam.EncodedType) *expandValuesCombineFn {
	return &expandValuesCombineFn{VType: vType}
}

func (fn *expandValuesCombineFn) Setup() {
	fn.vEnc = beam.NewElementEncoder(fn.VType.T)
}

func (fn *expandValuesCombineFn) CreateAccumulator() expandValuesAccum {
	return expandValuesAccum{Values: make([][]byte, 0)}
}

func (fn *expandValuesCombineFn) AddInput(a expandValuesAccum, value beam.V) (expandValuesAccum, error) {
	var vBuf bytes.Buffer
	if err := fn.vEnc.Encode(value, &vBuf); err != nil {
		return a, fmt.Errorf("pbeam.expandValuesCombineFn.AddInput: couldn't encode V %v: %w", value, err)
	}
	a.Values = append(a.Values, vBuf.Bytes())
	return a, nil
}

func (fn *expandValuesCombineFn) MergeAccumulators(a, b expandValuesAccum) expandValuesAccum {
	a.Values = append(a.Values, b.Values...)
	return a
}

func (fn *expandValuesCombineFn) ExtractOutput(a expandValuesAccum) [][]byte {
	return a.Values
}

type expandFloat64ValuesAccum struct {
	Values []float64
}

// expandFloat64ValuesCombineFn converts a PCollection<K,float64> to PCollection<K,[]float64>
// where each value corresponding to the same key are collected in a slice. Resulting
// PCollection has a single slice for each key.
type expandFloat64ValuesCombineFn struct{}

func (fn *expandFloat64ValuesCombineFn) CreateAccumulator() expandFloat64ValuesAccum {
	return expandFloat64ValuesAccum{Values: make([]float64, 0)}
}

func (fn *expandFloat64ValuesCombineFn) AddInput(a expandFloat64ValuesAccum, value float64) expandFloat64ValuesAccum {
	a.Values = append(a.Values, value)
	return a
}

func (fn *expandFloat64ValuesCombineFn) MergeAccumulators(a, b expandFloat64ValuesAccum) expandFloat64ValuesAccum {
	a.Values = append(a.Values, b.Values...)
	return a
}

func (fn *expandFloat64ValuesCombineFn) ExtractOutput(a expandFloat64ValuesAccum) []float64 {
	return a.Values
}

// pairArrayFloat64 contains an encoded value and a slice of float64 metrics.
type pairArrayFloat64 struct {
	X []byte
	M []float64
}

// rekeyArrayFloat64Fn transforms a PCollection<kv.Pair<codedK,codedV>,[]float64> into a
// PCollection<codedK,pairArrayFloat64<codedV,[]float64>>.
func rekeyArrayFloat64Fn(kv kv.Pair, m []float64) ([]byte, pairArrayFloat64) {
	return kv.K, pairArrayFloat64{kv.V, m}
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

// checkDelta returns an error if delta parameter of an aggregation is not valid. Delta
// is valid in the following cases:
//
//	delta == 0; when laplace noise with public partitions are used
//	0 < delta < 1; otherwise
func checkDelta(delta float64, noiseKind noise.Kind, publicPartitions any) error {
	if publicPartitions != nil && noiseKind == noise.LaplaceNoise {
		if delta != 0 {
			return fmt.Errorf("Delta is %e, using Laplace Noise with Public Partitions requires setting delta to 0", delta)
		}
	} else {
		return checks.CheckDeltaStrict(delta)
	}
	return nil
}
