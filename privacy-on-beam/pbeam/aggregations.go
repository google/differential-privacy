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

	"github.com/google/differential-privacy/go/v3/checks"
	"github.com/google/differential-privacy/go/v3/dpagg"
	"github.com/google/differential-privacy/go/v3/noise"
	"github.com/google/differential-privacy/privacy-on-beam/v3/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/core/typex"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/register"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/top"
)

func init() {
	register.Combiner3[boundedSumAccumInt64, int64, *int64](&boundedSumInt64Fn{})
	register.Combiner3[boundedSumAccumFloat64, float64, *float64](&boundedSumFloat64Fn{})
	register.Combiner3[expandValuesAccum, beam.V, [][]byte](&expandValuesCombineFn{})
	register.Combiner3[expandFloat64ValuesAccum, float64, []float64](&expandFloat64ValuesCombineFn{})

	register.DoFn1x3[pairInt64, beam.W, int64, error](&decodePairInt64Fn{})
	register.DoFn1x3[pairFloat64, beam.W, float64, error](&decodePairFloat64Fn{})
	register.DoFn2x3[beam.U, kv.Pair, beam.U, beam.W, error](&dropValuesFn{})
	register.DoFn2x3[kv.Pair, []byte, beam.W, kv.Pair, error](&encodeKVFn{})
	register.DoFn2x3[beam.W, kv.Pair, kv.Pair, beam.V, error](&encodeIDKFn{})
	register.DoFn2x3[kv.Pair, beam.V, beam.W, kv.Pair, error](&decodeIDKFn{})
	register.DoFn1x3[pairArrayFloat64, beam.W, []float64, error](&decodePairArrayFloat64Fn{})

	register.Function2x1[beam.V, beam.V, bool](randBool)
	register.Function3x0[beam.W, []beam.V, func(beam.W, beam.V)](flattenValues)
	register.Emitter2[beam.W, beam.V]()
	register.Function2x2[kv.Pair, int64, []byte, pairInt64](rekeyInt64)
	register.Function2x2[kv.Pair, float64, []byte, pairFloat64](rekeyFloat64)
	register.Function2x2[kv.Pair, []float64, []byte, pairArrayFloat64](rekeyArrayFloat64)
	register.Function2x2[beam.W, int64, beam.W, int64](clampNegativePartitionsInt64)
	register.Function2x2[beam.W, float64, beam.W, float64](clampNegativePartitionsFloat64)
	register.Function3x0[beam.V, *int64, func(beam.V, int64)](dropThresholdedPartitionsInt64)
	register.Emitter2[beam.V, int64]()
	register.Function3x0[beam.V, *float64, func(beam.V, float64)](dropThresholdedPartitionsFloat64)
	register.Emitter2[beam.V, float64]()
	register.Function3x0[beam.V, *VarianceStatistics, func(beam.V, VarianceStatistics)](
		dropThresholdedPartitionsVarianceStatistics)
	register.Emitter2[beam.V, VarianceStatistics]()
	register.Function3x0[beam.V, []float64, func(beam.V, []float64)](dropThresholdedPartitionsFloat64Slice)
	register.Emitter2[beam.V, []float64]()
	register.Function2x2[beam.W, *int64, beam.W, int64](dereferenceValueInt64)
	register.Function2x2[beam.W, *float64, beam.W, float64](dereferenceValueFloat64)
	register.Function2x2[beam.W, *VarianceStatistics, beam.W, VarianceStatistics](
		dereferenceVarianceStatistics)
	register.Function2x3[kv.Pair, beam.V, kv.Pair, int64, error](convertToInt64Fn)
	register.Function2x3[kv.Pair, beam.V, kv.Pair, float64, error](convertToFloat64Fn)
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
	return beam.ParDo(s, flattenValues, sampled)
}

// Given a PCollection<K,[]V>, flattens the second argument to return a PCollection<K,V>.
func flattenValues(key beam.W, values []beam.V, emit func(beam.W, beam.V)) {
	for _, v := range values {
		emit(key, v)
	}
}

func findRekeyFn(kind reflect.Kind) (any, error) {
	switch kind {
	case reflect.Int64:
		return rekeyInt64, nil
	case reflect.Float64:
		return rekeyFloat64, nil
	default:
		return nil, fmt.Errorf("kind(%v) should be int64 or float64", kind)
	}
}

// pairInt64 contains an encoded partition key and an int64 metric.
type pairInt64 struct {
	K []byte
	M int64
}

// rekeyInt64 transforms a PCollection<kv.Pair<codedK,codedV>,int64> into a
// PCollection<codedK,pairInt64<codedV,int>>.
func rekeyInt64(kv kv.Pair, m int64) ([]byte, pairInt64) {
	return kv.K, pairInt64{kv.V, m}
}

// pairFloat64 contains an encoded value and an float64 metric.
type pairFloat64 struct {
	K []byte
	M float64
}

// rekeyFloat64 transforms a PCollection<kv.Pair<codedK,codedV>,float64> into a
// PCollection<codedK,pairFloat64<codedV,int>>.
func rekeyFloat64(kv kv.Pair, m float64) ([]byte, pairFloat64) {
	return kv.K, pairFloat64{kv.V, m}
}

// pairArrayFloat64 contains an encoded partition key and a slice of float64 metrics.
type pairArrayFloat64 struct {
	K []byte
	M []float64
}

// rekeyArrayFloat64 transforms a PCollection<kv.Pair<codedK,codedV>,[]float64> into a
// PCollection<codedK,pairArrayFloat64<codedV,[]float64>>.
func rekeyArrayFloat64(kv kv.Pair, m []float64) ([]byte, pairArrayFloat64) {
	return kv.K, pairArrayFloat64{kv.V, m}
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

// decodePairInt64Fn transforms a PCollection<pairInt64<KX,int64>> into a
// PCollection<K,int64>.
type decodePairInt64Fn struct {
	KType beam.EncodedType
	kDec  beam.ElementDecoder
}

func newDecodePairInt64Fn(t reflect.Type) *decodePairInt64Fn {
	return &decodePairInt64Fn{KType: beam.EncodedType{t}}
}

func (fn *decodePairInt64Fn) Setup() {
	fn.kDec = beam.NewElementDecoder(fn.KType.T)
}

func (fn *decodePairInt64Fn) ProcessElement(pair pairInt64) (beam.W, int64, error) {
	k, err := fn.kDec.Decode(bytes.NewBuffer(pair.K))
	if err != nil {
		return nil, 0, fmt.Errorf("pbeam.decodePairInt64Fn.ProcessElement: couldn't decode pair %v: %w", pair, err)
	}
	return k, pair.M, nil
}

// decodePairFloat64Fn transforms a PCollection<pairFloat64<codedK,float64>> into a
// PCollection<K,float64>.
type decodePairFloat64Fn struct {
	KType beam.EncodedType
	kDec  beam.ElementDecoder
}

func newDecodePairFloat64Fn(t reflect.Type) *decodePairFloat64Fn {
	return &decodePairFloat64Fn{KType: beam.EncodedType{t}}
}

func (fn *decodePairFloat64Fn) Setup() {
	fn.kDec = beam.NewElementDecoder(fn.KType.T)
}

func (fn *decodePairFloat64Fn) ProcessElement(pair pairFloat64) (beam.W, float64, error) {
	k, err := fn.kDec.Decode(bytes.NewBuffer(pair.K))
	if err != nil {
		return nil, 0.0, fmt.Errorf("pbeam.decodePairFloat64Fn.ProcessElement: couldn't decode pair %v: %w", pair, err)
	}
	return k, pair.M, nil
}

// decodePairArrayFloat64Fn transforms a PCollection<pairArrayFloat64<codedK,[]float64>> into a
// PCollection<K,[]float64>.
type decodePairArrayFloat64Fn struct {
	KType beam.EncodedType
	kDec  beam.ElementDecoder
}

func newDecodePairArrayFloat64Fn(t reflect.Type) *decodePairArrayFloat64Fn {
	return &decodePairArrayFloat64Fn{KType: beam.EncodedType{t}}
}

func (fn *decodePairArrayFloat64Fn) Setup() {
	fn.kDec = beam.NewElementDecoder(fn.KType.T)
}

func (fn *decodePairArrayFloat64Fn) ProcessElement(pair pairArrayFloat64) (beam.W, []float64, error) {
	k, err := fn.kDec.Decode(bytes.NewBuffer(pair.K))
	if err != nil {
		return nil, nil, fmt.Errorf("pbeam.decodePairArrayFloat64Fn.ProcessElement: couldn't decode pair %v: %w", pair, err)
	}
	return k, pair.M, nil
}

// newBoundedSumFn returns a boundedSumInt64Fn or boundedSumFloat64Fn depending on vKind.
func newBoundedSumFn(spec PrivacySpec, params SumParams, noiseKind noise.Kind, vKind reflect.Kind, publicPartitions bool) (any, error) {
	var err, checkErr error
	var bsFn any
	switch vKind {
	case reflect.Int64:
		checkErr = checks.CheckBoundsFloat64AsInt64(params.MinValue, params.MaxValue)
		if checkErr != nil {
			return nil, checkErr
		}
		bsFn, err = newBoundedSumInt64Fn(spec, params, noiseKind, publicPartitions)
	case reflect.Float64:
		checkErr = checks.CheckBoundsFloat64(params.MinValue, params.MaxValue)
		if checkErr != nil {
			return nil, checkErr
		}
		bsFn, err = newBoundedSumFloat64Fn(spec, params, noiseKind, publicPartitions)
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
	PreThreshold              int64
	MaxPartitionsContributed  int64
	Lower                     int64
	Upper                     int64
	NoiseKind                 noise.Kind
	noise                     noise.Noise // Set during Setup phase according to NoiseKind.
	PublicPartitions          bool
	TestMode                  TestMode
}

// newBoundedSumInt64Fn returns a boundedSumInt64Fn with the given budget and parameters.
func newBoundedSumInt64Fn(spec PrivacySpec, params SumParams, noiseKind noise.Kind, publicPartitions bool) (*boundedSumInt64Fn, error) {
	if noiseKind != noise.GaussianNoise && noiseKind != noise.LaplaceNoise {
		return nil, fmt.Errorf("unknown noise.Kind (%v) is specified. Please specify a valid noise", noiseKind)
	}
	return &boundedSumInt64Fn{
		NoiseEpsilon:              params.AggregationEpsilon,
		NoiseDelta:                params.AggregationDelta,
		PartitionSelectionEpsilon: params.PartitionSelectionParams.Epsilon,
		PartitionSelectionDelta:   params.PartitionSelectionParams.Delta,
		PreThreshold:              spec.preThreshold,
		MaxPartitionsContributed:  params.MaxPartitionsContributed,
		Lower:                     int64(params.MinValue),
		Upper:                     int64(params.MaxValue),
		NoiseKind:                 noiseKind,
		PublicPartitions:          publicPartitions,
		TestMode:                  spec.testMode,
	}, nil
}

func (fn *boundedSumInt64Fn) Setup() {
	fn.noise = noise.ToNoise(fn.NoiseKind)
	if fn.TestMode.isEnabled() {
		fn.noise = noNoise{}
	}
}

func (fn *boundedSumInt64Fn) CreateAccumulator() (boundedSumAccumInt64, error) {
	if fn.TestMode == TestModeWithoutContributionBounding {
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
			PreThreshold:             fn.PreThreshold,
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
	PreThreshold              int64
	MaxPartitionsContributed  int64
	Lower                     float64
	Upper                     float64
	NoiseKind                 noise.Kind
	// Noise, set during Setup phase according to NoiseKind.
	noise            noise.Noise
	PublicPartitions bool
	TestMode         TestMode
}

// newBoundedSumFloat64Fn returns a boundedSumFloat64Fn with the given budget and parameters.
func newBoundedSumFloat64Fn(spec PrivacySpec, params SumParams, noiseKind noise.Kind, publicPartitions bool) (*boundedSumFloat64Fn, error) {
	if noiseKind != noise.GaussianNoise && noiseKind != noise.LaplaceNoise {
		return nil, fmt.Errorf("unknown noise.Kind (%v) is specified. Please specify a valid noise", noiseKind)
	}
	return &boundedSumFloat64Fn{
		NoiseEpsilon:              params.AggregationEpsilon,
		NoiseDelta:                params.AggregationDelta,
		PartitionSelectionEpsilon: params.PartitionSelectionParams.Epsilon,
		PartitionSelectionDelta:   params.PartitionSelectionParams.Delta,
		PreThreshold:              spec.preThreshold,
		MaxPartitionsContributed:  params.MaxPartitionsContributed,
		Lower:                     params.MinValue,
		Upper:                     params.MaxValue,
		NoiseKind:                 noiseKind,
		PublicPartitions:          publicPartitions,
		TestMode:                  spec.testMode,
	}, nil
}

func (fn *boundedSumFloat64Fn) Setup() {
	fn.noise = noise.ToNoise(fn.NoiseKind)
	if fn.TestMode.isEnabled() {
		fn.noise = noNoise{}
	}
}

func (fn *boundedSumFloat64Fn) CreateAccumulator() (boundedSumAccumFloat64, error) {
	if fn.TestMode == TestModeWithoutContributionBounding {
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
			PreThreshold:             fn.PreThreshold,
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

func (fn *boundedSumFloat64Fn) String() string {
	return fmt.Sprintf("%#v", fn)
}

// findDereferenceValueFn dereferences a *int64 to int64 or *float64 to float64.
func findDereferenceValueFn(kind reflect.Kind) (any, error) {
	switch kind {
	case reflect.Int64:
		return dereferenceValueInt64, nil
	case reflect.Float64:
		return dereferenceValueFloat64, nil
	default:
		return nil, fmt.Errorf("kind(%v) should be int64 or float64", kind)
	}
}

func dereferenceValueInt64(key beam.W, value *int64) (k beam.W, v int64) {
	return key, *value
}

func dereferenceValueFloat64(key beam.W, value *float64) (k beam.W, v float64) {
	return key, *value
}

func dereferenceVarianceStatistics(
	key beam.W, value *VarianceStatistics,
) (k beam.W, v VarianceStatistics) {
	return key, *value
}

func findDropThresholdedPartitionsFn(kind reflect.Kind) (any, error) {
	switch kind {
	case reflect.Int64:
		return dropThresholdedPartitionsInt64, nil
	case reflect.Float64:
		return dropThresholdedPartitionsFloat64, nil
	default:
		return nil, fmt.Errorf("kind(%v) should be int64 or float64", kind)
	}
}

// dropThresholdedPartitionsInt64 drops thresholded int partitions, i.e. those
// that have nil r, by emitting only non-thresholded partitions.
func dropThresholdedPartitionsInt64(v beam.V, r *int64, emit func(beam.V, int64)) {
	if r != nil {
		emit(v, *r)
	}
}

// dropThresholdedPartitionsFloat64 drops thresholded float partitions, i.e. those
// that have nil r, by emitting only non-thresholded partitions.
func dropThresholdedPartitionsFloat64(v beam.V, r *float64, emit func(beam.V, float64)) {
	if r != nil {
		emit(v, *r)
	}
}

// dropThresholdedPartitionsVarianceStatistics drops thresholded partitions, i.e. those
// that have nil r, by emitting only non-thresholded partitions.
func dropThresholdedPartitionsVarianceStatistics(
	v beam.V, r *VarianceStatistics, emit func(beam.V, VarianceStatistics),
) {
	if r != nil {
		emit(v, *r)
	}
}

// dropThresholdedPartitionsFloat64Slice drops thresholded []float64 partitions, i.e.
// those that have nil r, by emitting only non-thresholded partitions.
func dropThresholdedPartitionsFloat64Slice(v beam.V, r []float64, emit func(beam.V, []float64)) {
	if len(r) != 0 {
		emit(v, r)
	}
}

func findClampNegativePartitionsFn(kind reflect.Kind) (any, error) {
	switch kind {
	case reflect.Int64:
		return clampNegativePartitionsInt64, nil
	case reflect.Float64:
		return clampNegativePartitionsFloat64, nil
	default:
		return nil, fmt.Errorf("kind(%v) should be int64 or float64", kind)
	}
}

// Clamp negative partitions to zero for int64 partitions, e.g., as a post aggregation step for Count.
func clampNegativePartitionsInt64(k beam.W, r int64) (beam.W, int64) {
	if r < 0 {
		return k, 0
	}
	return k, r
}

// Clamp negative partitions to zero for float64 partitions.
func clampNegativePartitionsFloat64(k beam.W, r float64) (beam.W, float64) {
	if r < 0 {
		return k, 0
	}
	return k, r
}

type dropValuesFn struct {
	Codec *kv.Codec
}

func (fn *dropValuesFn) Setup() {
	fn.Codec.Setup()
}

func (fn *dropValuesFn) ProcessElement(id beam.U, kv kv.Pair) (beam.U, beam.W, error) {
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

func convertToInt64Fn(idk kv.Pair, i beam.V) (kv.Pair, int64, error) {
	v := reflect.ValueOf(i)
	if !v.Type().ConvertibleTo(reflect.TypeOf(int64(0))) {
		return kv.Pair{}, 0, fmt.Errorf("unexpected value type of %v", v.Type())
	}
	return idk, v.Convert(reflect.TypeOf(int64(0))).Int(), nil
}

func convertToFloat64Fn(idk kv.Pair, i beam.V) (kv.Pair, float64, error) {
	v := reflect.ValueOf(i)
	if !v.Type().ConvertibleTo(reflect.TypeOf(float64(0))) {
		return kv.Pair{}, 0, fmt.Errorf("unexpected value type of %v", v.Type())
	}
	return idk, v.Convert(reflect.TypeOf(float64(0))).Float(), nil
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

// checkAggregationEpsilon returns an error if the AggregationEpsilon parameter of an aggregation is not valid.
// AggregationEpsilon is valid if 0 < AggregationEpsilon < +∞.
func checkAggregationEpsilon(epsilon float64) error {
	return checks.CheckEpsilonStrict(epsilon, "AggregationEpsilon")
}

// checkPartitionSelectionEpsilon returns an error if the PartitionSelectionEpsilon parameter of an aggregation is not valid.
// PartitionSelectionEpsilon is valid in the following cases:
//
//	PartitionSelectionEpsilon == 0; if public partitions are used
//	0 < PartitionSelectionEpsilon < +∞; otherwise
func checkPartitionSelectionEpsilon(epsilon float64, publicPartitions any) error {
	if publicPartitions != nil {
		if epsilon != 0 {
			return fmt.Errorf("PartitionSelectionEpsilon is %e, using public partitions requires setting PartitionSelectionEpsilon to 0", epsilon)
		}
		return nil
	}
	return checks.CheckEpsilonStrict(epsilon, "PartitionSelectionEpsilon")
}

// checkAggregationDelta returns an error if the AggregationDelta parameter of an aggregation is not valid.
// AggregationDelta is valid in the following cases:
//
//	AggregationDelta == 0; when laplace noise is used
//	0 < AggregationDelta < 1; otherwise
func checkAggregationDelta(delta float64, noiseKind noise.Kind) error {
	if noiseKind == noise.LaplaceNoise {
		return checks.CheckNoDelta(delta, "AggregationDelta")
	}
	return checks.CheckDeltaStrict(delta, "AggregationDelta")
}

// checkPartitionSelectionDelta returns an error if the PartitionSelectionDelta parameter of an aggregation is not valid.
// PartitionSelectionDelta is valid in the following cases:
//
//	PartitionSelectionDelta == 0; if public partitions are used
//	0 < PartitionSelectionDelta < 1; otherwise
func checkPartitionSelectionDelta(delta float64, publicPartitions any) error {
	if publicPartitions != nil {
		return checks.CheckNoDelta(delta, "PartitionSelectionDelta")
	}
	return checks.CheckDeltaStrict(delta, "PartitionSelectionDelta")
}

// checkMaxPartitionsContributed returns an error if maxPartitionsContributed parameter of an aggregation
// is smaller than or equal to 0.
func checkMaxPartitionsContributed(maxPartitionsContributed int64) error {
	if maxPartitionsContributed <= 0 {
		return fmt.Errorf("MaxPartitionsContributed must be set to a positive value, was %d instead", maxPartitionsContributed)
	}
	return nil
}

// checkMaxPartitionsContributed returns an error if maxPartitionsContributed parameter of a PartitionSelectionParams
// is set to anything other than 0.
func checkMaxPartitionsContributedPartitionSelection(maxPartitionsContributed int64) error {
	if maxPartitionsContributed != 0 {
		return fmt.Errorf("separate contribution bounding for partition selection is not supported: "+
			"PartitionSelectionParams.MaxPartitionsContributed must be unset (i.e. 0), was %d instead", maxPartitionsContributed)
	}
	return nil
}

// checkNumericType returns an error if t is not a numeric type.
func checkNumericType(t typex.FullType) error {
	switch t.Type().Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return nil
	case reflect.Float32, reflect.Float64:
		return nil
	default:
		return fmt.Errorf("unexpected value type of %v", t)
	}
}
