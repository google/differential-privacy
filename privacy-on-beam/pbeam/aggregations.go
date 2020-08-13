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
	"fmt"
	"math/rand"
	"reflect"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/checks"
	"github.com/google/differential-privacy/go/dpagg"
	"github.com/google/differential-privacy/go/noise"
	"github.com/google/differential-privacy/privacy-on-beam/internal/kv"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/transforms/top"
)

// This file contains methods & ParDos used by multiple DP aggregations.
func init() {
	beam.RegisterType(reflect.TypeOf((*boundedSumInt64Fn)(nil)))
	beam.RegisterType(reflect.TypeOf((*boundedSumFloat64Fn)(nil)))
	beam.RegisterType(reflect.TypeOf((*decodePairInt64Fn)(nil)))
	beam.RegisterType(reflect.TypeOf((*decodePairFloat64Fn)(nil)))
	beam.RegisterFunction(randBool)
	beam.RegisterFunction(clampNegativePartitionsInt64Fn)
	beam.RegisterFunction(clampNegativePartitionsFloat64Fn)
	// TODO: add tests to make sure we don't forget anything here
}

// randBool returns a uniformly random boolean. The randomness used here is not
// cryptographically secure, and using this with top.LargestPerKey doesn't
// necessarily result in a uniformly random permutation: the distribution of
// the permutation depends on the exact sorting algorithm used by Beam and the
// order in which the input values are processed within the pipeline.
//
// The fact that the resulting permutation is not nesessarily uniformly random is
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
// 	1. the key to be the privacy ID.
//  2. the value to be the pair = {partition ID, aggregated statistic
//  (either array of values which are associated with the given id and partition, or sum/count/etc of these values)}.
//
// In order to do the per-partition contribution bounding we need:
// 	1. the key to be the pair = {privacy ID, partition ID}.
// 	2. the value to be just the value which is associated with that {privacy ID, partition ID} pair
// 	(there could be multiple entries with the same key).
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

func findRekeyFn(kind reflect.Kind) interface{} {
	switch kind {
	case reflect.Int64:
		return rekeyInt64Fn
	case reflect.Float64:
		return rekeyFloat64Fn
	default:
		log.Exitf("pbeam.findRekeyFn: kind(%v) should be int64 or float64", kind)
	}
	return nil
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

func newDecodePairFn(t reflect.Type, kind reflect.Kind) interface{} {
	switch kind {
	case reflect.Int64:
		return newDecodePairInt64Fn(t)
	case reflect.Float64:
		return newDecodePairFloat64Fn(t)
	default:
		log.Exitf("pbeam.newDecodePairFn: kind(%v) should be int64 or float64", kind)
	}
	return nil
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

func (fn *decodePairInt64Fn) ProcessElement(pair pairInt64) (beam.X, int64) {
	x, err := fn.xDec.Decode(bytes.NewBuffer(pair.X))
	if err != nil {
		log.Exitf("pbeam.decodePairInt64Fn.ProcessElement: couldn't decode pair %v: %v", pair, err)
	}
	return x, pair.M
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

func (fn *decodePairFloat64Fn) ProcessElement(pair pairFloat64) (beam.X, float64) {
	x, err := fn.xDec.Decode(bytes.NewBuffer(pair.X))
	if err != nil {
		log.Exitf("pbeam.decodePairFloat64Fn.ProcessElement: couldn't decode pair %v: %v", pair, err)
	}
	return x, pair.M
}

func newBoundedSumFn(epsilon, delta float64, maxPartitionsContributed int64, lower, upper float64, noiseKind noise.Kind, vKind reflect.Kind) interface{} {
	var err error
	var bsFn interface{}

	switch vKind {
	case reflect.Int64:
		err = checks.CheckBoundsFloat64AsInt64("pbeam.newBoundedSumFn", lower, upper)
		bsFn = newBoundedSumInt64Fn(epsilon, delta, maxPartitionsContributed, int64(lower), int64(upper), noiseKind)
	case reflect.Float64:
		err = checks.CheckBoundsFloat64("pbeam.newBoundedSumFn", lower, upper)
		bsFn = newBoundedSumFloat64Fn(epsilon, delta, maxPartitionsContributed, lower, upper, noiseKind)
	default:
		log.Exitf("pbeam.newBoundedSumFn: vKind(%v) should be int64 or float64", vKind)
	}

	if err != nil {
		log.Exit(err)
	}
	return bsFn
}

type boundedSumAccumInt64 struct {
	BS *dpagg.BoundedSumInt64
	SP *dpagg.PreAggSelectPartition
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
}

// newBoundedSumInt64Fn returns a boundedSumInt64Fn with the given budget and parameters.
func newBoundedSumInt64Fn(epsilon, delta float64, maxPartitionsContributed, lower, upper int64, noiseKind noise.Kind) *boundedSumInt64Fn {
	fn := &boundedSumInt64Fn{
		MaxPartitionsContributed: maxPartitionsContributed,
		Lower:                    lower,
		Upper:                    upper,
		NoiseKind:                noiseKind,
	}
	fn.NoiseEpsilon = epsilon / 2
	fn.PartitionSelectionEpsilon = epsilon / 2
	switch noiseKind {
	case noise.GaussianNoise:
		fn.NoiseDelta = delta / 2
		fn.PartitionSelectionDelta = delta / 2
	case noise.LaplaceNoise:
		fn.NoiseDelta = 0
		fn.PartitionSelectionDelta = delta
	default:
		log.Exitf("newBoundedSumInt64Fn: unknown noise.Kind (%v) is specified. Please specify a valid noise.", noiseKind)
	}
	return fn
}

func (fn *boundedSumInt64Fn) Setup() {
	fn.noise = noise.ToNoise(fn.NoiseKind)
}

func (fn *boundedSumInt64Fn) CreateAccumulator() boundedSumAccumInt64 {
	return boundedSumAccumInt64{
		BS: dpagg.NewBoundedSumInt64(&dpagg.BoundedSumInt64Options{
			Epsilon:                  fn.NoiseEpsilon,
			Delta:                    fn.NoiseDelta,
			MaxPartitionsContributed: fn.MaxPartitionsContributed,
			Lower:                    fn.Lower,
			Upper:                    fn.Upper,
			Noise:                    fn.noise,
		}),
		SP: dpagg.NewPreAggSelectPartition(&dpagg.PreAggSelectPartitionOptions{
			Epsilon:                  fn.PartitionSelectionEpsilon,
			Delta:                    fn.PartitionSelectionDelta,
			MaxPartitionsContributed: fn.MaxPartitionsContributed,
		}),
	}
}

func (fn *boundedSumInt64Fn) AddInput(a boundedSumAccumInt64, value int64) boundedSumAccumInt64 {
	a.BS.Add(value)
	a.SP.Increment()
	return a
}

func (fn *boundedSumInt64Fn) MergeAccumulators(a, b boundedSumAccumInt64) boundedSumAccumInt64 {
	a.BS.Merge(b.BS)
	a.SP.Merge(b.SP)
	return a
}

func (fn *boundedSumInt64Fn) ExtractOutput(a boundedSumAccumInt64) *int64 {
	if a.SP.ShouldKeepPartition() {
		result := a.BS.Result()
		return &result
	}
	return nil
}

func (fn *boundedSumInt64Fn) String() string {
	return fmt.Sprintf("%#v", fn)
}

type boundedSumAccumFloat64 struct {
	BS *dpagg.BoundedSumFloat64
	SP *dpagg.PreAggSelectPartition
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
	noise noise.Noise
}

// newBoundedSumFloat64Fn returns a boundedSumFloat64Fn with the given budget and parameters.
func newBoundedSumFloat64Fn(epsilon, delta float64, maxPartitionsContributed int64, lower, upper float64, noiseKind noise.Kind) *boundedSumFloat64Fn {
	fn := &boundedSumFloat64Fn{
		MaxPartitionsContributed: maxPartitionsContributed,
		Lower:                    lower,
		Upper:                    upper,
		NoiseKind:                noiseKind,
	}
	fn.NoiseEpsilon = epsilon / 2
	fn.PartitionSelectionEpsilon = epsilon / 2
	switch noiseKind {
	case noise.GaussianNoise:
		fn.NoiseDelta = delta / 2
		fn.PartitionSelectionDelta = delta / 2
	case noise.LaplaceNoise:
		fn.NoiseDelta = 0
		fn.PartitionSelectionDelta = delta
	default:
		log.Exitf("newBoundedSumFloat64Fn: unknown noise.Kind (%v) is specified. Please specify a valid noise.", noiseKind)
	}
	return fn
}

func (fn *boundedSumFloat64Fn) Setup() {
	fn.noise = noise.ToNoise(fn.NoiseKind)
}

func (fn *boundedSumFloat64Fn) CreateAccumulator() boundedSumAccumFloat64 {
	return boundedSumAccumFloat64{
		BS: dpagg.NewBoundedSumFloat64(&dpagg.BoundedSumFloat64Options{
			Epsilon:                  fn.NoiseEpsilon,
			Delta:                    fn.NoiseDelta,
			MaxPartitionsContributed: fn.MaxPartitionsContributed,
			Lower:                    fn.Lower,
			Upper:                    fn.Upper,
			Noise:                    fn.noise,
		}),
		SP: dpagg.NewPreAggSelectPartition(&dpagg.PreAggSelectPartitionOptions{
			Epsilon:                  fn.PartitionSelectionEpsilon,
			Delta:                    fn.PartitionSelectionDelta,
			MaxPartitionsContributed: fn.MaxPartitionsContributed,
		})}
}

func (fn *boundedSumFloat64Fn) AddInput(a boundedSumAccumFloat64, value float64) boundedSumAccumFloat64 {
	a.BS.Add(value)
	a.SP.Increment()
	return a
}

func (fn *boundedSumFloat64Fn) MergeAccumulators(a, b boundedSumAccumFloat64) boundedSumAccumFloat64 {
	a.BS.Merge(b.BS)
	a.SP.Merge(b.SP)
	return a
}

func (fn *boundedSumFloat64Fn) ExtractOutput(a boundedSumAccumFloat64) *float64 {
	if a.SP.ShouldKeepPartition() {
		result := a.BS.Result()
		return &result
	}
	return nil
}

func (fn *boundedSumFloat64Fn) String() string {
	return fmt.Sprintf("%#v", fn)
}

func findDropThresholdedPartitionsFn(kind reflect.Kind) interface{} {
	switch kind {
	case reflect.Int64:
		return dropThresholdedPartitionsInt64Fn
	case reflect.Float64:
		return dropThresholdedPartitionsFloat64Fn
	default:
		log.Exitf("pbeam.findDropThresholdedPartitionsFn: kind(%v) should be int64 or float64", kind)
	}
	return nil
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

func findClampNegativePartitionsFn(kind reflect.Kind) interface{} {
	switch kind {
	case reflect.Int64:
		return clampNegativePartitionsInt64Fn
	case reflect.Float64:
		return clampNegativePartitionsFloat64Fn
	default:
		log.Exitf("pbeam.findClampNegativePartitionsFn: kind(%v) should be int64 or float64", kind)
	}
	return nil
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
