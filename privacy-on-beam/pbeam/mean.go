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
	"math"
	"reflect"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/checks"
	"github.com/google/differential-privacy/go/dpagg"
	"github.com/google/differential-privacy/go/noise"
	"github.com/google/differential-privacy/privacy-on-beam/internal/kv"
	"github.com/apache/beam/sdks/go/pkg/beam"
)

func init() {
	beam.RegisterType(reflect.TypeOf((*boundedMeanFloat64Fn)(nil)))
}

// MeanParams specifies the parameters associated with a Mean aggregation.
type MeanParams struct {
	// Noise type (which is either LaplaceNoise{} or GaussianNoise{}).
	//
	// Defaults to LaplaceNoise{}.
	NoiseKind NoiseKind
	// Differential privacy budget consumed by this aggregation. If there is
	// only one aggregation, both Epsilon and Delta can be left 0; in that
	// case, the entire budget of the PrivacySpec is consumed.
	Epsilon, Delta float64
	// The maximum number of distinct values that a given privacy identifier
	// can influence. There is an inherent trade-off when choosing this
	// parameter: a larger MaxPartitionsContributed leads to less data loss due
	// to contribution bounding, but since the noise added in aggregations is
	// scaled according to maxPartitionsContributed, it also means that more
	// noise is added to each mean.
	//
	// Required.
	MaxPartitionsContributed int64
	// The maximum number of contributions from a given privacy identifier
	// for each key. There is an inherent trade-off when choosing this
	// parameter: a larger MaxContributionsPerPartition leads to less data loss due
	// to contribution bounding, but since the noise added in aggregations is
	// scaled according to maxContributionsPerPartition, it also means that more
	// noise is added to each mean.
	//
	// Required.
	MaxContributionsPerPartition int64
	// The total contribution of a given privacy identifier to partition can be
	// at at least MinValue, and at most MaxValue; otherwise it will be clamped
	// to these bounds. For example, if a privacy identifier is associated with
	// the key-value pairs [("a", -5), ("a", 2), ("b", 7), ("c", 3)] and the
	// (MinValue, MaxValue) bounds are (0, 5), the contribution for "a" will be
	// clamped up to 0, the contribution for "b" will be clamped down to 5, and
	// the contribution for "c" will be untouched. There is an inherent
	// trade-off when choosing MinValue and MaxValue: a small MinValue and a
	// large MaxValue means that less records will be clamped, but that more
	// noise will be added.
	//
	// Required.
	MinValue, MaxValue float64
	// You can input the list of partitions present in the output if you know
	// them in advance. When you specify partitions, partition selection /
	// thresholding will be disabled and partitions will appear in the output
	// if and only if they appear in the set of public partitions.
	//
	// You should not derive the list of partitions non-privately from private
	// data. You should only use this in either of the following cases:
	// 	1. The list of partitions is data-independent. For example, if you are
	// 	aggregating a metric by hour, you could provide a list of all possible
	// 	hourly period.
	// 	2. You use a differentially private operation to come up with the list of
	// 	partitions. For example, you could use the keys of a DistinctPrivacyID
	// 	operation as the list of public partitions.
	//
	// Note that current implementation limitations only allow up to millions of
	// public partitions.
	//
	// Optional.
	PublicPartitions beam.PCollection
}

// MeanPerKey obtains the mean of the values associated with each key in a
// PrivatePCollection<K,V>, adding differentially private noise to the means and
// doing pre-aggregation thresholding to remove means with a low number of
// distinct privacy identifiers. Client can also specify a PCollection of partitions.
//
// Note: Do not use when your results may cause overflows for float64 values.
// This aggregation is not hardened for such applications yet.
//
// MeanPerKey transforms a PrivatePCollection<K,V> into a PCollection<K,float64>.
func MeanPerKey(s beam.Scope, pcol PrivatePCollection, params MeanParams) beam.PCollection {
	s = s.Scope("pbeam.MeanPerKey")
	// Obtain & validate type information from the underlying PCollection<K,V>.
	idT, kvT := beam.ValidateKVType(pcol.col)
	if kvT.Type() != reflect.TypeOf(kv.Pair{}) {
		log.Exitf("MeanPerKey must be used on a PrivatePCollection of type <K,V>, got type %v instead", kvT)
	}
	if pcol.codec == nil {
		log.Exitf("MeanPerKey: no codec found for the input PrivatePCollection.")
	}

	// Get privacy parameters.
	spec := pcol.privacySpec
	epsilon, delta, err := spec.consumeBudget(params.Epsilon, params.Delta)
	if err != nil {
		log.Exitf("Couldn't consume budget for Mean: %v", err)
	}
	var noiseKind noise.Kind
	if params.NoiseKind == nil {
		noiseKind = noise.LaplaceNoise
		log.Infof("No NoiseKind specified, using Laplace Noise by default.")
	} else {
		noiseKind = params.NoiseKind.toNoiseKind()
	}
	err = checkMeanPerKeyParams(params, epsilon, delta, noiseKind)
	if err != nil {
		log.Exit(err)
	}

	// Drop non-public partitions, if public partitions are specified.
	if (params.PublicPartitions).IsValid() {
		if pcol.codec.KType.T != (params.PublicPartitions).Type().Type() {
			log.Exitf("Public partitions must be of type %v. Got type %v instead.",
				pcol.codec.KType.T, (params.PublicPartitions).Type().Type())
		}
		pcol.col = dropNonPublicPartitionsKVFn(s, params.PublicPartitions, pcol, pcol.codec.KType)
	}

	// First, group together the privacy ID and the partition ID and do per-partition contribution bounding.
	// Result is PCollection<kv.Pair{ID,K},V>
	encodeIDKFn := newEncodeIDKFn(idT, pcol.codec)
	decoded := beam.ParDo(s,
		encodeIDKFn,
		pcol.col,
		beam.TypeDefinition{Var: beam.VType, T: pcol.codec.VType.T})

	maxContributionsPerPartition := getMaxContributionsPerPartition(params.MaxContributionsPerPartition)
	// Don't do per-partition contribution bounding if in test mode without contribution bounding.
	if spec.testMode != noNoiseWithoutContributionBounding {
		decoded = boundContributions(s, decoded, maxContributionsPerPartition)
	}

	// Convert value to float64.
	// Result is PCollection<kv.Pair{ID,K},float64>.
	_, valueT := beam.ValidateKVType(decoded)
	convertFn, err := findConvertToFloat64Fn(valueT)
	if err != nil {
		log.Exit(err)
	}
	converted := beam.ParDo(s, convertFn, decoded)

	// Combine all values for <id, partition> into a slice.
	// Result is PCollection<kv.Pair{ID,K},[]float64>.
	combined := beam.CombinePerKey(s,
		&expandFloat64ValuesCombineFn{},
		converted)

	// Result is PCollection<ID, pairArrayFloat64>.
	maxPartitionsContributed := getMaxPartitionsContributed(spec, params.MaxPartitionsContributed)
	rekeyed := beam.ParDo(s, rekeyArrayFloat64Fn, combined)
	// Second, do cross-partition contribution bounding if not in test mode without contribution bounding.
	if spec.testMode != noNoiseWithoutContributionBounding {
		rekeyed = boundContributions(s, rekeyed, maxPartitionsContributed)
	}

	// Now that the cross-partition contribution bounding is done, remove the privacy keys and decode the values.
	// Result is PCollection<partition, []float64>.
	partialPairs := beam.DropKey(s, rekeyed)
	partitionT := pcol.codec.KType.T
	partialKV := beam.ParDo(s,
		newDecodePairArrayFloat64Fn(partitionT),
		partialPairs,
		beam.TypeDefinition{Var: beam.XType, T: partitionT})
	// Add public partitions and return the aggregation output, if public partitions are specified.
	if (params.PublicPartitions).IsValid() {
		return addPublicPartitionsForMean(s, epsilon, delta, maxPartitionsContributed,
			params, noiseKind, partialKV, spec.testMode)
	}
	// Compute the mean for each partition. Result is PCollection<partition, float64>.
	means := beam.CombinePerKey(s,
		newBoundedMeanFloat64Fn(epsilon, delta, maxPartitionsContributed, params.MaxContributionsPerPartition, params.MinValue, params.MaxValue, noiseKind, false, spec.testMode, false),
		partialKV)
	// Finally, drop thresholded partitions.
	return beam.ParDo(s, dropThresholdedPartitionsFloat64Fn, means)
}

func addPublicPartitionsForMean(s beam.Scope, epsilon, delta float64, maxPartitionsContributed int64, params MeanParams, noiseKind noise.Kind, partialKV beam.PCollection, testMode testMode) beam.PCollection {
	// Compute the mean for each partition with non-public partitions dropped. Result is PCollection<partition, float64>.
	means := beam.CombinePerKey(s,
		newBoundedMeanFloat64Fn(epsilon, delta, maxPartitionsContributed, params.MaxContributionsPerPartition, params.MinValue, params.MaxValue, noiseKind, true, testMode, false),
		partialKV)
	partitionT, _ := beam.ValidateKVType(means)
	meansPartitions := beam.DropValue(s, means)
	// Create map with partitions in the data as keys.
	partitionMap := beam.Combine(s, newPartitionsMapFn(beam.EncodedType{partitionT.Type()}), meansPartitions)
	partitionsCol := params.PublicPartitions
	// Add value of empty array to each partition key in PublicPartitions.
	publicPartitionsWithValues := beam.ParDo(s, addDummyValuesToPublicPartitionsFloat64SliceFn, partitionsCol)
	// emptyPublicPartitions are the partitions that are public but not found in the data.
	emptyPublicPartitions := beam.ParDo(s, newEmitPartitionsNotInTheDataFn(partitionT), publicPartitionsWithValues, beam.SideInput{Input: partitionMap})
	// Add noise to the empty public partitions.
	emptyMeans := beam.CombinePerKey(s,
		newBoundedMeanFloat64Fn(epsilon, delta, maxPartitionsContributed, params.MaxContributionsPerPartition, params.MinValue, params.MaxValue, noiseKind, true, testMode, true),
		emptyPublicPartitions)
	means = beam.ParDo(s, dereferenceValueToFloat64, means)
	emptyMeans = beam.ParDo(s, dereferenceValueToFloat64, emptyMeans)
	// Merge means from data with means from the empty public partitions.
	allMeans := beam.Flatten(s, means, emptyMeans)
	return allMeans
}

func checkMeanPerKeyParams(params MeanParams, epsilon, delta float64, noiseKind noise.Kind) error {
	err := checks.CheckEpsilon("pbeam.MeanPerKey", epsilon)
	if err != nil {
		return err
	}
	if (params.PublicPartitions).IsValid() && noiseKind == noise.LaplaceNoise {
		err = checks.CheckNoDelta("pbeam.MeanPerKey", delta)
	} else {
		err = checks.CheckDeltaStrict("pbeam.MeanPerKey", delta)
	}
	if err != nil {
		return err
	}
	err = checks.CheckBoundsFloat64("pbeam.MeanPerKey", params.MinValue, params.MaxValue)
	if err != nil {
		return err
	}
	return checks.CheckMaxPartitionsContributed("pbeam.MeanPerKey", params.MaxPartitionsContributed)
}

type boundedMeanAccumFloat64 struct {
	BM               *dpagg.BoundedMeanFloat64
	SP               *dpagg.PreAggSelectPartition
	PublicPartitions bool
}

// boundedMeanFloat64Fn is a differentially private combineFn for obtaining mean of values. Do not
// initialize it yourself, use newBoundedMeanFloat64Fn to create a boundedMeanFloat64Fn instance.
type boundedMeanFloat64Fn struct {
	// Privacy spec parameters (set during initial construction).
	NoiseEpsilon                 float64
	PartitionSelectionEpsilon    float64
	NoiseDelta                   float64
	PartitionSelectionDelta      float64
	MaxPartitionsContributed     int64
	MaxContributionsPerPartition int64
	Lower                        float64
	Upper                        float64
	NoiseKind                    noise.Kind
	noise                        noise.Noise // Set during Setup phase according to NoiseKind.
	PublicPartitions             bool
	TestMode                     testMode
	EmptyPartitions              bool // Set to true if this combineFn is for adding noise to empty public partitions.
}

// newBoundedMeanFloat64Fn returns a boundedMeanFloat64Fn with the given budget and parameters.
func newBoundedMeanFloat64Fn(epsilon, delta float64, maxPartitionsContributed, maxContributionsPerPartition int64, lower, upper float64, noiseKind noise.Kind, publicPartitions bool, testMode testMode, emptyPartitions bool) *boundedMeanFloat64Fn {
	fn := &boundedMeanFloat64Fn{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		Lower:                        lower,
		Upper:                        upper,
		NoiseKind:                    noiseKind,
		PublicPartitions:             publicPartitions,
		TestMode:                     testMode,
		EmptyPartitions:              emptyPartitions,
	}
	if fn.PublicPartitions {
		fn.NoiseEpsilon = epsilon
		fn.NoiseDelta = delta
		return fn
	}
	fn.NoiseEpsilon = epsilon / 2
	fn.PartitionSelectionEpsilon = epsilon - fn.NoiseEpsilon
	switch noiseKind {
	case noise.GaussianNoise:
		fn.NoiseDelta = delta / 2
	case noise.LaplaceNoise:
		fn.NoiseDelta = 0
	default:
		// TODO: return error instead
		log.Exitf("newBoundedMeanFloat64Fn: unknown noise.Kind (%v) is specified. Please specify a valid noise.", noiseKind)
	}
	fn.PartitionSelectionDelta = delta - fn.NoiseDelta
	return fn
}

func (fn *boundedMeanFloat64Fn) Setup() {
	fn.noise = noise.ToNoise(fn.NoiseKind)
	if fn.TestMode.isEnabled() {
		fn.noise = noNoise{}
	}
}

func (fn *boundedMeanFloat64Fn) CreateAccumulator() boundedMeanAccumFloat64 {
	if fn.TestMode == noNoiseWithoutContributionBounding && !fn.EmptyPartitions {
		fn.Lower = math.Inf(-1)
		fn.Upper = math.Inf(1)
	}
	accum := boundedMeanAccumFloat64{
		BM: dpagg.NewBoundedMeanFloat64(&dpagg.BoundedMeanFloat64Options{
			Epsilon:                      fn.NoiseEpsilon,
			Delta:                        fn.NoiseDelta,
			MaxPartitionsContributed:     fn.MaxPartitionsContributed,
			MaxContributionsPerPartition: fn.MaxContributionsPerPartition,
			Lower:                        fn.Lower,
			Upper:                        fn.Upper,
			Noise:                        fn.noise,
		}), PublicPartitions: fn.PublicPartitions}
	if !fn.PublicPartitions {
		accum.SP = dpagg.NewPreAggSelectPartition(&dpagg.PreAggSelectPartitionOptions{
			Epsilon:                  fn.PartitionSelectionEpsilon,
			Delta:                    fn.PartitionSelectionDelta,
			MaxPartitionsContributed: fn.MaxPartitionsContributed,
		})
	}
	return accum
}

func (fn *boundedMeanFloat64Fn) AddInput(a boundedMeanAccumFloat64, values []float64) boundedMeanAccumFloat64 {
	// We can have multiple values for each (privacy_key, partition_key) pair.
	// We need to add each value to BoundedMean as input but we need to add a single input
	// for each privacy_key to SelectPartition.
	for _, v := range values {
		a.BM.Add(v)
	}
	if !fn.PublicPartitions {
		a.SP.Increment()
	}
	return a
}

func (fn *boundedMeanFloat64Fn) MergeAccumulators(a, b boundedMeanAccumFloat64) boundedMeanAccumFloat64 {
	a.BM.Merge(b.BM)
	if !fn.PublicPartitions {
		a.SP.Merge(b.SP)
	}
	return a
}

func (fn *boundedMeanFloat64Fn) ExtractOutput(a boundedMeanAccumFloat64) *float64 {
	if fn.TestMode.isEnabled() {
		a.BM.NormalizedSum.Noise = noNoise{}
		a.BM.Count.Noise = noNoise{}
	}
	if fn.TestMode.isEnabled() || a.PublicPartitions || a.SP.ShouldKeepPartition() {
		result := a.BM.Result()
		return &result
	}
	return nil
}

func (fn *boundedMeanFloat64Fn) String() string {
	return fmt.Sprintf("%#v", fn)
}
