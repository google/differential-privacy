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

package pbeam

import (
	"fmt"
	"reflect"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/checks"
	"github.com/google/differential-privacy/go/dpagg"
	"github.com/google/differential-privacy/go/noise"
	"github.com/google/differential-privacy/privacy-on-beam/internal/kv"
	"github.com/apache/beam/sdks/go/pkg/beam"
)

func init() {
	beam.RegisterType(reflect.TypeOf((*boundedQuantilesFn)(nil)))
}

// QuantilesParams specifies the parameters associated with a Quantiles aggregation.
type QuantilesParams struct {
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
	// noise is added to each quantile.
	//
	// Required.
	MaxPartitionsContributed int64
	// The maximum number of contributions from a given privacy identifier
	// for each key. There is an inherent trade-off when choosing this
	// parameter: a larger MaxContributionsPerPartition leads to less data loss due
	// to contribution bounding, but since the noise added in aggregations is
	// scaled according to maxContributionsPerPartition, it also means that more
	// noise is added to each quantile.
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
	// Percentile ranks that the quantiles should be computed for. Each rank must
	// be between zero and one. The DP quantile operation returns a list of
	// quantile values corresponding to the respective ranks. E.g., a percentile
	// rank of 0.2 yields a quantile value that is greater than 20% and less than
	// 80% of the values in the data set.
	//
	// Note that computing multiple quantiles does not consume extra privacy budget,
	// i.e. computing multiple quantiles does not make each quantile less accurate
	// for a fixed privacy budget.
	Ranks []float64
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

// QuantilesPerKey computes one or multiple quantiles of the values associated with each
// key in a PrivatePCollection<K,V>, adding differentially private noise to the quantiles
// and doing pre-aggregation thresholding to remove partitions with a low number of
// distinct privacy identifiers. Client can also specify a PCollection of partitions.
//
// QuantilesPerKey transforms a PrivatePCollection<K,V> into a PCollection<K,[]float64>.
//
// Note that due to the implementation details of the internal Quantiles algorithm, using pbeamtest
// with QuantilesPerKey has two caveats:
//
// 	1. Even without DP noise, the output will be slightly noisy. You can use
//  pbeamtest.QuantilesTolerance() to account for that noise.
//  2. It is not possible to not clamp input values when using
//  pbeamtest.NewPrivacySpecNoNoiseWithoutContributionBounding(), so clamping to Min/MaxValue will
//  still be applied. However, MaxContributionsPerPartition and MaxPartitionsContributed contribution
//  bounding will be disabled.
func QuantilesPerKey(s beam.Scope, pcol PrivatePCollection, params QuantilesParams) beam.PCollection {
	s = s.Scope("pbeam.QuantilesPerKey")
	// Obtain & validate type information from the underlying PCollection<K,V>.
	idT, kvT := beam.ValidateKVType(pcol.col)
	if kvT.Type() != reflect.TypeOf(kv.Pair{}) {
		log.Fatalf("QuantilesPerKey must be used on a PrivatePCollection of type <K,V>, got type %v instead", kvT)
	}
	if pcol.codec == nil {
		log.Fatalf("QuantilesPerKey: no codec found for the input PrivatePCollection.")
	}

	// Get privacy parameters.
	spec := pcol.privacySpec
	epsilon, delta, err := spec.consumeBudget(params.Epsilon, params.Delta)
	if err != nil {
		log.Fatalf("Couldn't consume budget for Quantiles: %v", err)
	}
	var noiseKind noise.Kind
	if params.NoiseKind == nil {
		noiseKind = noise.LaplaceNoise
		log.Infof("No NoiseKind specified, using Laplace Noise by default.")
	} else {
		noiseKind = params.NoiseKind.toNoiseKind()
	}
	err = checkQuantilesPerKeyParams(params, epsilon, delta, noiseKind)
	if err != nil {
		log.Fatal(err)
	}

	// Drop non-public partitions, if public partitions are specified.
	if (params.PublicPartitions).IsValid() {
		if pcol.codec.KType.T != (params.PublicPartitions).Type().Type() {
			log.Fatalf("Public partitions must be of type %v. Got type %v instead.",
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

	maxContributionsPerPartition, err := getMaxContributionsPerPartition(params.MaxContributionsPerPartition)
	if err != nil {
		log.Fatalf("Couldn't get MaxContributionsPerPartition for QuantilesPerKey: %v", err)
	}

	// Don't do per-partition contribution bounding if in test mode without contribution bounding.
	if spec.testMode != noNoiseWithoutContributionBounding {
		decoded = boundContributions(s, decoded, maxContributionsPerPartition)
	}

	// Convert value to float64.
	// Result is PCollection<kv.Pair{ID,K},float64>.
	_, valueT := beam.ValidateKVType(decoded)
	convertFn, err := findConvertToFloat64Fn(valueT)
	if err != nil {
		log.Fatalf("Couldn't get convertFn for QuantilesPerKey: %v", err)
	}
	converted := beam.ParDo(s, convertFn, decoded)

	// Combine all values for <id, partition> into a slice.
	// Result is PCollection<kv.Pair{ID,K},[]float64>.
	combined := beam.CombinePerKey(s,
		&expandFloat64ValuesCombineFn{},
		converted)

	// Result is PCollection<ID, pairArrayFloat64>.
	maxPartitionsContributed, err := getMaxPartitionsContributed(spec, params.MaxPartitionsContributed)
	if err != nil {
		log.Fatalf("Couldn't get MaxPartitionsContributed for QuantilesPerKey: %v", err)
	}
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
		return addPublicPartitionsForQuantiles(s, epsilon, delta, maxPartitionsContributed,
			params, noiseKind, partialKV, spec.testMode)
	}
	// Compute the quantiles for each partition. Result is PCollection<partition, []float64>.
	boundedQuantilesFn, err := newBoundedQuantilesFn(boundedQuantilesFnParams{
		epsilon:                      epsilon,
		delta:                        delta,
		maxPartitionsContributed:     maxPartitionsContributed,
		maxContributionsPerPartition: params.MaxContributionsPerPartition,
		minValue:                     params.MinValue,
		maxValue:                     params.MaxValue,
		noiseKind:                    noiseKind,
		ranks:                        params.Ranks,
		publicPartitions:             false,
		testMode:                     spec.testMode,
		emptyPartitions:              false})
	if err != nil {
		log.Fatalf("Couldn't get boundedQuantilesFn for QuantilesPerKey: %v", err)
	}
	quantiles := beam.CombinePerKey(s,
		boundedQuantilesFn,
		partialKV)
	// Finally, drop thresholded partitions.
	return beam.ParDo(s, dropThresholdedPartitionsFloat64SliceFn, quantiles)
}

func addPublicPartitionsForQuantiles(s beam.Scope, epsilon, delta float64, maxPartitionsContributed int64, params QuantilesParams, noiseKind noise.Kind, partialKV beam.PCollection, testMode testMode) beam.PCollection {
	// Compute the quantiles for each partition with non-public partitions dropped. Result is PCollection<partition, map[float64]float64>.
	boundedQuantilesFn, err := newBoundedQuantilesFn(boundedQuantilesFnParams{
		epsilon:                      epsilon,
		delta:                        delta,
		maxPartitionsContributed:     maxPartitionsContributed,
		maxContributionsPerPartition: params.MaxContributionsPerPartition,
		minValue:                     params.MinValue,
		maxValue:                     params.MaxValue,
		noiseKind:                    noiseKind,
		ranks:                        params.Ranks,
		publicPartitions:             true,
		testMode:                     testMode,
		emptyPartitions:              false})
	if err != nil {
		log.Fatalf("Couldn't get boundedQuantilesFn for QuantilesPerKey: %v", err)
	}
	quantiles := beam.CombinePerKey(s,
		boundedQuantilesFn,
		partialKV)
	partitionT, _ := beam.ValidateKVType(quantiles)
	quantilesPartitions := beam.DropValue(s, quantiles)
	// Create map with partitions in the data as keys.
	partitionMap := beam.Combine(s, newPartitionsMapFn(beam.EncodedType{partitionT.Type()}), quantilesPartitions)
	partitionsCol := params.PublicPartitions
	// Add value of empty array to each partition key in PublicPartitions.
	publicPartitionsWithValues := beam.ParDo(s, addDummyValuesToPublicPartitionsFloat64SliceFn, partitionsCol)
	// emptyPublicPartitions are the partitions that are public but not found in the data.
	emptyPublicPartitions := beam.ParDo(s, newEmitPartitionsNotInTheDataFn(partitionT), publicPartitionsWithValues, beam.SideInput{Input: partitionMap})
	// Compute DP quantiles for empty public partitions.
	boundedQuantilesFn, err = newBoundedQuantilesFn(boundedQuantilesFnParams{
		epsilon:                      epsilon,
		delta:                        delta,
		maxPartitionsContributed:     maxPartitionsContributed,
		maxContributionsPerPartition: params.MaxContributionsPerPartition,
		minValue:                     params.MinValue,
		maxValue:                     params.MaxValue,
		noiseKind:                    noiseKind,
		ranks:                        params.Ranks,
		publicPartitions:             true,
		testMode:                     testMode,
		emptyPartitions:              true})
	if err != nil {
		log.Fatalf("Couldn't get boundedQuantilesFn for QuantilesPerKey: %v", err)
	}
	emptyQuantiles := beam.CombinePerKey(s,
		boundedQuantilesFn,
		emptyPublicPartitions)
	// Merge quantiles from data with quantiles from the empty public partitions.
	allQuantiles := beam.Flatten(s, quantiles, emptyQuantiles)
	return allQuantiles
}

func checkQuantilesPerKeyParams(params QuantilesParams, epsilon, delta float64, noiseKind noise.Kind) error {
	err := checks.CheckEpsilon("pbeam.QuantilesPerKey", epsilon)
	if err != nil {
		return err
	}
	if (params.PublicPartitions).IsValid() && noiseKind == noise.LaplaceNoise {
		err = checks.CheckNoDelta("pbeam.QuantilesPerKey", delta)
	} else {
		err = checks.CheckDeltaStrict("pbeam.QuantilesPerKey", delta)
	}
	if err != nil {
		return err
	}
	err = checks.CheckBoundsFloat64("pbeam.QuantilesPerKey", params.MinValue, params.MaxValue)
	if err != nil {
		return err
	}
	err = checks.CheckBoundsNotEqual("pbeam.QuantilesPerKey", params.MinValue, params.MaxValue)
	if err != nil {
		return err
	}
	if len(params.Ranks) == 0 {
		return fmt.Errorf("QuantilesPerKey requires at least one rank to compute")
	}
	for _, rank := range params.Ranks {
		if rank < 0.0 || rank > 1.0 {
			return fmt.Errorf("rank %f must be >= 0 and <= 1", rank)
		}
	}
	return checks.CheckMaxPartitionsContributed("pbeam.QuantilesPerKey", params.MaxPartitionsContributed)
}

type boundedQuantilesAccum struct {
	BQ               *dpagg.BoundedQuantiles
	SP               *dpagg.PreAggSelectPartition
	PublicPartitions bool
}

// boundedQuantilesFn is a differentially private combineFn for computing quantiles of values. Do not
// initialize it yourself, use newBoundedQuantilesFn to create a boundedQuantilesFn instance.
type boundedQuantilesFn struct {
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
	Ranks                        []float64
	PublicPartitions             bool // Set to true if public partitions are used.
	TestMode                     testMode
	EmptyPartitions              bool // Set to true if this combineFn is for adding noise to empty public partitions.
}

// boundedQuantilesFnParams contains the parameters for creating a new boundedQuantilesFn.
type boundedQuantilesFnParams struct {
	epsilon                      float64
	delta                        float64
	maxPartitionsContributed     int64
	maxContributionsPerPartition int64
	minValue                     float64
	maxValue                     float64
	noiseKind                    noise.Kind
	ranks                        []float64
	publicPartitions             bool // Set to true if public partitions are used.
	testMode                     testMode
	emptyPartitions              bool // Set to true if the boundedQuantilesFn is for adding noise to empty public partitions.
}

// newBoundedQuantilesFn returns a boundedQuantilesFn with the given budget and parameters.
func newBoundedQuantilesFn(params boundedQuantilesFnParams) (*boundedQuantilesFn, error) {
	fn := &boundedQuantilesFn{
		MaxPartitionsContributed:     params.maxPartitionsContributed,
		MaxContributionsPerPartition: params.maxContributionsPerPartition,
		Lower:                        params.minValue,
		Upper:                        params.maxValue,
		NoiseKind:                    params.noiseKind,
		Ranks:                        params.ranks,
		PublicPartitions:             params.publicPartitions,
		TestMode:                     params.testMode,
		EmptyPartitions:              params.emptyPartitions,
	}
	if fn.PublicPartitions {
		fn.NoiseEpsilon = params.epsilon
		fn.NoiseDelta = params.delta
		return fn, nil
	}
	fn.NoiseEpsilon = params.epsilon / 2
	fn.PartitionSelectionEpsilon = params.epsilon - fn.NoiseEpsilon
	switch params.noiseKind {
	case noise.GaussianNoise:
		fn.NoiseDelta = params.delta / 2
	case noise.LaplaceNoise:
		fn.NoiseDelta = 0
	default:
		return nil, fmt.Errorf("unknown noise.Kind (%v) is specified. Please specify a valid noise", params.noiseKind)
	}
	fn.PartitionSelectionDelta = params.delta - fn.NoiseDelta
	return fn, nil
}

func (fn *boundedQuantilesFn) Setup() {
	fn.noise = noise.ToNoise(fn.NoiseKind)
	if fn.TestMode.isEnabled() {
		fn.noise = noNoise{}
	}
}

func (fn *boundedQuantilesFn) CreateAccumulator() boundedQuantilesAccum {
	accum := boundedQuantilesAccum{
		BQ: dpagg.NewBoundedQuantiles(&dpagg.BoundedQuantilesOptions{
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

func (fn *boundedQuantilesFn) AddInput(a boundedQuantilesAccum, values []float64) boundedQuantilesAccum {
	// We can have multiple values for each (privacy_key, partition_key) pair.
	// We need to add each value to BoundedQuantiles as input but we need to add a single input
	// for each privacy_key to SelectPartition.
	for _, v := range values {
		a.BQ.Add(v)
	}
	if !fn.PublicPartitions {
		a.SP.Increment()
	}
	return a
}

func (fn *boundedQuantilesFn) MergeAccumulators(a, b boundedQuantilesAccum) boundedQuantilesAccum {
	a.BQ.Merge(b.BQ)
	if !fn.PublicPartitions {
		a.SP.Merge(b.SP)
	}
	return a
}

func (fn *boundedQuantilesFn) ExtractOutput(a boundedQuantilesAccum) []float64 {
	if fn.TestMode.isEnabled() {
		a.BQ.Noise = noNoise{}
	}
	if fn.TestMode.isEnabled() || a.PublicPartitions || a.SP.ShouldKeepPartition() {
		result := make([]float64, len(fn.Ranks))
		for i, rank := range fn.Ranks {
			result[i] = a.BQ.Result(rank)
		}
		return result
	}
	return nil
}

func (fn *boundedQuantilesFn) String() string {
	return fmt.Sprintf("%#v", fn)
}
