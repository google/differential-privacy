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
	"github.com/google/differential-privacy/go/v2/checks"
	"github.com/google/differential-privacy/go/v2/dpagg"
	"github.com/google/differential-privacy/go/v2/noise"
	"github.com/google/differential-privacy/privacy-on-beam/v2/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
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
	// A single contribution of a given privacy identifier to partition can be
	// at at least MinValue, and at most MaxValue; otherwise it will be clamped
	// to these bounds. There is an inherent trade-off when choosing MinValue and
	// MaxValue: a small MinValue and a large MaxValue means that less records
	// will be clamped, but that more noise will be added.
	// For example, if a privacy identifier is associated with the key-value
	// pairs [("a", -5), ("a", 2), ("b", 7), ("c", 3)] and the (MinValue, MaxValue)
	// bounds are (0, 5), the contribution for "a" will be clamped up to 0,
	// the contribution for "b" will be clamped down to 5, and the contribution
	// for "c" will be untouched.
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
	// 	partitions. For example, you could use the output of a SelectPartitions
	//  operation or the keys of a DistinctPrivacyID operation as the list of
	//  public partitions.
	//
	// PublicPartitions needs to be a beam.PCollection, slice, or array. The
	// underlying type needs to match the partition type of the PrivatePCollection.
	//
	// Prefer slices or arrays if the list of public partitions is small and
	// can fit into memory (e.g., up to a million). Prefer beam.PCollection
	// otherwise.
	//
	// Optional.
	PublicPartitions interface{}
}

// QuantilesPerKey computes one or multiple quantiles of the values associated with each
// key in a PrivatePCollection<K,V>, adding differentially private noise to the quantiles
// and doing pre-aggregation thresholding to remove partitions with a low number of
// distinct privacy identifiers.
//
// It is also possible to manually specify the list of partitions
// present in the output, in which case the partition selection/thresholding
// step is skipped.
//
// QuantilesPerKey transforms a PrivatePCollection<K,V> into a PCollection<K,[]float64>.
//
// Note that due to the implementation details of the internal Quantiles algorithm, using pbeamtest
// with QuantilesPerKey has two caveats:
//
//  1. Even without DP noise, the output will be slightly noisy. You can use
//     pbeamtest.QuantilesTolerance() to account for that noise.
//  2. It is not possible to not clamp input values when using
//     pbeamtest.NewPrivacySpecNoNoiseWithoutContributionBounding(), so clamping to Min/MaxValue will
//     still be applied. However, MaxContributionsPerPartition and MaxPartitionsContributed contribution
//     bounding will be disabled.
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
	err = checkQuantilesPerKeyParams(params, epsilon, delta, noiseKind, pcol.codec.KType.T)
	if err != nil {
		log.Fatalf("pbeam.QuantilesPerKey: %v", err)
	}

	// Drop non-public partitions, if public partitions are specified.
	pcol.col, err = dropNonPublicPartitions(s, pcol, params.PublicPartitions, pcol.codec.KType.T)
	if err != nil {
		log.Fatalf("Couldn't drop non-public partitions for Quantiles: %v", err)
	}

	// First, group together the privacy ID and the partition ID and do per-partition contribution bounding.
	// Result is PCollection<kv.Pair{ID,K},V>
	decoded := beam.ParDo(s,
		newEncodeIDKFn(idT, pcol.codec),
		pcol.col,
		beam.TypeDefinition{Var: beam.VType, T: pcol.codec.VType.T})

	// Don't do per-partition contribution bounding if in test mode without contribution bounding.
	if spec.testMode != noNoiseWithoutContributionBounding {
		decoded = boundContributions(s, decoded, params.MaxContributionsPerPartition)
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

	var result beam.PCollection
	// Add public partitions and return the aggregation output, if public partitions are specified.
	if params.PublicPartitions != nil {
		result = addPublicPartitionsForQuantiles(s, epsilon, delta, maxPartitionsContributed,
			params, noiseKind, partialKV, spec.testMode)
	} else {
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
		result = beam.ParDo(s, dropThresholdedPartitionsFloat64SliceFn, quantiles)
	}
	return result
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
	publicPartitions, isPCollection := params.PublicPartitions.(beam.PCollection)
	if !isPCollection {
		publicPartitions = beam.Reshuffle(s, beam.CreateList(s, params.PublicPartitions))
	}
	// Add value of empty array to each partition key in PublicPartitions.
	publicPartitionsWithValues := beam.ParDo(s, addEmptySliceToPublicPartitionsFloat64Fn, publicPartitions)
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
	// Merge quantiles from data with quantiles from the empty public partitions and return.
	return beam.Flatten(s, quantiles, emptyQuantiles)
}

func checkQuantilesPerKeyParams(params QuantilesParams, epsilon, delta float64, noiseKind noise.Kind, partitionType reflect.Type) error {
	err := checkPublicPartitions(params.PublicPartitions, partitionType)
	if err != nil {
		return err
	}
	err = checks.CheckEpsilon(epsilon)
	if err != nil {
		return err
	}
	err = checkDelta(delta, noiseKind, params.PublicPartitions)
	if err != nil {
		return err
	}
	err = checks.CheckBoundsFloat64(params.MinValue, params.MaxValue)
	if err != nil {
		return err
	}
	err = checks.CheckBoundsNotEqual(params.MinValue, params.MaxValue)
	if err != nil {
		return err
	}
	if len(params.Ranks) == 0 {
		return fmt.Errorf("there should at least be one rank to compute")
	}
	for i, rank := range params.Ranks {
		if rank < 0.0 || rank > 1.0 {
			return fmt.Errorf("Ranks[%d]=%f must be >= 0 and <= 1", i, rank)
		}
	}
	return checks.CheckMaxContributionsPerPartition(params.MaxContributionsPerPartition)
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

func (fn *boundedQuantilesFn) CreateAccumulator() (boundedQuantilesAccum, error) {
	var bq *dpagg.BoundedQuantiles
	var err error
	bq, err = dpagg.NewBoundedQuantiles(&dpagg.BoundedQuantilesOptions{
		Epsilon:                      fn.NoiseEpsilon,
		Delta:                        fn.NoiseDelta,
		MaxPartitionsContributed:     fn.MaxPartitionsContributed,
		MaxContributionsPerPartition: fn.MaxContributionsPerPartition,
		Lower:                        fn.Lower,
		Upper:                        fn.Upper,
		Noise:                        fn.noise,
	})
	if err != nil {
		return boundedQuantilesAccum{}, err
	}
	accum := boundedQuantilesAccum{BQ: bq, PublicPartitions: fn.PublicPartitions}
	if !fn.PublicPartitions {
		accum.SP, err = dpagg.NewPreAggSelectPartition(&dpagg.PreAggSelectPartitionOptions{
			Epsilon:                  fn.PartitionSelectionEpsilon,
			Delta:                    fn.PartitionSelectionDelta,
			MaxPartitionsContributed: fn.MaxPartitionsContributed,
		})
	}
	return accum, err
}

func (fn *boundedQuantilesFn) AddInput(a boundedQuantilesAccum, values []float64) (boundedQuantilesAccum, error) {
	var err error
	// We can have multiple values for each (privacy_key, partition_key) pair.
	// We need to add each value to BoundedQuantiles as input but we need to add a single input
	// for each privacy_key to SelectPartition.
	for _, v := range values {
		err = a.BQ.Add(v)
		if err != nil {
			return a, err
		}
	}
	if !fn.PublicPartitions {
		err = a.SP.Increment()
	}
	return a, err
}

func (fn *boundedQuantilesFn) MergeAccumulators(a, b boundedQuantilesAccum) (boundedQuantilesAccum, error) {
	var err error
	err = a.BQ.Merge(b.BQ)
	if err != nil {
		return a, err
	}
	if !fn.PublicPartitions {
		err = a.SP.Merge(b.SP)
	}
	return a, err
}

func (fn *boundedQuantilesFn) ExtractOutput(a boundedQuantilesAccum) ([]float64, error) {
	if fn.TestMode.isEnabled() {
		a.BQ.Noise = noNoise{}
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
		result := make([]float64, len(fn.Ranks))
		for i, rank := range fn.Ranks {
			result[i], err = a.BQ.Result(rank)
			if err != nil {
				return nil, err
			}
		}
		return result, nil
	}
	return nil, nil
}

func (fn *boundedQuantilesFn) String() string {
	return fmt.Sprintf("%#v", fn)
}
