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

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/v3/dpagg"
	"github.com/google/differential-privacy/go/v3/noise"
	"github.com/google/differential-privacy/privacy-on-beam/v3/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/register"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/filter"
)

func init() {
	register.Combiner3[countAccum, int64, *int64](&countFn{})

	register.Function1x2[beam.V, beam.V, int64](addOneValueFn)
}

// DistinctPrivacyIDParams specifies the parameters associated with a
// DistinctPrivacyID aggregation.
type DistinctPrivacyIDParams struct {
	// Noise type (which is either LaplaceNoise{} or GaussianNoise{}).
	//
	// Defaults to LaplaceNoise{}.
	NoiseKind NoiseKind
	// Differential privacy budget consumed by this aggregation. If there is
	// only one aggregation, both epsilon and delta can be left 0; in that case
	// the entire budget reserved for aggregation in the PrivacySpec is consumed.
	AggregationEpsilon, AggregationDelta float64
	// Differential privacy budget consumed by partition selection of this
	// aggregation. Note that DistinctPrivacyID doesn't consume epsilon for
	// partition selection, so you can only specify a delta.
	//
	// If PublicPartitions are specified, this needs to be left unset.
	//
	// If there is only one aggregation, this can be left unset; in that case
	// the entire budget reserved for partition selection in the PrivacySpec
	// is consumed.
	//
	// Optional.
	PartitionSelectionDelta float64
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
	// If PartitionSelectionDelta is specified, this needs to be left unset.
	//
	// Optional.
	PublicPartitions any
	// The maximum number of distinct values that a given privacy identifier
	// can influence. If a privacy identifier is associated with more values,
	// random values will be dropped. There is an inherent trade-off when
	// choosing this parameter: a larger MaxPartitionsContributed leads to less
	// data loss due to contribution bounding, but since the noise added in
	// aggregations is scaled according to maxPartitionsContributed, it also
	// means that more noise is added to each count.
	//
	// Required.
	MaxPartitionsContributed int64
}

// DistinctPrivacyID counts the number of distinct privacy identifiers
// associated with each value in a PrivatePCollection, adding differentially
// private noise to the counts and doing post-aggregation thresholding to
// remove low counts. It is conceptually equivalent to calling Count with
// MaxValue=1, but is specifically optimized for this use case.
//
// It is also possible to manually specify the list of partitions
// present in the output, in which case the partition selection/thresholding
// step is skipped.
//
// DistinctPrivacyID transforms a PrivatePCollection<V> into a
// PCollection<V,int64>.
//
// Note: Do not use when your results may cause overflows for int64 values.
// This aggregation is not hardened for such applications yet.
func DistinctPrivacyID(s beam.Scope, pcol PrivatePCollection, params DistinctPrivacyIDParams) beam.PCollection {
	s = s.Scope("pbeam.DistinctPrivacyID")
	// Obtain type information from the underlying PCollection<K,V>.
	idT, partitionT := beam.ValidateKVType(pcol.col)

	var noiseKind noise.Kind
	if params.NoiseKind == nil {
		noiseKind = noise.LaplaceNoise
		log.Infof("No NoiseKind specified, using Laplace Noise by default.")
	} else {
		noiseKind = params.NoiseKind.toNoiseKind()
	}
	// Get privacy parameters.
	spec := pcol.privacySpec
	var err error
	params.AggregationEpsilon, params.AggregationDelta, err = spec.aggregationBudget.consume(params.AggregationEpsilon, params.AggregationDelta)
	if err != nil {
		log.Fatalf("Couldn't consume aggregation budget for DistinctPrivacyID: %v", err)
	}
	if params.PublicPartitions == nil {
		_, params.PartitionSelectionDelta, err = spec.partitionSelectionBudget.consume(0, params.PartitionSelectionDelta)
		if err != nil {
			log.Fatalf("Couldn't consume partition selection budget for DistinctPrivacyID: %v", err)
		}
	}

	err = checkDistinctPrivacyIDParams(params, noiseKind, partitionT.Type())
	if err != nil {
		log.Fatalf("pbeam.DistinctPrivacyID: %v", err)
	}

	// Drop non-public partitions, if public partitions are specified.
	pcol.col, err = dropNonPublicPartitions(s, pcol, params.PublicPartitions, partitionT.Type())
	if err != nil {
		log.Fatalf("Couldn't drop non-public partitions for DistinctPrivacyID: %v", err)
	}

	// First, deduplicate KV pairs by encoding them and calling Distinct.
	coded := beam.ParDo(s, kv.NewEncodeFn(idT, partitionT), pcol.col)
	distinct := filter.Distinct(s, coded)
	decoded := beam.ParDo(s,
		kv.NewDecodeFn(idT, partitionT),
		distinct,
		beam.TypeDefinition{Var: beam.TType, T: idT.Type()},
		beam.TypeDefinition{Var: beam.VType, T: partitionT.Type()})
	// Second, do cross-partition contribution bounding if not in test mode without contribution bounding.
	if spec.testMode != TestModeWithoutContributionBounding {
		decoded = boundContributions(s, decoded, params.MaxPartitionsContributed)
	}
	// Third, now that KV pairs are deduplicated and contribution bounding is
	// done, remove the keys and count how many times each value appears.
	values := beam.DropKey(s, decoded)
	emptyCounts := beam.ParDo(s, addOneValueFn, values)

	var result beam.PCollection
	// Add public partitions and return the aggregation output, if public partitions are specified.
	if params.PublicPartitions != nil {
		result = addPublicPartitionsForDistinctID(s, *spec, params, noiseKind, emptyCounts)
	} else {
		countFn, err := newCountFn(*spec, params, noiseKind, false)
		if err != nil {
			log.Fatalf("pbeam.DistinctPrivacyID: %v", err)
		}
		noisedCounts := beam.CombinePerKey(s, countFn, emptyCounts)
		// Drop thresholded partitions.
		result = beam.ParDo(s, dropThresholdedPartitionsInt64, noisedCounts)
	}

	// Clamp negative counts to zero and return.
	result = beam.ParDo(s, clampNegativePartitionsInt64, result)

	return result
}

func addPublicPartitionsForDistinctID(s beam.Scope, spec PrivacySpec, params DistinctPrivacyIDParams, noiseKind noise.Kind, countsKV beam.PCollection) beam.PCollection {
	publicPartitions, isPCollection := params.PublicPartitions.(beam.PCollection)
	if !isPCollection {
		publicPartitions = beam.Reshuffle(s, beam.CreateList(s, params.PublicPartitions))
	}
	prepareAddPublicPartitions := beam.ParDo(s, addZeroValuesToPublicPartitionsInt64, publicPartitions)
	// Merge countsKV and prepareAddPublicPartitions.
	allAddPartitions := beam.Flatten(s, countsKV, prepareAddPublicPartitions)
	countFn, err := newCountFn(spec, params, noiseKind, true)
	if err != nil {
		log.Fatalf("pbeam.DistinctPrivacyID: %v", err)
	}
	noisedCounts := beam.CombinePerKey(s, countFn, allAddPartitions)
	finalPartitions := beam.ParDo(s, dereferenceValueInt64, noisedCounts)
	// Clamp negative counts to zero and return.
	return beam.ParDo(s, clampNegativePartitionsInt64, finalPartitions)
}

func checkDistinctPrivacyIDParams(params DistinctPrivacyIDParams, noiseKind noise.Kind, partitionType reflect.Type) error {
	err := checkPublicPartitions(params.PublicPartitions, partitionType)
	if err != nil {
		return err
	}
	err = checkAggregationEpsilon(params.AggregationEpsilon)
	if err != nil {
		return err
	}
	err = checkAggregationDelta(params.AggregationDelta, noiseKind)
	if err != nil {
		return err
	}
	err = checkPartitionSelectionDelta(params.PartitionSelectionDelta, params.PublicPartitions)
	if err != nil {
		return err
	}
	return checkMaxPartitionsContributed(params.MaxPartitionsContributed)
}

func addOneValueFn(v beam.V) (beam.V, int64) {
	return v, 1
}

// countFn is the combineFn used for counting the number of privacy units per partition.
type countFn struct {
	// Privacy spec parameters (set during initial construction).
	Epsilon                  float64
	NoiseDelta               float64
	ThresholdDelta           float64
	PreThreshold             int64
	MaxPartitionsContributed int64
	NoiseKind                noise.Kind
	noise                    noise.Noise // Set during Setup phase according to NoiseKind.
	PublicPartitions         bool
	TestMode                 TestMode
}

// newCountFn returns a newCountFn with the given budget and parameters.
func newCountFn(spec PrivacySpec, params DistinctPrivacyIDParams, noiseKind noise.Kind, publicPartitions bool) (*countFn, error) {
	if noiseKind != noise.GaussianNoise && noiseKind != noise.LaplaceNoise {
		return nil, fmt.Errorf("unknown noise.Kind (%v) is specified. Please specify a valid noise", noiseKind)
	}
	return &countFn{
		Epsilon:                  params.AggregationEpsilon,
		NoiseDelta:               params.AggregationDelta,
		ThresholdDelta:           params.PartitionSelectionDelta,
		PreThreshold:             spec.preThreshold,
		MaxPartitionsContributed: params.MaxPartitionsContributed,
		NoiseKind:                noiseKind,
		PublicPartitions:         publicPartitions,
		TestMode:                 spec.testMode,
	}, nil
}

func (fn *countFn) Setup() {
	fn.noise = noise.ToNoise(fn.NoiseKind)
	if fn.TestMode.isEnabled() {
		fn.noise = noNoise{}
	}
}

type countAccum struct {
	C                *dpagg.Count
	PublicPartitions bool
}

func (fn *countFn) CreateAccumulator() (countAccum, error) {
	c, err := dpagg.NewCount(&dpagg.CountOptions{
		Epsilon:                  fn.Epsilon,
		Delta:                    fn.NoiseDelta,
		MaxPartitionsContributed: fn.MaxPartitionsContributed,
		Noise:                    fn.noise,
	})
	return countAccum{C: c, PublicPartitions: fn.PublicPartitions}, err
}

// AddInput increments the count by one for each contribution. Does nothing when the
// the value is 0, which is the case only for empty public partitions.
func (fn *countFn) AddInput(a countAccum, value int64) (countAccum, error) {
	var err error
	if value != 0 {
		err = a.C.Increment()
	}
	return a, err
}

func (fn *countFn) MergeAccumulators(a, b countAccum) (countAccum, error) {
	err := a.C.Merge(b.C)
	return a, err
}

func (fn *countFn) ExtractOutput(a countAccum) (*int64, error) {
	if fn.TestMode.isEnabled() {
		a.C.Noise = noNoise{}
	}
	if a.PublicPartitions {
		result, err := a.C.Result()
		return &result, err
	}
	return a.C.PreThresholdedResult(fn.PreThreshold, fn.ThresholdDelta)
}

func (fn *countFn) String() string {
	return fmt.Sprintf("%#v", fn)
}
