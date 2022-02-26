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
	"github.com/google/differential-privacy/go/checks"
	"github.com/google/differential-privacy/go/noise"
	"github.com/google/differential-privacy/privacy-on-beam/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/stats"
)

// CountParams specifies the parameters associated with a Count aggregation.
type CountParams struct {
	// Noise type (which is either LaplaceNoise{} or GaussianNoise{}).
	//
	// Defaults to LaplaceNoise{}.
	NoiseKind NoiseKind
	// Differential privacy budget consumed by this aggregation. If there is
	// only one aggregation, both Epsilon and Delta can be left 0; in that
	// case, the entire budget of the PrivacySpec is consumed.
	Epsilon, Delta float64
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
	// The maximum number of times that a privacy identifier can contribute to
	// a single count (or, equivalently, the maximum value that a privacy
	// identifier can add to a single count in total). If MaxValue=10 and a
	// privacy identifier is associated with the same value in 15 records, Count
	// ignores 5 of these records and only adds 10 to the count for this value.
	// There is an inherent trade-off when choosing MaxValue: a larger
	// parameter means that fewer records are lost, but a larger noise is added.
	//
	// Required.
	MaxValue int64
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

// Count counts the number of times a value appears in a PrivatePCollection,
// adding differentially private noise to the counts and doing pre-aggregation
// thresholding to remove counts with a low number of distinct privacy
// identifiers.
//
// It is also possible to manually specify the list of partitions
// present in the output, in which case the partition selection/thresholding
// step is skipped.
//
// Note: Do not use when your results may cause overflows for int64 values.
// This aggregation is not hardened for such applications yet.
//
// Count transforms a PrivatePCollection<V> into a PCollection<V, int64>.
func Count(s beam.Scope, pcol PrivatePCollection, params CountParams) beam.PCollection {
	s = s.Scope("pbeam.Count")
	// Obtain type information from the underlying PCollection<K,V>.
	idT, partitionT := beam.ValidateKVType(pcol.col)

	// Get privacy parameters.
	spec := pcol.privacySpec
	epsilon, delta, err := spec.consumeBudget(params.Epsilon, params.Delta)
	if err != nil {
		log.Fatalf("Couldn't consume budget for Count: %v", err)
	}

	var noiseKind noise.Kind
	if params.NoiseKind == nil {
		noiseKind = noise.LaplaceNoise
		log.Infof("No NoiseKind specified, using Laplace Noise by default.")
	} else {
		noiseKind = params.NoiseKind.toNoiseKind()
	}
	err = checkCountParams(params, epsilon, delta, noiseKind, partitionT.Type())
	if err != nil {
		log.Fatalf("pbeam.Count: %v", err)
	}

	maxPartitionsContributed, err := getMaxPartitionsContributed(spec, params.MaxPartitionsContributed)
	if err != nil {
		log.Fatalf("Couldn't get MaxPartitionsContributed for Count: %v", err)
	}

	// Drop non-public partitions, if public partitions are specified.
	pcol.col, err = dropNonPublicPartitions(s, pcol, params.PublicPartitions, partitionT.Type())
	if err != nil {
		log.Fatalf("Couldn't drop non-public partitions for Count: %v", err)
	}

	// First, encode KV pairs, count how many times each one appears,
	// and re-key by the original privacy key.
	coded := beam.ParDo(s, kv.NewEncodeFn(idT, partitionT), pcol.col)
	kvCounts := stats.Count(s, coded)
	counts64 := beam.ParDo(s, vToInt64Fn, kvCounts)
	rekeyed := beam.ParDo(s, rekeyInt64Fn, counts64)
	// Second, do cross-partition contribution bounding if not in test mode without contribution bounding.
	if spec.testMode != noNoiseWithoutContributionBounding {
		rekeyed = boundContributions(s, rekeyed, maxPartitionsContributed)
	}
	// Third, now that contribution bounding is done, remove the privacy keys,
	// decode the value, and sum all the counts bounded by maxCountContrib.
	countPairs := beam.DropKey(s, rekeyed)
	decodePairInt64Fn := newDecodePairInt64Fn(partitionT.Type())
	countsKV := beam.ParDo(s,
		decodePairInt64Fn,
		countPairs,
		beam.TypeDefinition{Var: beam.XType, T: partitionT.Type()})
	// Add public partitions and return the aggregation output, if public partitions are specified.
	if params.PublicPartitions != nil {
		return addPublicPartitionsForCount(s, epsilon, delta, maxPartitionsContributed, params, noiseKind, countsKV, spec.testMode)
	}
	boundedSumInt64Fn, err := newBoundedSumInt64Fn(epsilon, delta, maxPartitionsContributed, 0, params.MaxValue, noiseKind, false, spec.testMode)
	if err != nil {
		log.Fatalf("Couldn't get boundedSumInt64Fn for Count: %v", err)
	}
	sums := beam.CombinePerKey(s,
		boundedSumInt64Fn,
		countsKV)
	// Drop thresholded partitions.
	counts := beam.ParDo(s, dropThresholdedPartitionsInt64Fn, sums)
	// Clamp negative counts to zero and return.
	return beam.ParDo(s, clampNegativePartitionsInt64Fn, counts)
}

func checkCountParams(params CountParams, epsilon, delta float64, noiseKind noise.Kind, partitionType reflect.Type) error {
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
	if params.MaxValue <= 0 {
		return fmt.Errorf("MaxValue should be strictly positive, got %d", params.MaxValue)
	}
	return nil
}

func addPublicPartitionsForCount(s beam.Scope, epsilon, delta float64, maxPartitionsContributed int64, params CountParams, noiseKind noise.Kind, countsKV beam.PCollection, testMode testMode) beam.PCollection {
	// Turn PublicPartitions from PCollection<K> into PCollection<K, int64> by adding
	// the value zero to each K.
	publicPartitions, isPCollection := params.PublicPartitions.(beam.PCollection)
	if !isPCollection {
		publicPartitions = beam.Reshuffle(s, beam.CreateList(s, params.PublicPartitions))
	}
	emptyCounts := beam.ParDo(s, addZeroValuesToPublicPartitionsInt64Fn, publicPartitions)
	// Merge countsKV and emptyCounts.
	allPartitions := beam.Flatten(s, emptyCounts, countsKV)
	// Sum and add noise.
	boundedSumInt64Fn, err := newBoundedSumInt64Fn(epsilon, delta, maxPartitionsContributed, 0, params.MaxValue, noiseKind, true, testMode)
	if err != nil {
		log.Fatalf("Couldn't get boundedSumInt64Fn for Count: %v", err)
	}
	sums := beam.CombinePerKey(s, boundedSumInt64Fn, allPartitions)
	finalPartitions := beam.ParDo(s, dereferenceValueToInt64Fn, sums)
	// Clamp negative counts to zero and return.
	return beam.ParDo(s, clampNegativePartitionsInt64Fn, finalPartitions)
}
