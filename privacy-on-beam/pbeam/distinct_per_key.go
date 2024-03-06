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
	"reflect"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/v3/checks"
	"github.com/google/differential-privacy/go/v3/noise"
	"github.com/google/differential-privacy/privacy-on-beam/v3/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
)

// DistinctPerKeyParams specifies the parameters associated with a
// DistinctPerKeyParams aggregation.
type DistinctPerKeyParams struct {
	// Noise type (which is either LaplaceNoise{} or GaussianNoise{}).
	//
	// Defaults to LaplaceNoise{}.
	NoiseKind NoiseKind
	// Differential privacy budget consumed by this aggregation. If there is
	// only one aggregation, both epsilon and delta can be left 0; in that case
	// the entire budget reserved for aggregation in the PrivacySpec is consumed.
	AggregationEpsilon, AggregationDelta float64
	// Differential privacy budget consumed by partition selection of this
	// aggregation.
	//
	// If PublicPartitions are specified, this needs to be left unset.
	//
	// If there is only one aggregation, this can be left unset; in that case
	// the entire budget reserved for partition selection in the PrivacySpec
	// is consumed.
	//
	// Optional.
	PartitionSelectionParams PartitionSelectionParams
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
	// If PartitionSelectionParams are specified, this needs to be left unset.
	//
	// Optional.
	PublicPartitions any
	// The maximum number of distinct keys that a given privacy identifier
	// can influence. If a privacy identifier is associated to more keys,
	// random keys will be dropped. There is an inherent trade-off when
	// choosing this parameter: a larger MaxPartitionsContributed leads to less
	// data loss due to contribution bounding, but since the noise added in
	// aggregations is scaled according to maxPartitionsContributed, it also
	// means that more noise is added to each count.
	//
	// Required.
	MaxPartitionsContributed int64
	// The maximum number of distinct values a given privacy identifier can
	// contribute to for each key. There is an inherent trade-off when choosing this
	// parameter: a larger MaxContributionsPerPartition leads to less data loss due
	// to contribution bounding, but since the noise added in aggregations is
	// scaled according to maxContributionsPerPartition, it also means that more
	// noise is added to each mean.
	//
	// Required.
	MaxContributionsPerPartition int64
}

// DistinctPerKey estimates the number of distinct values associated to
// each key in a PrivatePCollection, adding differentially private noise
// to the estimates and doing pre-aggregation thresholding to remove
// estimates with a low number of distinct privacy identifiers.
//
// It is also possible to manually specify the list of partitions
// present in the output, in which case the partition selection/thresholding
// step is skipped.
//
// DistinctPerKey transforms a PrivatePCollection<K,V> into a
// PCollection<K,int64>.
func DistinctPerKey(s beam.Scope, pcol PrivatePCollection, params DistinctPerKeyParams) beam.PCollection {
	s = s.Scope("pbeam.DistinctPerKey")
	// Obtain type information from the underlying PCollection<K,V>.
	idT, kvT := beam.ValidateKVType(pcol.col)
	if kvT.Type() != reflect.TypeOf(kv.Pair{}) {
		log.Fatalf("DistinctPerKey must be used on a PrivatePCollection of type <K,V>, got type %v instead", kvT)
	}
	if pcol.codec == nil {
		log.Fatalf("DistinctPerKey: no codec found for the input PrivatePCollection.")
	}

	var noiseKind noise.Kind
	if params.NoiseKind == nil {
		noiseKind = noise.LaplaceNoise
		log.Infof("No NoiseKind specified, using Laplace Noise by default.")
	} else {
		noiseKind = params.NoiseKind.toNoiseKind()
	}

	// We get the total budget for DistinctPerKey with getBudget, split it and
	// consume it separately in partition selection and Count with consumeBudget.
	// In the new privacy budget API, budgets are already split.
	spec := pcol.privacySpec
	var err error
	params.AggregationEpsilon, params.AggregationDelta, err = spec.aggregationBudget.get(params.AggregationEpsilon, params.AggregationDelta)
	if err != nil {
		log.Fatalf("Couldn't get aggregation budget for DistinctPerKey: %v", err)
	}
	if params.PublicPartitions == nil {
		params.PartitionSelectionParams.Epsilon, params.PartitionSelectionParams.Delta, err = spec.partitionSelectionBudget.get(params.PartitionSelectionParams.Epsilon, params.PartitionSelectionParams.Delta)
		if err != nil {
			log.Fatalf("Couldn't get partition selection budget for DistinctPerKey: %v", err)
		}
	}
	err = checkDistinctPerKeyParams(params, noiseKind, pcol.codec.KType.T)
	if err != nil {
		log.Fatalf("pbeam.DistinctPerKey: %v", err)
	}

	// Drop non-public partitions, if public partitions are specified.
	pcol.col, err = dropNonPublicPartitions(s, pcol, params.PublicPartitions, pcol.codec.KType.T)
	if err != nil {
		log.Fatalf("Couldn't drop non-public partitions for DistinctPerKey: %v", err)
	}

	// Do initial per- and cross-partition contribution bounding and swap kv.Pair<K,V> and ID.
	// This is not great in terms of utility, since dropping contributions randomly might
	// mean that we keep duplicates instead of distinct values. However, this is necessary
	// for the current algorithm to be DP.
	if spec.testMode != TestModeWithoutContributionBounding {
		// First, rekey by kv.Pair{ID,K} and do per-partition contribution bounding.
		rekeyed := beam.ParDo(
			s,
			newEncodeIDKFn(idT, pcol.codec),
			pcol.col,
			beam.TypeDefinition{Var: beam.VType, T: pcol.codec.VType.T}) // PCollection<kv.Pair{ID,K}, V>.
		// Keep only maxContributionsPerPartition values per (privacyKey, partitionKey) pair.
		sampled := boundContributions(s, rekeyed, params.MaxContributionsPerPartition)

		// Collect all values per kv.Pair{ID,K} in a slice.
		combined := beam.CombinePerKey(s,
			newExpandValuesCombineFn(pcol.codec.VType),
			sampled) // PCollection<kv.Pair{ID,K}, []codedV}>, where codedV=[]byte

		_, codedVSliceType := beam.ValidateKVType(combined)

		decoded := beam.ParDo(
			s,
			newDecodeIDKFn(codedVSliceType, kv.NewCodec(idT.Type(), pcol.codec.KType.T)),
			combined,
			beam.TypeDefinition{Var: beam.WType, T: idT.Type()}) // PCollection<ID, kv.Pair{K,[]codedV}>, where codedV=[]byte

		// Second, do cross-partition contribution bounding.
		decoded = boundContributions(s, decoded, params.MaxPartitionsContributed)

		rekeyed = beam.ParDo(
			s,
			newEncodeIDKFn(idT, kv.NewCodec(pcol.codec.KType.T, codedVSliceType.Type())),
			decoded,
			beam.TypeDefinition{Var: beam.VType, T: codedVSliceType.Type()}) // PCollection<kv.Pair{ID,K}, []codedV>, where codedV=[]byte

		flattened := beam.ParDo(s, flattenValues, rekeyed) // PCollection<kv.Pair{ID,K}, codedV>, where codedV=[]byte

		pcol.col = beam.ParDo(
			s,
			newEncodeKVFn(kv.NewCodec(idT.Type(), pcol.codec.KType.T)),
			flattened,
			beam.TypeDefinition{Var: beam.WType, T: idT.Type()}) // PCollection<ID, kv.Pair{K,V}>
	}

	// Perform partition selection.
	// We do partition selection after cross-partition contribution bounding because
	// we want to keep the same contributions across partitions for partition selection
	// and Count.
	if params.PublicPartitions == nil {
		params.PublicPartitions = SelectPartitions(s, pcol, SelectPartitionsParams{
			Epsilon:                  params.PartitionSelectionParams.Epsilon,
			Delta:                    params.PartitionSelectionParams.Delta,
			MaxPartitionsContributed: params.MaxPartitionsContributed,
		})
	}

	// Keep only one privacyKey per (partitionKey, value) pair
	// (i.e. remove duplicate values for each partition).
	swapped := beam.SwapKV(s, pcol.col) // PCollection<kv.Pair{K,V}, ID>
	sampled := boundContributions(s, swapped, 1)

	// Drop V's, each <privacyKey, partitionKey> pair now corresponds to a unique V.
	sampled = beam.SwapKV(s, sampled) // PCollection<ID, kv.Pair{K,V}>.
	idK := beam.ParDo(s, &dropValuesFn{pcol.codec}, sampled, beam.TypeDefinition{Var: beam.WType, T: pcol.codec.VType.T})

	// Perform DP count.
	pcol.col = idK
	pcol.codec = nil
	return Count(s, pcol, CountParams{
		NoiseKind:                params.NoiseKind,
		AggregationEpsilon:       params.AggregationEpsilon,
		AggregationDelta:         params.AggregationDelta,
		MaxPartitionsContributed: params.MaxPartitionsContributed,
		MaxValue:                 params.MaxContributionsPerPartition,
		PublicPartitions:         params.PublicPartitions,
	})
}

// splitBudget splits the privacy budget between adding noise and partition selection for DistinctPerKey.
func splitBudget(epsilon, delta float64, noiseKind noise.Kind) (noiseEpsilon float64, noiseDelta float64, partitionSelectionEpsilon float64, partitionSelectionDelta float64) {
	noiseEpsilon = epsilon / 2
	partitionSelectionEpsilon = epsilon - noiseEpsilon
	switch noiseKind {
	case noise.GaussianNoise:
		noiseDelta = delta / 2
		partitionSelectionDelta = delta - noiseDelta
	case noise.LaplaceNoise:
		noiseDelta = 0
		partitionSelectionDelta = delta
	default:
		log.Fatalf("splitBudget: unknown noise.Kind (%v) is specified. Please specify a valid noise.", noiseKind)
	}
	return noiseEpsilon, noiseDelta, partitionSelectionEpsilon, partitionSelectionDelta
}

func checkDistinctPerKeyParams(params DistinctPerKeyParams, noiseKind noise.Kind, partitionType reflect.Type) error {
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
	err = checkPartitionSelectionEpsilon(params.PartitionSelectionParams.Epsilon, params.PublicPartitions)
	if err != nil {
		return err
	}
	err = checkPartitionSelectionDelta(params.PartitionSelectionParams.Delta, params.PublicPartitions)
	if err != nil {
		return err
	}
	err = checkMaxPartitionsContributedPartitionSelection(params.PartitionSelectionParams.MaxPartitionsContributed)
	if err != nil {
		return err
	}
	return checks.CheckMaxContributionsPerPartition(params.MaxContributionsPerPartition)
}
