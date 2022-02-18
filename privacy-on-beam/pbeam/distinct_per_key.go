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
	"github.com/google/differential-privacy/go/checks"
	"github.com/google/differential-privacy/go/noise"
	"github.com/google/differential-privacy/privacy-on-beam/internal/kv"
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
	// only one aggregation, both Epsilon and Delta can be left 0; in that
	// case, the entire budget of the PrivacySpec is consumed.
	Epsilon, Delta float64
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
// DistinctPerKey does not support public partitions yet.
//
// Note: Do not use when your results may cause overflows for Int64 values.
// This aggregation is not hardened for such applications yet.
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
	spec := pcol.privacySpec
	maxPartitionsContributed, err := getMaxPartitionsContributed(spec, params.MaxPartitionsContributed)
	if err != nil {
		log.Fatalf("Couldn't get MaxPartitionsContributed for DistinctPerKey: %v", err)
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
	epsilon, delta, err := spec.getBudget(params.Epsilon, params.Delta)
	if err != nil {
		log.Fatalf("Couldn't consume budget for DistinctPerKey: %v", err)
	}
	err = checkDistinctPerKeyParams(params, epsilon, delta, maxPartitionsContributed)
	if err != nil {
		log.Fatalf("pbeam.DistinctPerKey: %v", err)
	}

	// Do initial per- and cross-partition contribution bounding and swap kv.Pair<K,V> and ID.
	// This is not great in terms of utility, since dropping contributions randomly might
	// mean that we keep duplicates instead of distinct values. However, this is necessary
	// for the current algorithm to be DP.
	if spec.testMode != noNoiseWithoutContributionBounding {
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

		flattened := beam.ParDo(s, flattenValuesFn, rekeyed) // PCollection<kv.Pair{ID,K}, codedV>, where codedV=[]byte

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
	noiseEpsilon, partitionSelectionEpsilon, noiseDelta, partitionSelectionDelta := splitBudget(epsilon, delta, noiseKind)
	partitions := SelectPartitions(s, pcol, SelectPartitionsParams{Epsilon: partitionSelectionEpsilon, Delta: partitionSelectionDelta, MaxPartitionsContributed: params.MaxPartitionsContributed})

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
		Epsilon:                  noiseEpsilon,
		Delta:                    noiseDelta,
		MaxPartitionsContributed: params.MaxPartitionsContributed,
		MaxValue:                 params.MaxContributionsPerPartition,
		PublicPartitions:         partitions,
	})
}

// splitBudget splits the privacy budget between adding noise and partition selection for DistinctPerKey.
func splitBudget(epsilon, delta float64, noiseKind noise.Kind) (noiseEpsilon float64, partitionSelectionEpsilon float64, noiseDelta float64, partitionSelectionDelta float64) {
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
	return noiseEpsilon, partitionSelectionEpsilon, noiseDelta, partitionSelectionDelta
}

func checkDistinctPerKeyParams(params DistinctPerKeyParams, epsilon, delta float64, maxPartitionsContributed int64) error {
	err := checks.CheckEpsilon(epsilon)
	if err != nil {
		return err
	}
	err = checks.CheckDeltaStrict(delta)
	if err != nil {
		return err
	}
	return checks.CheckMaxContributionsPerPartition(params.MaxContributionsPerPartition)
}
