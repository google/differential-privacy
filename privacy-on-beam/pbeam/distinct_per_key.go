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
	"github.com/apache/beam/sdks/go/pkg/beam"
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
	_, kvT := beam.ValidateKVType(pcol.col)
	if kvT.Type() != reflect.TypeOf(kv.Pair{}) {
		log.Exitf("DistinctPerKey must be used on a PrivatePCollection of type <K,V>, got type %v instead", kvT)
	}
	if pcol.codec == nil {
		log.Exitf("DistinctPerKey: no codec found for the input PrivatePCollection.")
	}
	spec := pcol.privacySpec
	maxPartitionsContributed := getMaxPartitionsContributed(spec, params.MaxPartitionsContributed)

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
		log.Exitf("Couldn't consume budget for DistinctPerKey: %v", err)
	}
	checkDistinctPerKeyParams(params, epsilon, delta, maxPartitionsContributed)

	// Perform partition selection
	noiseEpsilon, partitionSelectionEpsilon, noiseDelta, partitionSelectionDelta := splitBudget(epsilon, delta, noiseKind)
	partitions := SelectPartitions(s, pcol, SelectPartitionsParams{Epsilon: partitionSelectionEpsilon, Delta: partitionSelectionDelta, MaxPartitionsContributed: params.MaxPartitionsContributed})

	// Deduplicate (partitionKey,value) pairs across users.
	rekeyed := beam.SwapKV(s, pcol.col) // PCollection<kv.Pair{K,V}, ID>.
	// Only keep one privacyKey per (partitionKey,value) pair.
	sampled := boundContributions(s, rekeyed, 1)

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
		log.Exitf("splitBudget: unknown noise.Kind (%v) is specified. Please specify a valid noise.", noiseKind)
	}
	return noiseEpsilon, partitionSelectionEpsilon, noiseDelta, partitionSelectionDelta
}

func checkDistinctPerKeyParams(params DistinctPerKeyParams, epsilon, delta float64, maxPartitionsContributed int64) error {
	err := checks.CheckEpsilon("pbeam.DistinctPerKey", epsilon)
	if err != nil {
		return err
	}
	err = checks.CheckDeltaStrict("pbeam.DistinctPerKey", delta)
	if err != nil {
		return err
	}
	if params.MaxContributionsPerPartition <= 0 {
		return fmt.Errorf("pbeam.DistinctPerKey: MaxContributionsPerPartition should be strictly positive, got %d", params.MaxContributionsPerPartition)
	}
	return checks.CheckMaxPartitionsContributed("pbeam.DistinctPerKey", maxPartitionsContributed)
}
