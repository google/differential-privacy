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

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/checks"
	"github.com/google/differential-privacy/go/noise"
	"github.com/google/differential-privacy/privacy-on-beam/internal/kv"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/transforms/stats"
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
	// can influence. If a privacy identifier is associated to more values,
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
	// privacy identifier is associated to the same value in 15 records, Count
	// ignores 5 of these records and only adds 10 to the count for this value.
	// There is an inherent trade-off when choosing MaxValue: a larger
	// parameter means that less records are lost, but a larger noise.
	//
	// Required.
	MaxValue int64
}

// Count counts the number of times a value appears in a PrivatePCollection,
// adding differentially private noise to the counts and doing pre-aggregation
// thresholding to remove counts with a low number of distinct privacy
// identifiers.
//
// Note: Do not use when your results may cause overflows for Int64 values.
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
	err = checkCountParams(params, epsilon, delta)
	if err != nil {
		log.Exit(err)
	}

	if err != nil {
		log.Exitf("couldn't consume budget: %v", err)
	}
	var noiseKind noise.Kind
	if params.NoiseKind == nil {
		noiseKind = noise.LaplaceNoise
		log.Infof("No NoiseKind specified, using Laplace Noise by default.")
	} else {
		noiseKind = params.NoiseKind.toNoiseKind()
	}
	maxPartitionsContributed := getMaxPartitionsContributed(spec, params.MaxPartitionsContributed)
	// First, encode KV pairs, count how many times each one appears,
	// and re-key by the original privacy key.
	coded := beam.ParDo(s, kv.NewEncodeFn(idT, partitionT), pcol.col)
	kvCounts := stats.Count(s, coded)
	counts64 := beam.ParDo(s, vToInt64Fn, kvCounts)
	rekeyed := beam.ParDo(s, rekeyInt64Fn, counts64)
	// Second, do per-user contribution bounding.
	rekeyed = boundContributions(s, rekeyed, maxPartitionsContributed)
	// Third, now that contribution bounding is done, remove the privacy keys,
	// decode the value, and sum all the counts bounded by maxCountContrib.
	countPairs := beam.DropKey(s, rekeyed)
	countsKV := beam.ParDo(s,
		newDecodePairInt64Fn(partitionT.Type()),
		countPairs,
		beam.TypeDefinition{Var: beam.XType, T: partitionT.Type()})
	sums := beam.CombinePerKey(s,
		newBoundedSumInt64Fn(epsilon, delta, maxPartitionsContributed, 0, params.MaxValue, noiseKind),
		countsKV)
	// Drop thresholded partitions.
	counts := beam.ParDo(s, dropThresholdedPartitionsInt64Fn, sums)
	// Clamp negative counts to zero and return.
	return beam.ParDo(s, clampNegativePartitionsInt64Fn, counts)
}

func checkCountParams(params CountParams, epsilon, delta float64) error{
	err := checks.CheckEpsilon("pbeam.Count", epsilon)
	if err != nil {
		return err
	}
	err = checks.CheckDeltaStrict("pbeam.Count", delta)
	if err != nil {
		return err
	}
	err = checks.CheckMaxPartitionsContributed("pbeam.Count", params.MaxPartitionsContributed)
	if err != nil {
		return err
	}
	if params.MaxValue <= 0 {
		return fmt.Errorf("pbeam.Count: MaxValue should be strictly positive, got %d", params.MaxValue)
	}
	return nil
}