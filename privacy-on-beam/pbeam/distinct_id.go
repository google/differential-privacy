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
	"github.com/google/differential-privacy/go/dpagg"
	"github.com/google/differential-privacy/go/noise"
	"github.com/google/differential-privacy/privacy-on-beam/internal/kv"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/transforms/filter"
)

func init() {
	beam.RegisterType(reflect.TypeOf((*countFn)(nil)))
	// TODO: add tests to make sure we don't forget anything here
}

// DistinctPrivacyIDParams specifies the parameters associated with a
// DistinctPrivacyID aggregation.
type DistinctPrivacyIDParams struct {
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
}

// DistinctPrivacyID counts the number of distinct privacy identifiers
// associated to each value in a PrivatePCollection, adding differentially
// private noise to the counts and doing post-aggregation thresholding to
// remove low counts. It is conceptually equivalent to calling Count with
// MaxValue=1, but is specifically optimized for this use case.
//
// Note: Do not use when your results may cause overflows for Int64 values.
// This aggregation is not hardened for such applications yet.
//
// DistinctPrivacyID transforms a PrivatePCollection<V> into a
// PCollection<V,int64>.
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
	epsilon, delta, err := spec.consumeBudget(params.Epsilon, params.Delta)
	if err != nil {
		log.Exitf("couldn't consume budget: %v", err)
	}
	err = checkDistinctPrivacyIDParams(params, noiseKind, epsilon, delta)
	if err != nil {
		log.Exit(err)
	}

	maxPartitionsContributed := getMaxPartitionsContributed(spec, params.MaxPartitionsContributed)
	// First, deduplicate KV pairs by encoding them and calling Distinct.
	coded := beam.ParDo(s, kv.NewEncodeFn(idT, partitionT), pcol.col)
	distinct := filter.Distinct(s, coded)
	decoded := beam.ParDo(s,
		kv.NewDecodeFn(idT, partitionT),
		distinct,
		beam.TypeDefinition{Var: beam.TType, T: idT.Type()},
		beam.TypeDefinition{Var: beam.VType, T: partitionT.Type()})
	// Second, do contribution bounding.
	decoded = boundContributions(s, decoded, maxPartitionsContributed)
	// Third, now that KV pairs are deduplicated and contribution bounding is
	// done, remove the keys and count how many times each value appears.
	values := beam.DropKey(s, decoded)
	dummyCounts := beam.ParDo(s, addOneValueFn, values)
	noisedCounts := beam.CombinePerKey(s,
		newCountFn(epsilon, delta, maxPartitionsContributed, noiseKind),
		dummyCounts)
	// Finally, drop thresholded partitions and return the result
	return beam.ParDo(s, dropThresholdedPartitionsInt64Fn, noisedCounts)
}

func checkDistinctPrivacyIDParams(params DistinctPrivacyIDParams, noiseKind noise.Kind, epsilon, delta float64) error {
	err := checks.CheckEpsilon("pbeam.DistinctPrivacyID", epsilon)
	if err != nil {
		return err
	}
	if noiseKind == noise.LaplaceNoise {
		err = checks.CheckDelta("pbeam.DistinctPrivacyID", delta)
	} else {
		err = checks.CheckDeltaStrict("pbeam.DistinctPrivacyID", delta)
	}
	if err != nil {
		return err
	}
	return checks.CheckMaxPartitionsContributed("pbeam.DistinctPrivacyID", params.MaxPartitionsContributed)
}

func addOneValueFn(v beam.V) (beam.V, int64) {
	return v, 1
}

// countFn is the accumulator used for counting the number of values in a partition.
type countFn struct {
	// Privacy spec parameters (set during initial construction).
	Epsilon                  float64
	DeltaNoise               float64
	DeltaThreshold           float64
	MaxPartitionsContributed int64
	NoiseKind                noise.Kind
	noise                    noise.Noise // Set during Setup phase according to NoiseKind.
}

// newCountFn returns a newCountFn with the given budget and parameters.
func newCountFn(epsilon, delta float64, maxPartitionsContributed int64, noiseKind noise.Kind) *countFn {
	fn := &countFn{
		MaxPartitionsContributed: maxPartitionsContributed,
		NoiseKind:                noiseKind,
	}
	fn.Epsilon = epsilon
	switch noiseKind {
	case noise.GaussianNoise:
		fn.DeltaNoise = delta / 2
		fn.DeltaThreshold = delta / 2
	case noise.LaplaceNoise:
		fn.DeltaNoise = 0
		fn.DeltaThreshold = delta
	default:
		log.Exitf("newCountFn: unknown NoiseKind (%v) is specified. Please specify a valid noise.", noiseKind)
	}
	return fn
}

func (fn *countFn) Setup() {
	fn.noise = noise.ToNoise(fn.NoiseKind)
}

type countAccum struct {
	C *dpagg.Count
}

func (fn *countFn) CreateAccumulator() countAccum {
	return countAccum{C: dpagg.NewCount(&dpagg.CountOptions{
		Epsilon:                  fn.Epsilon,
		Delta:                    fn.DeltaNoise,
		MaxPartitionsContributed: fn.MaxPartitionsContributed,
		Noise:                    fn.noise,
	})}
}

// AddInput adds one to the count of observed values. It ignores the actual
// contents of value.
func (fn *countFn) AddInput(a countAccum, value beam.X) countAccum {
	a.C.Increment()
	return a
}

func (fn *countFn) MergeAccumulators(a, b countAccum) countAccum {
	a.C.Merge(b.C)
	return a
}

func (fn *countFn) ExtractOutput(a countAccum) *int64 {
	return a.C.ThresholdedResult(fn.DeltaThreshold)
}

func (fn *countFn) String() string {
	return fmt.Sprintf("%#v", fn)
}
