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
	// can influence. If a privacy identifier is associated with more values,
	// random values will be dropped. There is an inherent trade-off when
	// choosing this parameter: a larger MaxPartitionsContributed leads to less
	// data loss due to contribution bounding, but since the noise added in
	// aggregations is scaled according to maxPartitionsContributed, it also
	// means that more noise is added to each count.
	//
	// Required.
	MaxPartitionsContributed int64
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

// DistinctPrivacyID counts the number of distinct privacy identifiers
// associated with each value in a PrivatePCollection, adding differentially
// private noise to the counts and doing post-aggregation thresholding to
// remove low counts. It is conceptually equivalent to calling Count with
// MaxValue=1, but is specifically optimized for this use case.
// Client can also specify a PCollection of partitions.
//
// Note: Do not use when your results may cause overflows for int64 values.
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
		log.Fatalf("Couldn't consume budget for DistinctPrivacyID: %v", err)
	}
	err = checkDistinctPrivacyIDParams(params, epsilon, delta, noiseKind)
	if err != nil {
		log.Fatal(err)
	}

	maxPartitionsContributed, err := getMaxPartitionsContributed(spec, params.MaxPartitionsContributed)
	if err != nil {
		log.Fatalf("Couldn't get maxPartitionsContributed for DistinctPrivacyID: %v", err)
	}
	// Drop non-public partitions, if public partitions are specified.
	if (params.PublicPartitions).IsValid() {
		if partitionT.Type() != (params.PublicPartitions).Type().Type() {
			log.Fatalf("Public partitions must be of type %v. Got type %v instead.",
				partitionT.Type(), (params.PublicPartitions).Type().Type())
		}
		partitionEncodedType := beam.EncodedType{partitionT.Type()}
		pcol.col = dropNonPublicPartitionsVFn(s, params.PublicPartitions, pcol, partitionEncodedType)
	}
	// First, deduplicate KV pairs by encoding them and calling Distinct.
	coded := beam.ParDo(s, kv.NewEncodeFn(idT, partitionT), pcol.col)
	distinct := filter.Distinct(s, coded)
	decodeFn := kv.NewDecodeFn(idT, partitionT)
	decoded := beam.ParDo(s,
		decodeFn,
		distinct,
		beam.TypeDefinition{Var: beam.TType, T: idT.Type()},
		beam.TypeDefinition{Var: beam.VType, T: partitionT.Type()})
	// Second, do cross-partition contribution bounding if not in test mode without contribution bounding.
	if spec.testMode != noNoiseWithoutContributionBounding {
		decoded = boundContributions(s, decoded, maxPartitionsContributed)
	}
	// Third, now that KV pairs are deduplicated and contribution bounding is
	// done, remove the keys and count how many times each value appears.
	values := beam.DropKey(s, decoded)
	dummyCounts := beam.ParDo(s, addOneValueFn, values)
	// Add public partitions and return the aggregation output, if public partitions are specified.
	if (params.PublicPartitions).IsValid() {
		return addPublicPartitionsForDistinctID(s, params, epsilon, delta, maxPartitionsContributed, noiseKind, dummyCounts, spec.testMode)
	}
	noisedCounts := beam.CombinePerKey(s,
		newCountFn(epsilon, delta, maxPartitionsContributed, noiseKind, false, spec.testMode),
		dummyCounts)
	// Drop thresholded partitions.
	counts := beam.ParDo(s, dropThresholdedPartitionsInt64Fn, noisedCounts)
	// Clamp negative counts to zero and return.
	return beam.ParDo(s, clampNegativePartitionsInt64Fn, counts)
}

func addPublicPartitionsForDistinctID(s beam.Scope, params DistinctPrivacyIDParams, epsilon, delta float64,
	maxPartitionsContributed int64, noiseKind noise.Kind, countsKV beam.PCollection, testMode testMode) beam.PCollection {
	prepareAddPublicPartitions := beam.ParDo(s, addDummyValuesToPublicPartitionsInt64Fn, params.PublicPartitions)
	// Merge countsKV and prepareAddPublicPartitions.
	allAddPartitions := beam.Flatten(s, countsKV, prepareAddPublicPartitions)
	noisedCounts := beam.CombinePerKey(s,
		newCountFn(epsilon, delta, maxPartitionsContributed, noiseKind, true, testMode),
		allAddPartitions)
	finalPartitions := beam.ParDo(s, dereferenceValueToInt64Fn, noisedCounts)
	// Clamp negative counts to zero and return.
	return beam.ParDo(s, clampNegativePartitionsInt64Fn, finalPartitions)
}

func checkDistinctPrivacyIDParams(params DistinctPrivacyIDParams, epsilon, delta float64, noiseKind noise.Kind) error {
	err := checks.CheckEpsilon("pbeam.DistinctPrivacyID", epsilon)
	if err != nil {
		return err
	}
	if noiseKind == noise.LaplaceNoise {
		err = checks.CheckDelta("pbeam.DistinctPrivacyID", delta)
		if (params.PublicPartitions).IsValid() {
			err = checks.CheckNoDelta("pbeam.DistinctPrivacyID", delta)
		}
	} else {
		checks.CheckDeltaStrict("pbeam.DistinctPrivacyID", delta)
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
	NoiseDelta               float64
	ThresholdDelta           float64
	MaxPartitionsContributed int64
	NoiseKind                noise.Kind
	noise                    noise.Noise // Set during Setup phase according to NoiseKind.
	PublicPartitions         bool
	TestMode                 testMode
}

// newCountFn returns a newCountFn with the given budget and parameters.
func newCountFn(epsilon, delta float64, maxPartitionsContributed int64, noiseKind noise.Kind, publicPartitions bool, testMode testMode) *countFn {
	fn := &countFn{
		MaxPartitionsContributed: maxPartitionsContributed,
		NoiseKind:                noiseKind,
		PublicPartitions:         publicPartitions,
		TestMode:                 testMode,
	}
	fn.Epsilon = epsilon
	if fn.PublicPartitions {
		fn.NoiseDelta = delta
		return fn
	}
	switch noiseKind {
	case noise.GaussianNoise:
		fn.NoiseDelta = delta / 2
	case noise.LaplaceNoise:
		fn.NoiseDelta = 0
	default:
		log.Fatalf("newCountFn: unknown NoiseKind (%v) is specified. Please specify a valid noise.", noiseKind)
	}
	fn.ThresholdDelta = delta - fn.NoiseDelta
	return fn
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

func (fn *countFn) CreateAccumulator() countAccum {
	return countAccum{C: dpagg.NewCount(&dpagg.CountOptions{
		Epsilon:                  fn.Epsilon,
		Delta:                    fn.NoiseDelta,
		MaxPartitionsContributed: fn.MaxPartitionsContributed,
		Noise:                    fn.noise,
	}), PublicPartitions: fn.PublicPartitions}
}

// AddInput increments the count by one for each contribution. Does nothing when the
// the value is 0, which is the case only for dummy public partitions.
func (fn *countFn) AddInput(a countAccum, value int64) countAccum {
	if value != 0 {
		a.C.Increment()
	}
	return a
}

func (fn *countFn) MergeAccumulators(a, b countAccum) countAccum {
	a.C.Merge(b.C)
	return a
}

func (fn *countFn) ExtractOutput(a countAccum) *int64 {
	if fn.TestMode.isEnabled() {
		a.C.Noise = noNoise{}
	}
	if a.PublicPartitions {
		result := a.C.Result()
		return &result
	}
	return a.C.ThresholdedResult(fn.ThresholdDelta)
}

func (fn *countFn) String() string {
	return fmt.Sprintf("%#v", fn)
}
