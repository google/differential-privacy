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
	"github.com/google/differential-privacy/go/v3/checks"
	"github.com/google/differential-privacy/go/v3/dpagg"
	"github.com/google/differential-privacy/privacy-on-beam/v3/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/register"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/filter"
)

func init() {
	register.Combiner3[partitionSelectionAccum, beam.W, bool](&partitionSelectionFn{})

	register.Function3x0[beam.W, bool, func(beam.W)](dropThresholdedPartitionsBool)
	register.Emitter1[beam.W]()
}

// SelectPartitionsParams specifies the parameters associated with a
// SelectPartitions aggregation.
//
// TODO: Remove this alias.
type SelectPartitionsParams = PartitionSelectionParams

// SelectPartitions performs differentially private partition selection using
// dpagg.PreAggSelectPartitions and returns the list of partitions to keep as
// a PCollection.
//
// In a PrivatePCollection<K,V>, K is the partition key and in a PrivatePCollection<V>,
// V is the partition key. SelectPartitions transforms a PrivatePCollection<K,V> into a
// PCollection<K> and a PrivatePCollection<V> into a PCollection<V>.
func SelectPartitions(s beam.Scope, pcol PrivatePCollection, params SelectPartitionsParams) beam.PCollection {
	s = s.Scope("pbeam.SelectPartitions")
	// Obtain type information from the underlying PCollection<K,V>.
	_, pT := beam.ValidateKVType(pcol.col)
	spec := pcol.privacySpec
	var err error
	params.Epsilon, params.Delta, err = spec.partitionSelectionBudget.consume(params.Epsilon, params.Delta)
	if err != nil {
		log.Fatalf("Couldn't consume budget for SelectPartitions: %v", err)
	}

	err = checkSelectPartitionsParams(params)
	if err != nil {
		log.Fatalf("pbeam.SelectPartitions: %v", err)
	}

	// First, we drop the values if we have (privacyKey, partitionKey, value) tuples.
	// Afterwards, we will have (privacyKey, partitionKey) pairs.
	// If we initially have (privacyKey, partitionKey) pairs already, we do nothing.
	partitions := pcol.col
	if pT.Type() == reflect.TypeOf(kv.Pair{}) {
		if pcol.codec == nil {
			log.Fatalf("SelectPartitions: no codec found for the input PrivatePCollection.")
		}
		partitions = beam.ParDo(s, &dropValuesFn{pcol.codec}, pcol.col, beam.TypeDefinition{Var: beam.WType, T: pcol.codec.VType.T})
	}

	// Second, we keep one contribution per user for each partition.
	idT, partitionT := beam.ValidateKVType(partitions)
	coded := beam.ParDo(s, kv.NewEncodeFn(idT, partitionT), partitions)
	coded = filter.Distinct(s, coded)
	partitions = beam.ParDo(s,
		kv.NewDecodeFn(idT, partitionT),
		coded,
		beam.TypeDefinition{Var: beam.TType, T: idT.Type()},
		beam.TypeDefinition{Var: beam.VType, T: partitionT.Type()})

	// Third, do cross-partition contribution bounding if not in test mode without contribution bounding.
	if spec.testMode != TestModeWithoutContributionBounding {
		partitions = boundContributions(s, partitions, params.MaxPartitionsContributed)
	}

	// Finally, we swap the privacy and partition key and perform partition selection.
	partitions = beam.SwapKV(s, partitions) // PCollection<K, ID>
	partitions = beam.CombinePerKey(s, newPartitionSelectionFn(*spec, params), partitions)
	result := beam.ParDo(s, dropThresholdedPartitionsBool, partitions)
	return result
}

func checkSelectPartitionsParams(params SelectPartitionsParams) error {
	err := checks.CheckEpsilonStrict(params.Epsilon)
	if err != nil {
		return err
	}
	err = checks.CheckDeltaStrict(params.Delta)
	if err != nil {
		return err
	}
	return checkMaxPartitionsContributed(params.MaxPartitionsContributed)
}

type partitionSelectionAccum struct {
	SP *dpagg.PreAggSelectPartition
}

type partitionSelectionFn struct {
	Epsilon                  float64
	Delta                    float64
	PreThreshold             int64
	MaxPartitionsContributed int64
	TestMode                 TestMode
}

func newPartitionSelectionFn(spec PrivacySpec, params SelectPartitionsParams) *partitionSelectionFn {
	return &partitionSelectionFn{Epsilon: params.Epsilon, Delta: params.Delta, PreThreshold: spec.preThreshold, MaxPartitionsContributed: params.MaxPartitionsContributed, TestMode: spec.testMode}
}

func (fn *partitionSelectionFn) CreateAccumulator() (partitionSelectionAccum, error) {
	sp, err := dpagg.NewPreAggSelectPartition(&dpagg.PreAggSelectPartitionOptions{
		Epsilon:                  fn.Epsilon,
		Delta:                    fn.Delta,
		PreThreshold:             fn.PreThreshold,
		MaxPartitionsContributed: fn.MaxPartitionsContributed})
	return partitionSelectionAccum{SP: sp}, err
}

func (fn *partitionSelectionFn) AddInput(a partitionSelectionAccum, _ beam.W) (partitionSelectionAccum, error) {
	err := a.SP.Increment()
	return a, err
}

func (fn *partitionSelectionFn) MergeAccumulators(a, b partitionSelectionAccum) (partitionSelectionAccum, error) {
	err := a.SP.Merge(b.SP)
	return a, err
}

func (fn *partitionSelectionFn) ExtractOutput(a partitionSelectionAccum) (bool, error) {
	if fn.TestMode.isEnabled() {
		return true, nil
	}
	return a.SP.ShouldKeepPartition()
}

func (fn *partitionSelectionFn) String() string {
	return fmt.Sprintf("%#v", fn)
}

// dropThresholdedPartitionsBool drops thresholded bool partitions, i.e. those
// that have false v, by emitting only non-thresholded partitions. Differently from
// other dropThresholdedPartitionsFn's, since v only indicates whether or not a
// partition should be kept, the value is not emitted with the partition key.
func dropThresholdedPartitionsBool(k beam.W, v bool, emit func(beam.W)) {
	if v {
		emit(k)
	}
}
