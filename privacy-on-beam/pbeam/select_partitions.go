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
	"github.com/google/differential-privacy/go/v2/checks"
	"github.com/google/differential-privacy/go/v2/dpagg"
	"github.com/google/differential-privacy/privacy-on-beam/v2/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/filter"
)

func init() {
	beam.RegisterType(reflect.TypeOf((*partitionSelectionFn)(nil)))
	beam.RegisterFunction(dropThresholdedPartitionsBoolFn)
}

// SelectPartitionsParams specifies the parameters associated with a
// SelectPartitions aggregation.
type SelectPartitionsParams struct {
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
}

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

	epsilon, delta, err := pcol.privacySpec.consumeBudget(params.Epsilon, params.Delta)
	if err != nil {
		log.Fatalf("Couldn't consume budget for SelectPartition: %v", err)
	}
	spec := pcol.privacySpec
	maxPartitionsContributed, err := getMaxPartitionsContributed(spec, params.MaxPartitionsContributed)
	if err != nil {
		log.Fatalf("Couldn't get MaxPartitionsContributed for SelectPartitions: %v", err)
	}
	err = checkSelectPartitionsParams(epsilon, delta, maxPartitionsContributed)
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
	if spec.testMode != noNoiseWithoutContributionBounding {
		partitions = boundContributions(s, partitions, maxPartitionsContributed)
	}

	// Finally, we swap the privacy and partition key and perform partition selection.
	partitions = beam.SwapKV(s, partitions) // PCollection<K, ID>
	partitions = beam.CombinePerKey(s, newPartitionSelectionFn(epsilon, delta, maxPartitionsContributed, spec.testMode), partitions)
	result := beam.ParDo(s, dropThresholdedPartitionsBoolFn, partitions)
	return result
}

func checkSelectPartitionsParams(epsilon, delta float64, maxPartitionsContributed int64) error {
	err := checks.CheckEpsilon(epsilon)
	if err != nil {
		return err
	}
	return checks.CheckDeltaStrict(delta)
}

type partitionSelectionAccum struct {
	SP *dpagg.PreAggSelectPartition
}

type partitionSelectionFn struct {
	Epsilon                  float64
	Delta                    float64
	MaxPartitionsContributed int64
	TestMode                 testMode
}

func newPartitionSelectionFn(epsilon, delta float64, maxPartitionsContributed int64, testMode testMode) *partitionSelectionFn {
	return &partitionSelectionFn{Epsilon: epsilon, Delta: delta, MaxPartitionsContributed: maxPartitionsContributed, TestMode: testMode}
}

func (fn *partitionSelectionFn) CreateAccumulator() (partitionSelectionAccum, error) {
	sp, err := dpagg.NewPreAggSelectPartition(&dpagg.PreAggSelectPartitionOptions{
		Epsilon:                  fn.Epsilon,
		Delta:                    fn.Delta,
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

// dropThresholdedPartitionsBoolFn drops thresholded bool partitions, i.e. those
// that have false v, by emitting only non-thresholded partitions. Differently from
// other dropThresholdedPartitionsFn's, since v only indicates whether or not a
// partition should be kept, the value is not emitted with the partition key.
func dropThresholdedPartitionsBoolFn(k beam.W, v bool, emit func(beam.W)) {
	if v {
		emit(k)
	}
}
