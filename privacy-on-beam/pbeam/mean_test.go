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
	"testing"

	"github.com/google/differential-privacy/go/dpagg"
	"github.com/google/differential-privacy/go/noise"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/core/typex"
	"github.com/apache/beam/sdks/go/pkg/beam/testing/ptest"
	"github.com/apache/beam/sdks/go/pkg/beam/transforms/stats"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestNewBoundedMeanFloat64Fn(t *testing.T) {
	opts := []cmp.Option{
		cmpopts.EquateApprox(0, 1e-10),
		cmpopts.IgnoreUnexported(boundedMeanFloat64Fn{}),
	}
	for _, tc := range []struct {
		desc      string
		noiseKind noise.Kind
		want      interface{}
	}{
		{"Laplace noise kind", noise.LaplaceNoise,
			&boundedMeanFloat64Fn{
				EpsilonNoise:                 0.5,
				EpsilonPartitionSelection:    0.5,
				DeltaNoise:                   0,
				DeltaPartitionSelection:      1e-5,
				MaxPartitionsContributed:     17,
				MaxContributionsPerPartition: 5,
				Lower:                        0,
				Upper:                        10,
				NoiseKind:                    noise.LaplaceNoise,
			}},
		{"Gaussian noise kind", noise.GaussianNoise,
			&boundedMeanFloat64Fn{
				EpsilonNoise:                 0.5,
				EpsilonPartitionSelection:    0.5,
				DeltaNoise:                   5e-6,
				DeltaPartitionSelection:      5e-6,
				MaxPartitionsContributed:     17,
				MaxContributionsPerPartition: 5,
				Lower:                        0,
				Upper:                        10,
				NoiseKind:                    noise.GaussianNoise,
			}},
	} {
		got := newBoundedMeanFloat64Fn(1, 1e-5, 17, 5, 0, 10, tc.noiseKind)
		if diff := cmp.Diff(tc.want, got, opts...); diff != "" {
			t.Errorf("newBoundedMeanFn: for %q (-want +got):\n%s", tc.desc, diff)
		}
	}
}

func TestBoundedMeanFloat64FnSetup(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		noiseKind noise.Kind
		wantNoise interface{}
	}{
		{"Laplace noise kind", noise.LaplaceNoise, noise.Laplace()},
		{"Gaussian noise kind", noise.GaussianNoise, noise.Gaussian()}} {
		got := newBoundedMeanFloat64Fn(1, 1e-5, 17, 5, 0, 10, tc.noiseKind)
		got.Setup()
		if !cmp.Equal(tc.wantNoise, got.noise) {
			t.Errorf("Setup: for %s got %v, want %v", tc.desc, got.noise, tc.wantNoise)
		}
	}
}

func TestBoundedMeanFloat64FnAddInput(t *testing.T) {
	// δ=10⁻²³, ε=1e100 and l0Sensitivity=1 gives a threshold of =2.
	// Since ε=1e100, the noise is added with probability in the order of exp(-1e100).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 1e100
	delta := 1e-23
	lower := 0.0
	upper := 5.0
	// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
	fn := newBoundedMeanFloat64Fn(2*epsilon, delta, maxPartitionsContributed, maxContributionsPerPartition, lower, upper, noise.LaplaceNoise)
	fn.Setup()

	accum := fn.CreateAccumulator()
	fn.AddInput(accum, []float64{2.0})
	fn.AddInput(accum, []float64{4.0})

	got := fn.ExtractOutput(accum)
	exactSum := 6.0
	exactCount := 2.0
	exactMean := exactSum / exactCount
	want := float64Ptr(exactMean)
	tolerance, err := laplaceToleranceForMean(23, lower, upper, maxContributionsPerPartition, maxPartitionsContributed, epsilon, 1, exactCount, exactMean)
	if err != nil {
		t.Fatalf("laplaceToleranceForMean: got error %v", err)
	}
	if !cmp.Equal(want, got, cmpopts.EquateApprox(0, tolerance)) {
		t.Errorf("AddInput: for 2 values got: %f, want %f", *got, *want)
	}
}

func TestBoundedMeanFloat64FnMergeAccumulators(t *testing.T) {
	// δ=10⁻²³, ε=1e100 and l0Sensitivity=1 gives a threshold of =2.
	// Since ε=1e100, the noise is added with probability in the order of exp(-1e100).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 1e100
	delta := 1e-23
	lower := 0.0
	upper := 5.0
	// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
	fn := newBoundedMeanFloat64Fn(2*epsilon, delta, maxPartitionsContributed, maxContributionsPerPartition, lower, upper, noise.LaplaceNoise)
	fn.Setup()

	accum1 := fn.CreateAccumulator()
	fn.AddInput(accum1, []float64{2.0})
	fn.AddInput(accum1, []float64{3.0})
	fn.AddInput(accum1, []float64{1.0})
	accum2 := fn.CreateAccumulator()
	fn.AddInput(accum2, []float64{4.0})
	fn.MergeAccumulators(accum1, accum2)

	got := fn.ExtractOutput(accum1)
	exactSum := 10.0
	exactCount := 4.0
	exactMean := exactSum / exactCount
	want := float64Ptr(exactMean)
	tolerance, err := laplaceToleranceForMean(23, lower, upper, maxContributionsPerPartition, maxPartitionsContributed, epsilon, 0, exactCount, exactMean)
	if err != nil {
		t.Fatalf("laplaceToleranceForMean: got error %v", err)
	}
	if !cmp.Equal(want, got, cmpopts.EquateApprox(0, tolerance)) {
		t.Errorf("MergeAccumulators: when merging 2 instances of boundedMeanAccumFloat64 got: %f, want %f", *got, *want)
	}
}

func TestBoundedMeanFloat64FnExtractOutputReturnsNilForSmallPartitions(t *testing.T) {
	for _, tc := range []struct {
		desc              string
		inputSize         int
		datapointsPerUser int
	}{
		// It's a special case for partition selection in which the algorithm should always eliminate the partition.
		{"Empty input", 0, 0},
		{"Input with 1 user with 1 contribution", 1, 1},
	} {

		// The choice of ε=1e100, δ=10⁻²³, and l0Sensitivity=1 gives a threshold of =2.
		// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
		fn := newBoundedMeanFloat64Fn(2*1e100, 1e-23, 1, 1, 0, 10, noise.LaplaceNoise)
		fn.Setup()
		accum := fn.CreateAccumulator()
		for i := 0; i < tc.inputSize; i++ {
			values := make([]float64, tc.datapointsPerUser)
			for i := 0; i < tc.datapointsPerUser; i++ {
				values[i] = 1.0
			}
			fn.AddInput(accum, values)
		}

		got := fn.ExtractOutput(accum)

		// Should return nil output for small partitions.
		if got != nil {
			t.Errorf("ExtractOutput: for %s got: %f, want nil", tc.desc, *got)
		}
	}
}

// Checks that MeanPerKey adds noise to its output with float values.
func TestMeanPerKeyAddsNoiseFloat(t *testing.T) {
	for _, tc := range []struct {
		name      string
		noiseKind NoiseKind
		// Differential privacy params used
		epsilon float64
		delta   float64
	}{
		{
			name:      "Gaussian",
			noiseKind: GaussianNoise{},
			epsilon:   2,    // It is split by 2: 1 for the noise and 1 for the partition selection.
			delta:     0.01, // It is split by 2: 0.005 for the noise and 0.005 for the partition selection.
		},
		{
			name:      "Laplace",
			noiseKind: LaplaceNoise{},
			epsilon:   0.2, // It is split by 2: 0.1 for the noise and 0.1 for the partition selection.
			delta:     0.01,
		},
	} {
		lower := 0.0
		upper := 3.0

		// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
		// we need to have each partition pass with 1-10⁻²³ probability (k=23).
		epsilonNoise, deltaNoise := tc.epsilon/2, 0.0
		k := 23.0
		l0Sensitivity, lInfSensitivity := 1.0, 2.0
		epsilonPartition, deltaPartition := tc.epsilon/2, tc.delta
		if tc.noiseKind == gaussianNoise {
			deltaNoise = tc.delta / 2
			deltaPartition = tc.delta / 2
		}

		// Compute the number of IDs needed to keep the partition.
		sp := dpagg.NewPreAggSelectPartition(&dpagg.PreAggSelectPartitionOptions{Epsilon: epsilonPartition, Delta: deltaPartition, MaxPartitionsContributed: 1})
		numIDs := sp.GetHardThreshold()

		var tolerance float64
		var err error
		if tc.noiseKind == gaussianNoise {
			tolerance, err = complementaryGaussianToleranceForMean(k, lower, upper, int64(lInfSensitivity), int64(l0Sensitivity), epsilonNoise, deltaNoise, -0.5*float64(numIDs), float64(numIDs), 1.0)
			if err != nil {
				t.Fatalf("complementaryGaussianToleranceForMean: got error %v", err)
			}
		} else {
			tolerance, err = complementaryLaplaceToleranceForMean(k, lower, upper, int64(lInfSensitivity), int64(l0Sensitivity), epsilonNoise, -0.5*float64(numIDs), float64(numIDs), 1.0)
			if err != nil {
				t.Fatalf("complementaryLaplaceToleranceForMean: got error %v", err)
			}
		}

		// triples contains {1,0,1}, {2,0,1}, …, {numIDs,0,1}.
		triples := makeDummyTripleWithFloatValue(numIDs, 0)
		p, s, col := ptest.CreateList(triples)
		col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)

		pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
		pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
		got := MeanPerKey(s, pcol, MeanParams{
			MaxPartitionsContributed:     1,
			MaxContributionsPerPartition: 1,
			MinValue:                     0.0,
			MaxValue:                     2.0,
			NoiseKind:                    tc.noiseKind,
		})
		got = beam.ParDo(s, kvToFloat64Metric, got)

		checkFloat64MetricsAreNoisy(s, got, 1.0, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("MeanPerKey didn't add any noise with float inputs and %s Noise: %v", tc.name, err)
		}
	}
}

// Checks that MeanPerKey returns a correct answer for int input values.
// They should be correctly converted to float64 and then correct result
// with float statistic should be computed.
func TestMeanPerKeyNoNoiseIntValues(t *testing.T) {
	triples := concatenateTriplesWithIntValue(
		makeTripleWithIntValue(7, 0, 2),
		makeTripleWithIntValueStartingFromKey(7, 100, 1, 1),
		makeTripleWithIntValueStartingFromKey(107, 150, 1, 2))

	exactCount := 250.0
	exactMean := (100.0 + 2.0*150.0) / exactCount
	result := []testFloat64Metric{
		{1, exactMean},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l0Sensitivity=1 gives a threshold of =11.
	// We have 2 partitions. So, to get an overall flakiness of 10⁻²³,
	// we can have each partition fail with 10⁻²⁵ probability (k=25).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 50.0
	delta := 1e-200
	lower := 0.0
	upper := 2.0

	// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
	pcol := MakePrivate(s, col, NewPrivacySpec(2*epsilon, delta))
	pcol = ParDo(s, tripleWithIntValueToKV, pcol)
	got := MeanPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     lower,
		MaxValue:                     upper,
		NoiseKind:                    LaplaceNoise{},
	})
	want = beam.ParDo(s, float64MetricToKV, want)

	tolerance, err := laplaceToleranceForMean(25, lower, upper, maxContributionsPerPartition, maxPartitionsContributed, epsilon, 150.0, exactCount, exactMean)
	if err != nil {
		t.Fatalf("laplaceToleranceForMean: got error %v", err)
	}
	if err := approxEqualsKVFloat64(s, got, want, tolerance); err != nil {
		t.Fatalf("TestMeanPerKeyNoNoise: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanPerKeyNoNoise: MeanPerKey(%v) = %v, want %v, error %v", col, got, want, err)
	}
}

// Checks that MeanPerKey does partition selection correctly by counting user ids correctly,
// which means if the user has  > 1 contributions to a partition the algorithm will not consider them as new user ids.
func TestMeanPerKeyCountsUserIDsWithMultipleContributionsCorrectly(t *testing.T) {
	triples := concatenateTriplesWithFloatValue(
		makeTripleWithFloatValue(7, 0, 2.0),
		makeTripleWithFloatValueStartingFromKey(7, 11, 1, 1.3),
		// We have a total of 42 contributions to partition 2, but users with id 18 and 19 each contribute 21 times each.
		// So the actual count of user ids in the partition 2 is equal to 2, not 42.
		// And the threshold is equal to 11, so the partition 2 should be eliminated,
		// because the probability of keeping the partition with 2 elements is negligible, ≈5.184e-179.
		makeTripleWithFloatValueStartingFromKey(18, 2, 2, 0))

	for i := 0; i < 20; i++ {
		triples = append(triples, makeTripleWithFloatValueStartingFromKey(18, 2, 2, 1)...)
	}

	exactCount := 42.0
	exactMean := 2 * 20 / exactCount
	result := []testFloat64Metric{
		{1, exactMean},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l0Sensitivity=1 gives a threshold of =11.
	// We have 2 partitions. So, to get an overall flakiness of 10⁻²³,
	// we can have each partition fail with 10⁻²⁵ probability (k=25).
	maxContributionsPerPartition := int64(20)
	maxPartitionsContributed := int64(1)
	epsilon := 50.0
	delta := 1e-200
	lower := 1.0
	upper := 3.0

	// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
	pcol := MakePrivate(s, col, NewPrivacySpec(2*epsilon, delta))
	pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
	got := MeanPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     lower,
		MaxValue:                     upper,
		NoiseKind:                    LaplaceNoise{},
	})

	want = beam.ParDo(s, float64MetricToKV, want)
	tolerance, err := laplaceToleranceForMean(25, lower, upper, maxContributionsPerPartition, maxPartitionsContributed, epsilon, -7.700000524520874, exactCount, exactMean)
	if err != nil {
		t.Fatalf("laplaceToleranceForMean: got error %v", err)
	}
	if err := approxEqualsKVFloat64(s, got, want, tolerance); err != nil {
		t.Fatalf("approxEqualsKVFloat64: got error %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("MeanPerKey: for %v got %v, want %v, error %v", col, got, want, err)
	}
}

// Checks that MeanPerKey returns a correct answer for float input values.
func TestMeanPerKeyNoNoiseFloatValues(t *testing.T) {
	triples := concatenateTriplesWithFloatValue(
		makeTripleWithFloatValue(7, 0, 2.0),
		makeTripleWithFloatValueStartingFromKey(7, 100, 1, 1.3),
		makeTripleWithFloatValueStartingFromKey(107, 150, 1, 2.5))

	exactCount := 250.0
	exactMean := (1.3*100 + 2.5*150) / exactCount
	result := []testFloat64Metric{
		{1, exactMean},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l0Sensitivity=1 gives a threshold of =11.
	// We have 2 partitions. So, to get an overall flakiness of 10⁻²³,
	// we can have each partition fail with 10⁻²⁵ probability (k=25).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 50.0
	delta := 1e-200
	lower := 1.0
	upper := 3.0

	// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
	pcol := MakePrivate(s, col, NewPrivacySpec(2*epsilon, delta))
	pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
	got := MeanPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     lower,
		MaxValue:                     upper,
		NoiseKind:                    LaplaceNoise{},
	})

	want = beam.ParDo(s, float64MetricToKV, want)
	tolerance, err := laplaceToleranceForMean(25, lower, upper, maxContributionsPerPartition, maxPartitionsContributed, epsilon, 5, exactCount, exactMean)
	if err != nil {
		t.Fatalf("laplaceToleranceForMean: got error %v", err)
	}
	if err := approxEqualsKVFloat64(s, got, want, tolerance); err != nil {
		t.Fatalf("TestMeanPerKeyNoNoise: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanPerKeyNoNoise: MeanPerKey(%v) = %v, want %v, error %v", col, got, want, err)
	}
}

var meanPartitionSelectionNonDeterministicTestCases = []struct {
	name                string
	noiseKind           NoiseKind
	epsilon             float64
	delta               float64
	numPartitions       int
	entriesPerPartition int
}{
	{
		name:      "Gaussian",
		noiseKind: GaussianNoise{},
		// After splitting the (ε, δ) budget between the noise and partition
		// selection portions of the privacy algorithm, this results in a ε=1,
		// δ=0.3 partition selection budget.
		epsilon: 2,
		delta:   0.6,
		// entriesPerPartition=1 yields a 30% chance of emitting any particular partition
		// (since δ_emit=0.3).
		entriesPerPartition: 1,
		// 143 distinct partitions implies that some (but not all) partitions are
		// emitted with high probability (at least 1 - 1e-20).
		numPartitions: 143,
	},
	{
		name:      "Laplace",
		noiseKind: LaplaceNoise{},
		// After splitting the (ε, δ) budget between the noise and partition
		// selection portions of the privacy algorithm, this results in the
		// partition selection portion of the budget being ε_selectPartition=1,
		// δ_selectPartition=0.3.
		epsilon: 2,
		delta:   0.3,
		// entriesPerPartition=1 yields a 30% chance of emitting any particular partition
		// (since δ_emit=0.3).
		entriesPerPartition: 1,
		numPartitions:       143,
	},
}

// Checks that MeanPerKey is performing a random partition selection.
func TestMeanPartitionSelectionNonDeterministic(t *testing.T) {
	for _, tc := range meanPartitionSelectionNonDeterministicTestCases {
		t.Run(tc.name, func(t *testing.T) {
			// Sanity check that the entriesPerPartition is sensical.
			if tc.entriesPerPartition <= 0 {
				t.Errorf("Invalid test case: entriesPerPartition must be positive. Got: %d", tc.entriesPerPartition)
			}

			// Build up {ID, Partition, Value} pairs where each user contributes 1
			// value to 1 partition:
			//    {0, 0, 1}, {1, 0, 1}, …, {entriesPerPartition-1, 0, 1}
			//    {entriesPerPartition, 1, 1}, {entriesPerPartition+1, 1, 1}, …, {entriesPerPartition+entriesPerPartition-1, 1, 1}
			//    …
			//    {entriesPerPartition*(numPartitions-1), numPartitions-1, 1}, …, {entriesPerPartition*numPartitions-1, numPartitions-1, 1}
			var triples []tripleWithFloatValue
			var kOffset = 0
			for i := 0; i < tc.numPartitions; i++ {
				triples = append(triples, makeDummyTripleWithFloatValueStartingFromKey(kOffset, tc.entriesPerPartition, i)...)
				kOffset += tc.entriesPerPartition
			}
			p, s, col := ptest.CreateList(triples)
			col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)

			// Run MeanPerKey on triples
			pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
			pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
			got := MeanPerKey(s, pcol, MeanParams{
				MinValue:                     0.0,
				MaxValue:                     1.0,
				MaxContributionsPerPartition: int64(tc.entriesPerPartition),
				MaxPartitionsContributed:     1,
				NoiseKind:                    tc.noiseKind,
			})
			got = beam.ParDo(s, kvToFloat64Metric, got)

			// Validate that partitions are selected randomly (i.e., some emitted and some dropped).
			checkSomePartitionsAreDropped(s, got, tc.numPartitions)
			if err := ptest.Run(p); err != nil {
				t.Errorf("%v", err)
			}
		})
	}
}

// Checks that MeanPerKey works correctly for negative bounds and negative float values.
func TestMeanKeyNegativeBounds(t *testing.T) {
	triples := concatenateTriplesWithFloatValue(
		makeTripleWithFloatValue(100, 1, -5.0),
		makeTripleWithFloatValueStartingFromKey(100, 150, 1, -1.0))

	exactCount := 250.0
	exactMean := (-5.0*100 - 1.0*150) / exactCount
	result := []testFloat64Metric{
		{1, exactMean},
	}

	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l0Sensitivity=1 gives a threshold of =11.
	// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
	// we can have each partition fail with 10⁻²³ probability (k=23).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 50.0
	delta := 1e-200
	lower := -6.0
	upper := -2.0

	// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
	pcol := MakePrivate(s, col, NewPrivacySpec(2*epsilon, delta))
	pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
	got := MeanPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     lower,
		MaxValue:                     upper,
		NoiseKind:                    LaplaceNoise{},
	})

	want = beam.ParDo(s, float64MetricToKV, want)
	tolerance, err := laplaceToleranceForMean(23, lower, upper, maxContributionsPerPartition, maxPartitionsContributed, epsilon, 200.0, exactCount, exactMean)
	if err != nil {
		t.Fatalf("laplaceToleranceForMean: got error %v", err)
	}
	if err := approxEqualsKVFloat64(s, got, want, tolerance); err != nil {
		t.Fatalf("TestMeanPerKeyNegativeBounds: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanPerKeyNegativeBounds: MeanPerKey(%v) = %v, want %v, error %v", col, got, want, err)
	}
}

// Checks that MeanPerKey does cross-partition contribution bounding correctly.
func TestMeanPerKeyCrossPartitionContributionBounding(t *testing.T) {
	var triples []tripleWithFloatValue

	triples = append(triples, makeTripleWithFloatValue(1, 0, 150)...)
	triples = append(triples, makeTripleWithFloatValue(1, 1, 150)...)

	triples = append(triples, makeTripleWithFloatValueStartingFromKey(1, 50, 0, 0)...)
	triples = append(triples, makeTripleWithFloatValueStartingFromKey(51, 50, 1, 0)...)

	// MaxPartitionContributed = 1, but id = 0 contributes to 2 partitions (0 and 1).
	// There will be cross-partition contribution bounding stage.
	// In this stage the algorithm will randomly chose either contribution for partition 0 or contribution to partition 1.
	// The sum of 2 means should be equal to 150/51 + 0/50 = 150/51 ≈ 2.94 in both cases (unlike 150/51 + 150/51 ≈ 5.88, if no cross-partition contribution bounding is done).
	// The difference between these numbers ≈ 2.94 and the tolerance (see below) is ≈ 0.04, so the test should catch if there was no cross-partition contribution bounding.
	exactCount := 51.0
	exactMean := 150.0 / exactCount
	result := []testFloat64Metric{
		{0, exactMean},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)

	// ε=60, δ=0.01 and l0Sensitivity=1 gives a threshold of =2.
	// We have 2 partitions. So, to get an overall flakiness of 10⁻²³,
	// we can have each partition fail with 10⁻²⁵ probability (k=25).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 60.0
	delta := 0.01
	lower := 0.0
	upper := 150.0

	// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
	pcol := MakePrivate(s, col, NewPrivacySpec(2*epsilon, delta))
	pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
	got := MeanPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     lower,
		MaxValue:                     upper,
		NoiseKind:                    LaplaceNoise{},
	})

	means := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, means)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.

	want = beam.ParDo(s, float64MetricToKV, want)

	// Tolerance for the partition with an extra contribution which is equal to 150.
	tolerance1, err := laplaceToleranceForMean(25, lower, upper, maxContributionsPerPartition, maxPartitionsContributed, epsilon, -3675.0, 51.0, exactMean) // ≈0.00367
	if err != nil {
		t.Fatalf("laplaceToleranceForMean: got error %v", err)
	}
	// Tolerance for the partition without an extra contribution.
	tolerance2, err := laplaceToleranceForMean(25, lower, upper, maxContributionsPerPartition, maxPartitionsContributed, epsilon, -3700.0, 50.0, 0.0) // ≈1.074
	if err != nil {
		t.Fatalf("laplaceToleranceForMean: got error %v", err)
	}
	if err := approxEqualsKVFloat64(s, got, want, tolerance1+tolerance2); err != nil {
		t.Fatalf("TestMeanPerKeyPerPartitionContributionBounding: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanPerKeyPerPartitionContributionBounding: MeanPerKey(%v) = %v, want %v, error %v", col, got, want, err)
	}
}

// Checks that MeanPerKey does per-partition contribution bounding correctly.
func TestMeanPerKeyPerPartitionContributionBounding(t *testing.T) {
	var triples []tripleWithFloatValue

	triples = append(triples, makeTripleWithFloatValue(1, 0, 50)...)
	triples = append(triples, makeTripleWithFloatValue(1, 0, 50)...)
	triples = append(triples, makeTripleWithFloatValue(1, 0, 50)...)
	triples = append(triples, makeTripleWithFloatValueStartingFromKey(1, 50, 0, 0)...)

	// MaxContributionsPerPartition = 1, but id = 0 contributes 3 times to partition 0.
	// There will be per-partition contribution bounding stage.
	// In this stage the algorithm will randomly chose one of these 3 contributions.
	// The mean should be equal to 50/51 = 0.98 (not 150/53 ≈ 2.83, if no per-partition contribution bounding is done).
	// The difference between these numbers ≈ 1,85 and the tolerance (see below) is ≈0.92, so the test should catch if there was no per-partition contribution bounding.
	exactCount := 51.0
	exactMean := 50.0 / exactCount
	result := []testFloat64Metric{
		{0, exactMean},
	}

	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)

	// ε=60, δ=0.01 and l0Sensitivity=1 gives a threshold of =2.
	// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
	// we can have that partition fail with 10⁻²³ probability (k=23).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 1000.0
	delta := 0.01
	lower := 0.0
	upper := 100.0

	// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
	pcol := MakePrivate(s, col, NewPrivacySpec(2*epsilon, delta))
	pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
	got := MeanPerKey(s, pcol, MeanParams{
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     lower,
		MaxValue:                     upper,
		MaxPartitionsContributed:     1,
		NoiseKind:                    LaplaceNoise{},
	})

	means := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, means)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.

	want = beam.ParDo(s, float64MetricToKV, want)
	tolerance, err := laplaceToleranceForMean(23, lower, upper, maxContributionsPerPartition, maxPartitionsContributed, epsilon, -2500.0, exactCount, exactMean) // ≈0.92
	if err != nil {
		t.Fatalf("laplaceToleranceForMean: got error %v", err)
	}
	if err := approxEqualsKVFloat64(s, got, want, tolerance); err != nil {
		t.Fatalf("TestMeanPerKeyPerPartitionContributionBounding: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanPerKeyPerPartitionContributionBounding: MeanPerKey(%v) = %v, want %v, error %v", col, got, want, err)
	}
}

// Expect non-negative results if MinValue >= 0 for float64 values.
func TestMeanPerKeyReturnsNonNegativeFloat64(t *testing.T) {
	var triples []tripleWithFloatValue
	for key := 0; key < 100; key++ {
		triples = append(triples, tripleWithFloatValue{key, key, 0.01})
	}
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)
	// Using a low ε, a high δ, and a high maxValue here to add a
	// lot of noise while keeping partitions.
	// ε=0.001. δ=0.999 and l0Sensitivity=1 gives a threshold of =2.
	maxContributionsPerPartition := int64(1)
	epsilon := 0.001
	delta := 0.999
	lower := 0.0
	upper := 1e8

	// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
	pcol := MakePrivate(s, col, NewPrivacySpec(2*epsilon, delta))
	pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
	means := MeanPerKey(s, pcol, MeanParams{
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     lower,
		MaxValue:                     upper,
		MaxPartitionsContributed:     1,
		NoiseKind:                    LaplaceNoise{},
	})
	values := beam.DropKey(s, means)
	beam.ParDo0(s, checkNoNegativeValuesFloat64Fn, values)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanPerKeyReturnsNonNegativeFloat64 returned errors: %v", err)
	}
}

// Expect at least one negative value after post-aggregation clamping when
// MinValue < 0 for float64 values.
func TestMeanPerKeyNoClampingForNegativeMinValueFloat64(t *testing.T) {
	var triples []tripleWithFloatValue
	// The probability that any given partition has a negative noisy mean is 1/2 * 0.999.
	// The probability of none of the partitions having a noisy negative mean is 1 - (1/2 * 0.999)^1000, which is negligible.
	for key := 0; key < 1000; key++ {
		triples = append(triples, tripleWithFloatValue{key, key, 0})
	}
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)

	// ε=0.1. δ=0.999 and l0Sensitivity=1 gives a threshold of =2.
	maxContributionsPerPartition := int64(1)
	epsilon := 0.1
	delta := 0.999
	lower := -100.0
	upper := 100.0

	// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
	pcol := MakePrivate(s, col, NewPrivacySpec(2*epsilon, delta))
	pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
	means := MeanPerKey(s, pcol, MeanParams{
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     lower,
		MaxValue:                     upper,
		MaxPartitionsContributed:     1,
		NoiseKind:                    LaplaceNoise{},
	})
	values := beam.DropKey(s, means)
	mValue := stats.Min(s, values)
	beam.ParDo0(s, checkAllValuesNegativeFloat64Fn, mValue)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanPerKeyNoClampingForNegativeMinValueFloat64 returned errors: %v", err)
	}
}

func TestFindConvertToFloat64Fn(t *testing.T) {
	for _, tc := range []struct {
		desc          string
		fullType      typex.FullType
		wantConvertFn interface{}
		wantErr       bool
	}{
		{"int", typex.New(reflect.TypeOf(int(0))), convertIntToFloat64Fn, false},
		{"int8", typex.New(reflect.TypeOf(int8(0))), convertInt8ToFloat64Fn, false},
		{"int16", typex.New(reflect.TypeOf(int16(0))), convertInt16ToFloat64Fn, false},
		{"int32", typex.New(reflect.TypeOf(int32(0))), convertInt32ToFloat64Fn, false},
		{"int64", typex.New(reflect.TypeOf(int64(0))), convertInt64ToFloat64Fn, false},
		{"uint", typex.New(reflect.TypeOf(uint(0))), convertUintToFloat64Fn, false},
		{"uint8", typex.New(reflect.TypeOf(uint8(0))), convertUint8ToFloat64Fn, false},
		{"uint16", typex.New(reflect.TypeOf(uint16(0))), convertUint16ToFloat64Fn, false},
		{"uint32", typex.New(reflect.TypeOf(uint32(0))), convertUint32ToFloat64Fn, false},
		{"uint64", typex.New(reflect.TypeOf(uint64(0))), convertUint64ToFloat64Fn, false},
		{"float32", typex.New(reflect.TypeOf(float32(0))), convertFloat32ToFloat64Fn, false},
		{"float64", typex.New(reflect.TypeOf(float64(0))), convertFloat64ToFloat64Fn, false},
		{"string", typex.New(reflect.TypeOf("")), nil, true},
	} {
		convertFn, err := findConvertToFloat64Fn(tc.fullType)
		if (err != nil) != tc.wantErr {
			t.Errorf("findConvertToFloat64Fn: when %s for err got got %v, want %t", tc.desc, err, tc.wantErr)
		}
		if !reflect.DeepEqual(reflect.TypeOf(convertFn), reflect.TypeOf(tc.wantConvertFn)) {
			t.Errorf("findConvertToFloat64Fn: when %s got %v, want %v", tc.desc, convertFn, tc.wantConvertFn)
		}
	}
}
