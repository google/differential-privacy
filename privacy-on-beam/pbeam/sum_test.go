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
	"testing"

	"github.com/google/differential-privacy/go/dpagg"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/core/typex"
	"github.com/apache/beam/sdks/go/pkg/beam/testing/ptest"
	"github.com/apache/beam/sdks/go/pkg/beam/transforms/stats"
)

func init() {
	beam.RegisterFunction(checkAllValuesNegativeInt64Fn)
}

// Checks that SumPerKey returns a correct answer with int values. The logic
// mirrors TestDistinctPrivacyIDNoNoise, without duplicates.
func TestSumPerKeyNoNoiseInt(t *testing.T) {
	triples := concatenateTriplesWithIntValue(
		makeDummyTripleWithIntValue(7, 0),
		makeDummyTripleWithIntValue(58, 1),
		makeDummyTripleWithIntValue(99, 2))
	result := []testInt64Metric{
		// The sum for value 0 is 7: should be thresholded.
		{1, 58},
		{2, 99},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l1Sensitivity=3 gives a threshold of ≈58.
	// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁵ probability (k=25).
	// To see the logic and the math behind flakiness and tolerance calculation,
	// See https://github.com/google/differential-privacy/blob/master/privacy-on-beam/docs/Tolerance_Calculation.pdf.
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-200, 25.0, 3.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	pcol = ParDo(s, tripleWithIntValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 3, MinValue: 0.0, MaxValue: 1, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, int64MetricToKV, want)
	if err := approxEqualsKVInt64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestSumPerKeyNoNoiseInt: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyNoNoiseInt: SumPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

// Checks that SumPerKey works correctly for negative bounds and negative values with int values.
func TestSumPerKeyNegativeBoundsInt(t *testing.T) {
	triples := concatenateTriplesWithIntValue(
		makeTripleWithIntValue(58, 1, -1), // should be clamped down to -2
		makeTripleWithIntValue(99, 2, -4)) // should be clamped up to -3
	result := []testInt64Metric{
		{1, -116},
		{2, -297},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l1Sensitivity=3 gives a threshold of ≈58.
	// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁵ probability (k=25).
	// To see the logic and the math behind flakiness and tolerance calculation,
	// See https://github.com/google/differential-privacy/blob/master/privacy-on-beam/docs/Tolerance_Calculation.pdf.
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-200, 25.0, 3.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	pcol = ParDo(s, tripleWithIntValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 3, MinValue: -3, MaxValue: -2, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, int64MetricToKV, want)
	if err := approxEqualsKVInt64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestSumPerKeyNegativeBoundsInt: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyNegativeBoundsInt: SumPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

// Checks that SumPerKey returns a correct answer with float values. The logic
// mirrors TestDistinctPrivacyIDNoNoise, without duplicates.
func TestSumPerKeyNoNoiseFloat(t *testing.T) {
	triples := concatenateTriplesWithFloatValue(
		makeDummyTripleWithFloatValue(7, 0),
		makeDummyTripleWithFloatValue(58, 1),
		makeDummyTripleWithFloatValue(99, 2))
	result := []testFloat64Metric{
		// Only 7 users are associated to value 0: should be thresholded.
		{1, 58},
		{2, 99},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l1Sensitivity=3 gives a threshold of ≈58.
	// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁵ probability (k=25).
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-200, 25.0, 3.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 3, MinValue: 0.0, MaxValue: 1.0, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, float64MetricToKV, want)
	if err := approxEqualsKVFloat64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestSumPerKeyNoNoiseFloat: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyNoNoiseFloat: SumPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

// Checks that SumPerKey works correctly for negative bounds and negative values with float values.
func TestSumPerKeyNegativeBoundsFloat(t *testing.T) {
	triples := concatenateTriplesWithFloatValue(
		makeTripleWithFloatValue(58, 1, -1.0), // should be clamped down to -2.0
		makeTripleWithFloatValue(99, 2, -4.0)) // should be clamped up to -3.0
	result := []testFloat64Metric{
		{1, -116.0},
		{2, -297.0},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l1Sensitivity=3 gives a threshold of ≈58.
	// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁵ probability (k=25).
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-200, 25.0, 3.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 3, MinValue: -3.0, MaxValue: -2.0, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, float64MetricToKV, want)
	if err := approxEqualsKVFloat64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestSumPerKeyNegativeBoundsFloat: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyNegativeBoundsFloat: SumPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

// Checks that SumPerKey adds noise to its output with int values. The logic
// mirrors TestDistinctPrivacyIDAddsNoise.
func TestSumPerKeyAddsNoiseInt(t *testing.T) {
	for _, tc := range []struct {
		name      string
		noiseKind NoiseKind
		// Differential privacy params used.
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
		// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
		// we need to have each partition pass with 1-10⁻²³ probability (k=23).
		epsilonNoise, deltaNoise := tc.epsilon/2, 0.0
		k := 23.0
		l0Sensitivity, lInfSensitivity := 1.0, 1.0
		epsilonPartition, deltaPartition := tc.epsilon/2, tc.delta
		l1Sensitivity := l0Sensitivity * lInfSensitivity
		tolerance := complementaryLaplaceTolerance(k, l1Sensitivity, epsilonNoise)
		if tc.noiseKind == gaussianNoise {
			deltaNoise = tc.delta / 2
			deltaPartition = tc.delta / 2
			tolerance = complementaryGaussianTolerance(k, l0Sensitivity, lInfSensitivity, epsilonNoise, deltaNoise)
		}

		// Compute the number of IDs needed to keep the partition.
		sp := dpagg.NewPreAggSelectPartition(&dpagg.PreAggSelectPartitionOptions{Epsilon: epsilonPartition, Delta: deltaPartition, MaxPartitionsContributed: 1})
		numIDs := sp.GetHardThreshold()

		// triples contains {1,0,1}, {2,0,1}, …, {numIDs,0,1}.
		triples := makeDummyTripleWithIntValue(numIDs, 0)
		p, s, col := ptest.CreateList(triples)
		col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

		pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
		pcol = ParDo(s, tripleWithIntValueToKV, pcol)
		got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 1, MinValue: 0, MaxValue: 1, NoiseKind: tc.noiseKind})
		got = beam.ParDo(s, kvToInt64Metric, got)

		checkInt64MetricsAreNoisy(s, got, numIDs, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("SumPerKey didn't add any noise with int inputs and %s Noise: %v", tc.name, err)
		}
	}
}

// Checks that SumPerKey adds noise to its output with float values. The logic
// mirrors TestDistinctPrivacyIDAddsNoise.
func TestSumPerKeyAddsNoiseFloat(t *testing.T) {
	for _, tc := range []struct {
		name      string
		noiseKind NoiseKind
		// Differential privacy params used.
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
		// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
		// we need to have each partition pass with 1-10⁻²³ probability (k=23).
		epsilonNoise, deltaNoise := tc.epsilon/2, 0.0
		k := 23.0
		l0Sensitivity, lInfSensitivity := 1.0, 1.0
		epsilonPartition, deltaPartition := tc.epsilon/2, tc.delta
		l1Sensitivity := l0Sensitivity * lInfSensitivity
		tolerance := complementaryLaplaceTolerance(k, l1Sensitivity, epsilonNoise)
		if tc.noiseKind == gaussianNoise {
			deltaNoise = tc.delta / 2
			deltaPartition = tc.delta / 2
			tolerance = complementaryGaussianTolerance(k, l0Sensitivity, lInfSensitivity, epsilonNoise, deltaNoise)
		}

		// Compute the number of IDs needed to keep the partition.
		sp := dpagg.NewPreAggSelectPartition(&dpagg.PreAggSelectPartitionOptions{Epsilon: epsilonPartition, Delta: deltaPartition, MaxPartitionsContributed: 1})
		numIDs := sp.GetHardThreshold()

		// triples contains {1,0,1}, {2,0,1}, …, {numIDs,0,1}.
		triples := makeDummyTripleWithFloatValue(numIDs, 0)
		p, s, col := ptest.CreateList(triples)
		col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)

		pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
		pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
		got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 1, MinValue: 0.0, MaxValue: 1.0, NoiseKind: tc.noiseKind})
		got = beam.ParDo(s, kvToFloat64Metric, got)

		checkFloat64MetricsAreNoisy(s, got, float64(numIDs), tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("SumPerKey didn't add any noise with float inputs and %s Noise: %v", tc.name, err)
		}
	}
}

// Checks that SumPerKey bounds per-user contributions correctly with int values.
// The logic mirrors TestCountCrossPartitionContributionBounding.
func TestSumPerKeyCrossPartitionContributionBoundingInt(t *testing.T) {
	// triples contains {1,0,1}, {2,0,1}, …, {50,0,1}, {1,1,1}, …, {50,1,1}, {1,2,1}, …, {50,9,1}.
	var triples []tripleWithIntValue
	for i := 0; i < 10; i++ {
		triples = append(triples, makeDummyTripleWithIntValue(50, i)...)
	}
	result := []testInt64Metric{
		{0, 150},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

	// ε=50, δ=0.01 and l1Sensitivity=3 gives a threshold of 1.
	// We have 10 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁵ probability (k=25).
	epsilon, delta, k, l1Sensitivity := 50.0, 0.01, 25.0, 3.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	pcol = ParDo(s, tripleWithIntValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 3, MinValue: 0, MaxValue: 1, NoiseKind: LaplaceNoise{}})
	// With a max contribution of 3, 70% of the data should have be
	// dropped. The sum of all elements must then be 150.
	counts := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, counts)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
	want = beam.ParDo(s, int64MetricToKV, want)
	if err := approxEqualsKVInt64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestSumPerKeyCrossPartitionContributionBoundingInt: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyCrossPartitionContributionBoundingInt: SumPerKey(%v) = %v, expected elements to sum to 150: %v", col, got, err)
	}
}

// Checks that SumPerKey bounds per-user contributions correctly with float values.
// The logic mirrors TestCountCrossPartitionContributionBounding.
func TestSumPerKeyCrossPartitionContributionBoundingFloat(t *testing.T) {
	// triples contains {1,0,1.0}, {2,0,1.0}, …, {50,0,1.0}, {1,1,1.0}, …, {50,1,1.0}, {1,2,1.0}, …, {50,9,1.0}.
	var triples []tripleWithFloatValue
	for i := 0; i < 10; i++ {
		triples = append(triples, makeDummyTripleWithFloatValue(50, i)...)
	}
	result := []testFloat64Metric{
		{0, 150.0},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)

	// ε=50, δ=0.01 and l1Sensitivity=3 gives a threshold of 1.
	// We have 10 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁵ probability (k=25).
	epsilon, delta, k, l1Sensitivity := 50.0, 0.01, 25.0, 3.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 3, MinValue: 0.0, MaxValue: 1.0, NoiseKind: LaplaceNoise{}})
	// With a max contribution of 3, 70% of the data should have be
	// dropped. The sum of all elements must then be 150.
	counts := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, counts)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
	want = beam.ParDo(s, float64MetricToKV, want)
	if err := approxEqualsKVFloat64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestSumPerKeyCrossPartitionContributionBoundingFloat: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyCrossPartitionContributionBoundingFloat: SumPerKey(%v) = %v, expected elements to sum to 150.0: %v", col, got, err)
	}
}

// Checks that SumPerKey does per-partition contribution bounding correctly for ints.
func TestSumPerKeyPerPartitionContributionBoundingInt(t *testing.T) {
	var triples []tripleWithIntValue
	for id := 1; id <= 50; id++ {
		triples = append(triples, tripleWithIntValue{id, 0, 1}) // partition 0 is associated to 50 times 1
		triples = append(triples, tripleWithIntValue{id, 1, 4}) // partition 1 is associated to 50 times 4
		// Additional values that should not influence the clamping
		triples = append(triples, tripleWithIntValue{id, 0, -17}) // should clamp to lower bound
		triples = append(triples, tripleWithIntValue{id, 1, 42})  // should clamp to upper bound
	}
	result := []testInt64Metric{
		{0, 100}, // each aggregated record in partition 0 must be clamped to 2
		{1, 150}, // each aggregated record in partition 1 must be clamped to 3
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

	// ε=60, δ=0.01 and l1Sensitivity=6 gives a threshold of ≈2.
	// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁵ probability (k=25).
	epsilon, delta, k, l1Sensitivity := 60.0, 0.01, 25.0, 6.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	pcol = ParDo(s, tripleWithIntValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MinValue: 2, MaxValue: 3, MaxPartitionsContributed: 2, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, int64MetricToKV, want)
	if err := approxEqualsKVInt64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestSumPerKeyPerPartitionContributionBoundingInt: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyPerPartitionContributionBoundingInt: SumPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

// Checks that SumPerKey does per-partition contribution bounding correctly for floats.
func TestSumPerKeyPerPartitionContributionBoundingFloat(t *testing.T) {
	var triples []tripleWithFloatValue
	for id := 1; id <= 50; id++ {
		triples = append(triples, tripleWithFloatValue{id, 0, 1}) // partition 0 is associated to 50 times 1
		triples = append(triples, tripleWithFloatValue{id, 1, 4}) // partition 1 is associated to 50 times 4
		// Additional values that are outside of range [lower, upper]
		triples = append(triples, tripleWithFloatValue{id, 0, -17}) // should clamp to lower bound
		triples = append(triples, tripleWithFloatValue{id, 1, 42})  // should clamp to upper bound
	}
	result := []testFloat64Metric{
		{0, 100.0}, // each aggregated record in partition 0 must be clamped to 2.0
		{1, 150.0}, // each aggregated record in partition 1 must be clamped to 3.0
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)

	// ε=60, δ=0.01 and l1Sensitivity=6 gives a threshold of ≈2.
	// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁵ probability (k=25).
	epsilon, delta, k, l1Sensitivity := 60.0, 0.01, 25.0, 6.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MinValue: 2.0, MaxValue: 3.0, MaxPartitionsContributed: 2, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, float64MetricToKV, want)
	if err := approxEqualsKVFloat64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestSumPerKeyPerPartitionContributionBoundingFloat: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyPerPartitionContributionBoundingFloat: SumPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

var sumPartitionSelectionNonDeterministicTestCases = []struct {
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

// Checks that SumPerKey is performing a random partition selection.
func TestSumPartitionSelectionNonDeterministicInt(t *testing.T) {
	for _, tc := range sumPartitionSelectionNonDeterministicTestCases {
		t.Run(tc.name, func(t *testing.T) {
			// Sanity check that the entriesPerPartition is sensical.
			if tc.entriesPerPartition <= 0 {
				t.Errorf("Invalid test case: entriesPerPartition must be positive. Got: %d", tc.entriesPerPartition)
			}

			// Build up {ID, Partition, Value} pairs such that each user contributes
			// 1 value to 1 partition:
			//    {0, 0, 1}, {1, 0, 1}, …, {entriesPerPartition-1, 0, 1}
			//    {entriesPerPartition, 1, 1}, {entriesPerPartition+1, 1, 1}, …, {entriesPerPartition+entriesPerPartition-1, 1, 1}
			//    …
			//    {entriesPerPartition*(numPartitions-1), numPartitions-1, 1}, …, {entriesPerPartition*numPartitions-1, numPartitions-1, 1}
			var triples []tripleWithIntValue
			var kOffset = 0
			for i := 0; i < tc.numPartitions; i++ {
				triples = append(triples, makeDummyTripleWithIntValueStartingFromKey(kOffset, tc.entriesPerPartition, i)...)
				kOffset += tc.entriesPerPartition
			}
			p, s, col := ptest.CreateList(triples)
			col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

			// Run SumPerKey on triples
			pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
			pcol = ParDo(s, tripleWithIntValueToKV, pcol)
			got := SumPerKey(s, pcol, SumParams{MinValue: 0, MaxValue: 1, NoiseKind: tc.noiseKind, MaxPartitionsContributed: 1})
			got = beam.ParDo(s, kvToInt64Metric, got)

			// Validate that partitions are selected randomly (i.e., some emitted and some dropped).
			checkSomePartitionsAreDropped(s, got, tc.numPartitions)
			if err := ptest.Run(p); err != nil {
				t.Errorf("%v", err)
			}
		})
	}
}

// Checks that SumPerKey is performing a random partition selection.
func TestSumPartitionSelectionNonDeterministicFloat(t *testing.T) {
	for _, tc := range sumPartitionSelectionNonDeterministicTestCases {
		t.Run(tc.name, func(t *testing.T) {
			// Sanity check that the entriesPerPartition is sensical.
			if tc.entriesPerPartition <= 0 {
				t.Errorf("Invalid test case: entriesPerPartition must be positive. Got: %d", tc.entriesPerPartition)
			}

			// Build up {ID, Partition, Value} pairs such that each user contributes
			// 1 value to 1 partition:
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

			// Run SumPerKey on triples
			pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
			pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
			got := SumPerKey(s, pcol, SumParams{MinValue: 0.0, MaxValue: 1.0, MaxPartitionsContributed: 1, NoiseKind: tc.noiseKind})
			got = beam.ParDo(s, kvToFloat64Metric, got)

			// Validate that partitions are selected randomly (i.e., some emitted and some dropped).
			checkSomePartitionsAreDropped(s, got, tc.numPartitions)
			if err := ptest.Run(p); err != nil {
				t.Errorf("%v", err)
			}
		})
	}
}

func TestFindConvertFn(t *testing.T) {
	for _, tc := range []struct {
		desc          string
		fullType      typex.FullType
		wantConvertFn interface{}
		wantErr       bool
	}{
		{"int", typex.New(reflect.TypeOf(int(0))), convertIntToInt64Fn, false},
		{"int8", typex.New(reflect.TypeOf(int8(0))), convertInt8ToInt64Fn, false},
		{"int16", typex.New(reflect.TypeOf(int16(0))), convertInt16ToInt64Fn, false},
		{"int32", typex.New(reflect.TypeOf(int32(0))), convertInt32ToInt64Fn, false},
		{"int64", typex.New(reflect.TypeOf(int64(0))), convertInt64ToInt64Fn, false},
		{"uint", typex.New(reflect.TypeOf(uint(0))), convertUintToInt64Fn, false},
		{"uint8", typex.New(reflect.TypeOf(uint8(0))), convertUint8ToInt64Fn, false},
		{"uint16", typex.New(reflect.TypeOf(uint16(0))), convertUint16ToInt64Fn, false},
		{"uint32", typex.New(reflect.TypeOf(uint32(0))), convertUint32ToInt64Fn, false},
		{"uint64", typex.New(reflect.TypeOf(uint64(0))), convertUint64ToInt64Fn, false},
		{"float32", typex.New(reflect.TypeOf(float32(0))), convertFloat32ToFloat64Fn, false},
		{"float64", typex.New(reflect.TypeOf(float64(0))), convertFloat64ToFloat64Fn, false},
		{"string", typex.New(reflect.TypeOf("")), nil, true},
	} {
		convertFn, err := findConvertFn(tc.fullType)
		if (err != nil) != tc.wantErr {
			t.Errorf("With %s, got=%v error, wantErr=%t", tc.desc, err, tc.wantErr)
		}
		if !reflect.DeepEqual(reflect.TypeOf(convertFn), reflect.TypeOf(tc.wantConvertFn)) {
			t.Errorf("With %s, got=%v , expected=%v", tc.desc, convertFn, tc.wantConvertFn)
		}
	}
}

func TestGetKind(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		convertFn interface{}
		wantKind  reflect.Kind
		wantErr   bool
	}{
		{"convertIntToInt64Fn", convertIntToInt64Fn, reflect.Int64, false},
		{"convertInt8ToInt64Fn", convertInt8ToInt64Fn, reflect.Int64, false},
		{"convertInt16ToInt64Fn", convertInt16ToInt64Fn, reflect.Int64, false},
		{"convertInt32ToInt64Fn", convertInt32ToInt64Fn, reflect.Int64, false},
		{"convertInt64ToInt64Fn", convertInt64ToInt64Fn, reflect.Int64, false},
		{"convertUintToInt64Fn", convertUintToInt64Fn, reflect.Int64, false},
		{"convertUint8ToInt64Fn", convertUint8ToInt64Fn, reflect.Int64, false},
		{"convertUint16ToInt64Fn", convertUint16ToInt64Fn, reflect.Int64, false},
		{"convertUint32ToInt64Fn", convertUint32ToInt64Fn, reflect.Int64, false},
		{"convertUint64ToInt64Fn", convertUint64ToInt64Fn, reflect.Int64, false},
		{"convertFloat32ToFloat64Fn", convertFloat32ToFloat64Fn, reflect.Float64, false},
		{"convertFloat64Fn", convertFloat64ToFloat64Fn, reflect.Float64, false},
		{"nil interface", nil, reflect.Invalid, true},
		{"function with less than 2 return values", func() int64 { return int64(0) }, reflect.Invalid, true},
	} {
		kind, err := getKind(tc.convertFn)
		if (err != nil) != tc.wantErr {
			t.Errorf("With %s, got=%v error, wantErr=%t", tc.desc, err, tc.wantErr)
		}
		if !reflect.DeepEqual(kind, tc.wantKind) {
			t.Errorf("With %s, got=%v , expected=%v", tc.desc, kind, tc.wantKind)
		}
	}
}

// Expect non-negative results if MinValue >= 0 for float64 values.
func TestSumPerKeyReturnsNonNegativeFloat64(t *testing.T) {
	var triples []tripleWithFloatValue
	for key := 0; key < 100; key++ {
		triples = append(triples, tripleWithFloatValue{key, key, 0.01})
	}
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)
	// Using a low epsilon, a high delta, and a high maxValue here to add a
	// lot of noise while keeping partitions.
	epsilon, delta, maxValue := 0.001, 0.999, 1e8
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
	sums := SumPerKey(s, pcol, SumParams{MinValue: 0, MaxValue: maxValue, MaxPartitionsContributed: 1, NoiseKind: GaussianNoise{}})
	values := beam.DropKey(s, sums)
	beam.ParDo0(s, checkNoNegativeValuesFloat64Fn, values)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyReturnsNonNegativeFloat64 returned errors: %v", err)
	}
}

// Expect non-negative results if MinValue >= 0 for int64 values.
func TestSumPerKeyReturnsNonNegativeInt64(t *testing.T) {
	var triples []tripleWithIntValue
	for key := 0; key < 100; key++ {
		triples = append(triples, tripleWithIntValue{key, key, 1})
	}
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)
	// Using a low epsilon, a high delta, and a high maxValue here to add a
	// lot of noise while keeping partitions.
	epsilon, delta, maxValue := 0.001, 0.999, 1e8
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	pcol = ParDo(s, tripleWithIntValueToKV, pcol)
	sums := SumPerKey(s, pcol, SumParams{MinValue: 0, MaxValue: maxValue, MaxPartitionsContributed: 1, NoiseKind: GaussianNoise{}})
	values := beam.DropKey(s, sums)
	beam.ParDo0(s, checkNoNegativeValuesInt64Fn, values)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyReturnsNonNegativeInt64 returned errors: %v", err)
	}
}

// Expect at least one negative value after post-aggregation clamping when
// MinValue < 0 for float64 values.
func TestSumPerKeyNoClampingForNegativeMinValueFloat64(t *testing.T) {
	var triples []tripleWithFloatValue
	for key := 0; key < 1000; key++ {
		triples = append(triples, tripleWithFloatValue{key, key, 0})
	}
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, extractIDFromTripleWithFloatValue, col)
	// Using `typical` privacy parameters with a high delta to keep
	// partitions.
	epsilon, delta, minValue, maxValue := 0.1, 0.999, -100.0, 100.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	pcol = ParDo(s, tripleWithFloatValueToKV, pcol)
	sums := SumPerKey(s, pcol, SumParams{MinValue: minValue, MaxValue: maxValue, MaxPartitionsContributed: 1, NoiseKind: GaussianNoise{}})
	values := beam.DropKey(s, sums)
	mValue := stats.Min(s, values)
	beam.ParDo0(s, checkAllValuesNegativeFloat64Fn, mValue)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyNoClampingForNegativeMinValueFloat64 returned errors: %v", err)
	}
}

func checkAllValuesNegativeInt64Fn(v int64) error {
	if v >= 0 {
		return fmt.Errorf("unexpected non-negative element: %v", v)
	}
	return nil
}

// Expect at least one negative value after post-aggregation clamping when
// MinValue < 0 for int64 values.
func TestSumPerKeyNoClampingForNegativeMinValueInt64(t *testing.T) {
	var triples []tripleWithIntValue
	for key := 0; key < 1000; key++ {
		triples = append(triples, tripleWithIntValue{key, key, 0})
	}
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)
	// Using `typical` privacy parameters with a high delta to keep
	// partitions.
	epsilon, delta, minValue, maxValue := 0.1, 0.999, -100.0, 100.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	pcol = ParDo(s, tripleWithIntValueToKV, pcol)
	sums := SumPerKey(s, pcol, SumParams{MinValue: minValue, MaxValue: maxValue, MaxPartitionsContributed: 1, NoiseKind: GaussianNoise{}})
	values := beam.DropKey(s, sums)
	mValue := stats.Min(s, values)
	beam.ParDo0(s, checkAllValuesNegativeInt64Fn, mValue)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyNoClampingForNegativeMinValueInt64 returned errors: %v", err)
	}
}
