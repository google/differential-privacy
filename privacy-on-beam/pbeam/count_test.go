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
	"testing"

	"github.com/google/differential-privacy/go/dpagg"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/testing/ptest"
	"github.com/apache/beam/sdks/go/pkg/beam/transforms/stats"
)

// Checks that Count returns a correct answer: duplicated pairs must be
// counted multiple times, but not too many times.
func TestCountNoNoise(t *testing.T) {
	// In this test, we set the per-partition l1Sensitivity to 2, and:
	// - value 0 is associated to 7 users, so it should be thresholded;
	// - value 1 is associated to 52 users appearing twice each, so each of
	//   them should be counted twice;
	// - value 2 is associated to 99 users appearing 3 times each, but the
	//   l1Sensitivity is 2, so each should only be counted twice.
	// Each user contributes to at most 1 partition.
	pairs := concatenatePairs(
		makePairsWithFixedVStartingFromKey(0, 7, 0),
		makePairsWithFixedVStartingFromKey(7, 52, 1),
		makePairsWithFixedVStartingFromKey(7, 52, 1),
		makePairsWithFixedVStartingFromKey(7+52, 99, 2),
		makePairsWithFixedVStartingFromKey(7+52, 99, 2),
		makePairsWithFixedVStartingFromKey(7+52, 99, 2),
	)
	result := []testInt64Metric{
		{1, 104}, // 52*2
		{2, 198}, // 99*2
	}
	p, s, col, want := ptest.CreateList2(pairs, result)
	col = beam.ParDo(s, pairToKV, col)

	// ε=50, δ=10⁻²⁰⁰ and l1Sensitivity=2 gives a threshold of ≈38.
	// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁵ probability (k=25).
	// To see the logic and the math behind flakiness and tolerance calculation,
	// See https://github.com/google/differential-privacy/blob/master/privacy-on-beam/docs/Tolerance_Calculation.pdf.
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-200, 25.0, 2.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	got := Count(s, pcol, CountParams{MaxValue: 2, MaxPartitionsContributed: 1, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, int64MetricToKV, want)
	if err := approxEqualsKVInt64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestCountNoNoise: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestCountNoNoise: Count(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

// Checks that Count is performing a random partition selection.
func TestCountPartitionSelectionNonDeterministic(t *testing.T) {
	for _, tc := range []struct {
		name          string
		noiseKind     NoiseKind
		epsilon       float64
		delta         float64
		numPartitions int
		countPerValue int
	}{
		{
			name:      "Gaussian",
			noiseKind: GaussianNoise{},
			// After splitting the (ε, δ) budget between the noise and partition
			// selection portions of the privacy algorithm, this results in a ε=1,
			// δ=0.3 partition selection budget.
			epsilon: 2,
			delta:   0.6,
			// countPerValue=1 yields a 30% chance of emitting any particular partition
			// (since δ_emit=0.3).
			countPerValue: 1,
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
			// countPerValue=1 yields a 30% chance of emitting any particular partition
			// (since δ_emit=0.3).
			countPerValue: 1,
			numPartitions: 143,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Sanity check that the countPerValue is sensical.
			if tc.countPerValue <= 0 {
				t.Errorf("Invalid test case: countPerValue must be positive. Got: %d", tc.countPerValue)
			}

			// Build up {ID, Value} pairs such that each user contributes at most 1
			// value to at most 1 partition:
			//    {0,0}, {1,0}, …, {countPerValue-1,0}
			//    {countPerValue,1}, {countPerValue+1,1}, …, {countPerValue*2-1,1}
			//    …
			//    {countPerValue*(numPartitions-1),numPartitions-1}, …, {countPerValue*numPartitions-1, numPartitions-1}
			var pairs []pairII
			var kOffset = 0
			for i := 0; i < tc.numPartitions; i++ {
				pairs = append(pairs, makePairsWithFixedVStartingFromKey(kOffset, int(tc.countPerValue), i)...)
				kOffset += tc.countPerValue
			}
			p, s, col := ptest.CreateList(pairs)
			col = beam.ParDo(s, pairToKV, col)

			// Run Count on pairs
			pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
			got := Count(s, pcol, CountParams{MaxValue: 1, MaxPartitionsContributed: 1, NoiseKind: tc.noiseKind})
			got = beam.ParDo(s, kvToInt64Metric, got)

			// Validate that partitions are selected randomly (i.e., some emitted and some dropped).
			checkSomePartitionsAreDropped(s, got, tc.numPartitions)
			if err := ptest.Run(p); err != nil {
				t.Errorf("%v", err)
			}
		})
	}
}

// Checks that Count adds noise to its output.
func TestCountAddsNoise(t *testing.T) {
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

		// pairs contains {1,0}, {2,0}, …, {numIDs,0}.
		pairs := makePairsWithFixedV(numIDs, 0)
		p, s, col := ptest.CreateList(pairs)
		col = beam.ParDo(s, pairToKV, col)

		pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
		got := Count(s, pcol, CountParams{MaxPartitionsContributed: 1, MaxValue: 1, NoiseKind: tc.noiseKind})
		got = beam.ParDo(s, kvToInt64Metric, got)
		checkInt64MetricsAreNoisy(s, got, numIDs, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("CountPerKey didn't add any noise: %v", err)
		}
	}
}

// Checks that Count bounds per-user contributions correctly.
func TestCountCrossPartitionContributionBounding(t *testing.T) {
	// pairs contains {1,0}, {2,0}, …, {50,0}, {1,1}, …, {50,1}, {1,2}, …, {50,9}.
	var pairs []pairII
	for i := 0; i < 10; i++ {
		pairs = append(pairs, makePairsWithFixedV(50, i)...)
	}
	result := []testInt64Metric{
		{0, 150},
	}
	p, s, col, want := ptest.CreateList2(pairs, result)
	col = beam.ParDo(s, pairToKV, col)

	// ε=50, δ=0.01 and l1Sensitivity=3 gives a threshold of 1.
	// We have 10 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁵ probability (k=25).
	epsilon, delta, k, l1Sensitivity := 50.0, 0.01, 25.0, 3.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	got := Count(s, pcol, CountParams{MaxPartitionsContributed: 3, MaxValue: 1, NoiseKind: LaplaceNoise{}})
	// With a max contribution of 3, 70% of the data should have be
	// dropped. The sum of all elements must then be 150.
	counts := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, counts)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
	want = beam.ParDo(s, int64MetricToKV, want)
	if err := approxEqualsKVInt64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestCountCrossPartitionContributionBounding: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestCountCrossPartitionContributionBounding: Metric(%v) = %v, expected elements to sum to 150: %v", col, got, err)
	}
}

// Check that no negative values are returned from Count.
func TestCountReturnsNonNegative(t *testing.T) {
	var pairs []pairII
	for i := 0; i < 100; i++ {
		pairs = append(pairs, pairII{i, i})
	}
	p, s, col := ptest.CreateList(pairs)
	col = beam.ParDo(s, pairToKV, col)
	// Using a low epsilon and high maxValue adds a lot of noise and using
	// a high delta keeps many partitions.
	epsilon, delta, maxValue := 0.001, 0.999, int64(1e8)
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	counts := Count(s, pcol, CountParams{MaxValue: maxValue, MaxPartitionsContributed: 1, NoiseKind: GaussianNoise{}})
	values := beam.DropKey(s, counts)
	// Check if we have negative elements.
	beam.ParDo0(s, checkNoNegativeValuesInt64Fn, values)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestCountReturnsNonNegative returned errors: %v", err)
	}
}
