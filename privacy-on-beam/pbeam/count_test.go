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
	// we can have each partition fail with 1-10⁻²⁵ probability (k=25).
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
		// Test is considered to pass if it passes during any run. Thus, numTries is
		// used to reduce Flakiness to negligible levels.
		numTries int
		// numIDs controls the number of distinct IDs associated with a value.
		numIDs int
		// Differential privacy params used. The test assumes sensitivities of 1.
		epsilon float64
		delta   float64
	}{
		{
			// The choice of ε=1, δ=0.01, and sensitivities of 1, the Gaussian threshold
			// is ≈28 and σ≈10.5. With numIDs one order of magnitude higher than the
			// threshold, the chance of pairs being reduced below the threshold by noise
			// is negligible. The probability that no noise is added is ≈4%
			name:      "Gaussian",
			noiseKind: GaussianNoise{},
			// Each run should fail with probability <4% (the chance that no noise is
			// added). Running 17 times reduces flakes to a negligible rate:
			// math.Pow(0.04, 17) = 1.7e-24.
			numTries: 17,
			numIDs:   280,
			epsilon:  1,
			delta:    0.01,
		},
		{
			// ε=0.001, δ=0.499 and sensitivity=1 gives a threshold of ≈160.  With such
			// a small ε, noise is added with probability > 99.5%. With
			// numIDs of 2000, noise will keep us above the threshold with very high
			// (>1-10⁻⁸) probability.
			name:      "Laplace",
			noiseKind: LaplaceNoise{},
			// Each run should fail with probability <0.5% (the chance that no noise is
			// added). Running 10 times gives a trivial flake rate:
			// math.Pow(0.005, 10) ≈ 10⁻²³.
			numTries: 10,
			numIDs:   2000,
			epsilon:  0.001,
			delta:    0.499,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fail := true
			for try := 0; try < tc.numTries && fail; try++ {
				// pairs contains {1,0}, {2,0}, …, {numIDs,0}.
				pairs := makePairsWithFixedV(tc.numIDs, 0)
				p, s, col := ptest.CreateList(pairs)
				col = beam.ParDo(s, pairToKV, col)

				pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
				got := Count(s, pcol, CountParams{MaxPartitionsContributed: 1, MaxValue: 1, NoiseKind: tc.noiseKind})
				got = beam.ParDo(s, kvToInt64Metric, got)
				checkInt64MetricsAreNoisy(s, got, tc.numIDs)
				if err := ptest.Run(p); err == nil {
					fail = false
				}
			}
			if fail {
				t.Errorf("Count didn't add any noise, %d times in a row.", tc.numTries)
			}
		})
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
	// we can have each partition fail with 1-10⁻²⁵ probability (k=25).
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
