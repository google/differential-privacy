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

	"github.com/google/differential-privacy/go/v3/dpagg"
	"github.com/google/differential-privacy/go/v3/noise"
	"github.com/google/differential-privacy/privacy-on-beam/v3/pbeam/testutils"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/testing/ptest"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/stats"
)

// Checks that Count returns a correct answer by verifying:
// - partition selection is applied
// - per-partition contribution bounding is applied
// - duplicate user contributions within per-partition contribution bound aren't dropped
func TestCountNoNoise(t *testing.T) {
	// In this test, we set the per-partition l1Sensitivity to 2, and:
	// - value 0 is associated with 7 privacy units, so it should be thresholded;
	// - value 1 is associated with 30 privacy units appearing twice each, so each of
	//   them should be counted twice;
	// - value 2 is associated with 50 privacy units appearing 3 times each, but the
	//   l1Sensitivity is 2, so each should only be counted twice.
	// Each privacy unit contributes to at most 1 partition.
	pairs := testutils.ConcatenatePairs(
		testutils.MakePairsWithFixedVStartingFromKey(0, 7, 0),
		testutils.MakePairsWithFixedVStartingFromKey(7, 30, 1),
		testutils.MakePairsWithFixedVStartingFromKey(7, 30, 1),
		testutils.MakePairsWithFixedVStartingFromKey(7+30, 50, 2),
		testutils.MakePairsWithFixedVStartingFromKey(7+30, 50, 2),
		testutils.MakePairsWithFixedVStartingFromKey(7+30, 50, 2),
	)
	result := []testutils.PairII64{
		{1, 60},  // 30*2
		{2, 100}, // 50*2
	}
	p, s, col, want := ptest.CreateList2(pairs, result)
	col = beam.ParDo(s, testutils.PairToKV, col)

	// ε=25, δ=10⁻²⁰⁰ and l0Sensitivity=1 gives a threshold of ≈21.
	// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
	// To see the logic and the math behind flakiness and tolerance calculation,
	// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf.
	epsilon, delta, k, l1Sensitivity := 25.0, 1e-200, 24.0, 2.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	got := Count(s, pcol, CountParams{MaxValue: 2, MaxPartitionsContributed: 1, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, testutils.PairII64ToKV, want)
	testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestCountNoNoise: Count(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

// Checks that Count with partitions returns a correct answer.
func TestCountWithPartitionsNoNoise(t *testing.T) {
	// We have two test cases, one for public partitions as a PCollection and one for public partitions as a slice (i.e., in-memory).
	for _, tc := range []struct {
		inMemory bool
	}{
		{true},
		{false},
	} {
		var pairs []testutils.PairII
		for i := 0; i < 10; i++ {
			pairs = append(pairs, testutils.PairII{1, i})
		}
		result := []testutils.PairII64{
			// Drop partitions 0 to 8 as they are not in public
			// partitions.
			{9, 1},  // Keep partition 9.
			{10, 0}, // Add partition 10.
		}

		p, s, col, want := ptest.CreateList2(pairs, result)
		col = beam.ParDo(s, testutils.PairToKV, col)
		publicPartitionsSlice := []int{9, 10}
		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		// We use ε=50, δ=0 and l1Sensitivity=2.
		// We have 2 partitions. So, to get an overall flakiness of 10⁻²³,
		// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
		epsilon, delta, k, l1Sensitivity := 50.0, 0.0, 24.0, 2.0
		pcol := MakePrivate(s, col, privacySpec(t,
			PrivacySpecParams{
				AggregationEpsilon:      epsilon,
				PartitionSelectionDelta: delta,
			}))
		countParams := CountParams{MaxValue: 2, MaxPartitionsContributed: 1, NoiseKind: LaplaceNoise{}, PublicPartitions: publicPartitions}

		got := Count(s, pcol, countParams)
		want = beam.ParDo(s, testutils.PairII64ToKV, want)
		testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestCountWithPartitionsNoNoise in-memory=%t: Count(%v) = %v, expected %v: %v", tc.inMemory, col, got, want, err)
		}
	}

}

// Checks that Count applies partition selection.
func TestCountPartitionSelection(t *testing.T) {
	for _, tc := range []struct {
		name                      string
		noiseKind                 NoiseKind
		aggregationEpsilon        float64
		aggregationDelta          float64
		partitionSelectionEpsilon float64
		partitionSelectionDelta   float64
		numPartitions             int
		countPerValue             int
	}{
		{
			name:                      "Gaussian",
			noiseKind:                 GaussianNoise{},
			aggregationEpsilon:        1,
			aggregationDelta:          0.3,
			partitionSelectionEpsilon: 1,
			partitionSelectionDelta:   0.3,
			// countPerValue=1 yields a 30% chance of emitting any particular partition
			// (since δ_emit=0.3).
			countPerValue: 1,
			// 143 distinct partitions implies that some (but not all) partitions are
			// emitted with high probability (at least 1 - 1e-20).
			numPartitions: 143,
		},
		{
			name:                      "Laplace",
			noiseKind:                 LaplaceNoise{},
			aggregationEpsilon:        1,
			partitionSelectionEpsilon: 1,
			partitionSelectionDelta:   0.3,
			// countPerValue=1 yields a 30% chance of emitting any particular partition
			// (since δ_emit=0.3).
			countPerValue: 1,
			numPartitions: 143,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Verify that countPerValue is sensical.
			if tc.countPerValue <= 0 {
				t.Fatalf("Invalid test case: countPerValue must be positive. Got: %d", tc.countPerValue)
			}

			// Build up {ID, Value} pairs such that tc.countPerValue privacy units
			// contribute to each of the tc.numPartitions partitions:
			//    {0,0}, {1,0}, …, {countPerValue-1,0}
			//    {countPerValue,1}, {countPerValue+1,1}, …, {countPerValue*2-1,1}
			//    …
			//    {countPerValue*(numPartitions-1),numPartitions-1}, …, {countPerValue*numPartitions-1, numPartitions-1}
			var (
				pairs   []testutils.PairII
				kOffset = 0
			)
			for i := 0; i < tc.numPartitions; i++ {
				for j := 0; j < tc.countPerValue; j++ {
					pairs = append(pairs, testutils.PairII{kOffset + j, i})
				}
			}
			p, s, col := ptest.CreateList(pairs)
			col = beam.ParDo(s, testutils.PairToKV, col)

			// Run Count on pairs
			pcol := MakePrivate(s, col, privacySpec(t,
				PrivacySpecParams{
					AggregationEpsilon:        tc.aggregationEpsilon,
					AggregationDelta:          tc.aggregationDelta,
					PartitionSelectionEpsilon: tc.partitionSelectionEpsilon,
					PartitionSelectionDelta:   tc.partitionSelectionDelta,
				}))
			got := Count(s, pcol, CountParams{MaxValue: 1, MaxPartitionsContributed: 1, NoiseKind: tc.noiseKind})
			got = beam.ParDo(s, testutils.KVToPairII64, got)

			// Validate that partition selection is applied (i.e., some emitted and some dropped).
			testutils.CheckSomePartitionsAreDropped(s, got, tc.numPartitions)
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
		aggregationEpsilon        float64
		aggregationDelta          float64
		partitionSelectionEpsilon float64
		partitionSelectionDelta   float64
	}{
		{
			name:                      "Gaussian",
			noiseKind:                 GaussianNoise{},
			aggregationEpsilon:        1e-15,
			aggregationDelta:          1e-5,
			partitionSelectionEpsilon: 1e-15,
			partitionSelectionDelta:   1e-5,
		},
		{
			name:                      "Laplace",
			noiseKind:                 LaplaceNoise{},
			aggregationEpsilon:        1e-15,
			partitionSelectionEpsilon: 1e-15,
			partitionSelectionDelta:   1e-5,
		},
	} {
		// Because this is an integer aggregation, we can't use the regular complementary
		// tolerance computations. Instead, we do the following:
		//
		// If generated noise is between -0.5 and 0.5, it will be rounded to 0 and the
		// test will fail. For Laplace, this will happen with probability
		//   P ~= Laplace_CDF(0.5) - Laplace_CDF(-0.5).
		// Given that Laplace scale = l1_sensitivity / ε = 10¹⁵, P ~= 5e-16.
		// For Gaussian, this will happen with probability
		//	 P ~= Gaussian_CDF(0.5) - Gaussian_CDF(-0.5).
		// For given ε=1e-15, δ=1e-5 => sigma = 39904, P ~= 1e-5.
		//
		// We want to keep numIDs low (otherwise the tests take a long time) while
		// also keeping P low. We use magic partition selection here, meaning that
		// numIDs cap at 1/δ. So, we can have tiny epsilon without having to worry
		// about tests taking long.
		tolerance := 0.0
		l0Sensitivity, lInfSensitivity := int64(1), int64(1)

		// Compute the number of IDs needed to keep the partition.
		sp, err := dpagg.NewPreAggSelectPartition(
			&dpagg.PreAggSelectPartitionOptions{
				Epsilon:                  tc.partitionSelectionEpsilon,
				Delta:                    tc.partitionSelectionDelta,
				MaxPartitionsContributed: l0Sensitivity,
			})
		if err != nil {
			t.Fatalf("Couldn't initialize PreAggSelectPartition necessary to compute the number of IDs needed: %v", err)
		}
		numIDs, err := sp.GetHardThreshold()
		if err != nil {
			t.Fatalf("Couldn't compute hard threshold: %v", err)
		}

		// pairs contains {1,0}, {2,0}, …, {numIDs,0}.
		pairs := testutils.MakePairsWithFixedV(numIDs, 0)
		p, s, col := ptest.CreateList(pairs)
		col = beam.ParDo(s, testutils.PairToKV, col)

		pcol := MakePrivate(s, col, privacySpec(t,
			PrivacySpecParams{
				AggregationEpsilon:        tc.aggregationEpsilon,
				AggregationDelta:          tc.aggregationDelta,
				PartitionSelectionEpsilon: tc.partitionSelectionEpsilon,
				PartitionSelectionDelta:   tc.partitionSelectionDelta,
			}))
		got := Count(s, pcol, CountParams{MaxPartitionsContributed: l0Sensitivity, MaxValue: lInfSensitivity, NoiseKind: tc.noiseKind})
		got = beam.ParDo(s, testutils.KVToPairII64, got)
		testutils.CheckInt64MetricsAreNoisy(s, got, numIDs, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("CountPerKey didn't add any %s noise: %v", tc.name, err)
		}
	}
}

// Checks that Count with partitions adds noise to its output.
func TestCountAddsNoiseWithPartitions(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		noiseKind NoiseKind
		// Differential privacy params used.
		epsilon  float64
		delta    float64
		inMemory bool
	}{
		// ε & δ are not split because partitions are public. All of them are used for the noise.
		{
			desc:      "as PCollection w/ Gaussian",
			noiseKind: GaussianNoise{},
			epsilon:   1e-15,
			delta:     1e-15,
			inMemory:  false,
		},
		{
			desc:      "as slice w/ Gaussian",
			noiseKind: GaussianNoise{},
			epsilon:   1e-15,
			delta:     1e-15,
			inMemory:  true,
		},
		{
			desc:      "as PCollection w/ Laplace",
			noiseKind: LaplaceNoise{},
			epsilon:   1e-15,
			delta:     0, // It is 0 because partitions are public and we are using Laplace noise.
			inMemory:  false,
		},
		{
			desc:      "as slice w/ Laplace",
			noiseKind: LaplaceNoise{},
			epsilon:   1e-15,
			delta:     0, // It is 0 because partitions are public and we are using Laplace noise.
			inMemory:  true,
		},
	} {
		// Because this is an integer aggregation, we can't use the regular complementary
		// tolerance computations. Instead, we do the following:
		//
		// If generated noise is between -0.5 and 0.5, it will be rounded to 0 and the
		// test will fail. For Laplace, this will happen with probability
		//   P ~= Laplace_CDF(0.5) - Laplace_CDF(-0.5).
		// Given that Laplace scale = l1_sensitivity / ε = 10¹⁵, P ~= 5e-16.
		// For Gaussian, this will happen with probability
		//	 P ~= Gaussian_CDF(0.5) - Gaussian_CDF(-0.5).
		// For given ε=1e-15, δ=1e-15 => sigma = 261134011596800, P ~= 1e-15.
		//
		// Since no partitions selection / thresholding happens, numIDs doesn't depend
		// on ε & δ. We can use arbitrarily small ε & δ.
		tolerance := 0.0
		l0Sensitivity, lInfSensitivity := int64(1), int64(1)
		numIDs := 10

		// pairs contains {1,0}, {2,0}, …, {10,0}.
		pairs := testutils.MakePairsWithFixedV(numIDs, 0)
		p, s, col := ptest.CreateList(pairs)

		publicPartitionsSlice := []int{0}
		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		col = beam.ParDo(s, testutils.PairToKV, col)
		pcol := MakePrivate(s, col, privacySpec(t,
			PrivacySpecParams{
				AggregationEpsilon: tc.epsilon,
				AggregationDelta:   tc.delta,
			}))
		countParams := CountParams{MaxPartitionsContributed: l0Sensitivity, MaxValue: lInfSensitivity, NoiseKind: tc.noiseKind, PublicPartitions: publicPartitions}
		got := Count(s, pcol, countParams)
		got = beam.ParDo(s, testutils.KVToPairII64, got)
		testutils.CheckInt64MetricsAreNoisy(s, got, 10, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("CountPerKey with public partitions %s didn't add any noise: %v", tc.desc, err)
		}
	}
}

// Checks that Count bounds cross-partition contributions correctly.
func TestCountCrossPartitionContributionBounding(t *testing.T) {
	// pairs contains {1,0}, {2,0}, …, {50,0}, {1,1}, …, {50,1}, {1,2}, …, {50,9}.
	var pairs []testutils.PairII
	for i := 0; i < 10; i++ {
		pairs = append(pairs, testutils.MakePairsWithFixedV(50, i)...)
	}
	result := []testutils.PairII64{
		{0, 150},
	}
	p, s, col, want := ptest.CreateList2(pairs, result)
	col = beam.ParDo(s, testutils.PairToKV, col)

	// ε=50, δ=0.01 and l0Sensitivity=3 gives a threshold of 3.
	// We have 10 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
	epsilon, delta, k, l1Sensitivity := 50.0, 0.01, 25.0, 3.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	got := Count(s, pcol, CountParams{MaxPartitionsContributed: 3, MaxValue: 1, NoiseKind: LaplaceNoise{}})
	// With a max contribution of 3, 70% of the data should be
	// dropped. The sum of all elements must then be 150.
	counts := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, counts)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
	want = beam.ParDo(s, testutils.PairII64ToKV, want)
	testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestCountCrossPartitionContributionBounding: Metric(%v) = %v, expected elements to sum to 150: %v", col, got, err)
	}
}

// Checks that Count with partitions bounds per-user contributions correctly.
func TestCountWithPartitionsCrossPartitionContributionBounding(t *testing.T) {
	// We have two test cases, one for public partitions as a PCollection and one for public partitions as a slice (i.e., in-memory).
	for _, tc := range []struct {
		inMemory bool
	}{
		{true},
		{false},
	} {
		// pairs contains {1,0}, {2,0}, …, {50,0}, {1,1}, …, {50,1}, {1,2}, …, {50,9}.
		var pairs []testutils.PairII
		for i := 0; i < 10; i++ {
			pairs = append(pairs, testutils.MakePairsWithFixedV(50, i)...)
		}
		result := []testutils.PairII64{
			{0, 150},
		}

		p, s, col, want := ptest.CreateList2(pairs, result)
		col = beam.ParDo(s, testutils.PairToKV, col)

		publicPartitionsSlice := []int{0, 1, 2, 3, 4}
		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		// We have 5 partitions. So, to get an overall flakiness of 10⁻²³,
		// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
		epsilon, delta, k, l1Sensitivity := 50.0, 0.0, 24.0, 3.0
		pcol := MakePrivate(s, col, privacySpec(t,
			PrivacySpecParams{
				AggregationEpsilon:      epsilon,
				PartitionSelectionDelta: delta,
			}))
		countParams := CountParams{MaxPartitionsContributed: 3, MaxValue: 1, NoiseKind: LaplaceNoise{}, PublicPartitions: publicPartitions}
		got := Count(s, pcol, countParams)
		// With a max contribution of 3, 40% of the data from the public partitions should be dropped.
		// The sum of all elements must then be 150.
		counts := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, counts)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.PairII64ToKV, want)
		testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestCountWithPartitionsCrossPartitionContributionBounding in-memory=%t: Metric(%v) = %v, expected elements to sum to 150: %v", tc.inMemory, col, got, err)
		}
	}
}

func TestCheckCountParams(t *testing.T) {
	_, _, partitions := ptest.CreateList([]int{0})
	for _, tc := range []struct {
		desc          string
		params        CountParams
		noiseKind     noise.Kind
		partitionType reflect.Type
		wantErr       bool
	}{
		{
			desc: "valid parameters w/o public partitions",
			params: CountParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MaxValue:                 1,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       false,
		},
		{
			desc: "PartitionSelectionParams.MaxPartitionsContributed set",
			params: CountParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5, MaxPartitionsContributed: 1},
				MaxPartitionsContributed: 1,
				MaxValue:                 1,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "valid parameters w/ Gaussian noise w/o public partitions",
			params: CountParams{
				AggregationEpsilon:       1.0,
				AggregationDelta:         1e-5,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MaxValue:                 1,
			},
			noiseKind:     noise.GaussianNoise,
			partitionType: nil,
			wantErr:       false,
		},
		{
			desc: "zero aggregationDelta w/ Gaussian noise w/o public partitions",
			params: CountParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MaxValue:                 1,
			},
			noiseKind:     noise.GaussianNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "valid parameters w/ public partitions",
			params: CountParams{
				AggregationEpsilon:       1.0,
				MaxPartitionsContributed: 1,
				MaxValue:                 1,
				PublicPartitions:         []int{0},
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(0),
			wantErr:       false,
		},
		{
			desc: "non-zero aggregationDelta w/ laplace noise",
			params: CountParams{
				AggregationEpsilon:       1.0,
				AggregationDelta:         1e-5,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MaxValue:                 1,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(0),
			wantErr:       true,
		},
		{
			desc: "negative aggregationEpsilon",
			params: CountParams{
				AggregationEpsilon:       -1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MaxValue:                 1,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "negative partitionSelectionEpsilon",
			params: CountParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: -1.0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MaxValue:                 1,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "zero partitionSelectionDelta w/o public partitions",
			params: CountParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 0},
				MaxPartitionsContributed: 1,
				MaxValue:                 1,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "zero partitionSelectionEpsilon w/o public partitions",
			params: CountParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MaxValue:                 1,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "wrong partition type w/ public partitions as beam.PCollection",
			params: CountParams{
				AggregationEpsilon:       1.0,
				MaxValue:                 1,
				MaxPartitionsContributed: 1,
				PublicPartitions:         partitions,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
		{
			desc: "wrong partition type w/ public partitions as slice",
			params: CountParams{
				AggregationEpsilon:       1.0,
				MaxValue:                 1,
				MaxPartitionsContributed: 1,
				PublicPartitions:         []int{0},
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
		{
			desc: "wrong partition type w/ public partitions as array",
			params: CountParams{
				AggregationEpsilon:       1.0,
				MaxValue:                 1,
				MaxPartitionsContributed: 1,
				PublicPartitions:         [1]int{0},
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
		{
			desc: "public partitions as something other than beam.PCollection, slice or array",
			params: CountParams{
				AggregationEpsilon:       1.0,
				MaxValue:                 1,
				MaxPartitionsContributed: 1,
				PublicPartitions:         "",
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
		{
			desc: "unset MaxPartitionsContributed",
			params: CountParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxValue:                 1,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "negative max value",
			params: CountParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MaxValue:                 -1,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
	} {
		if err := checkCountParams(tc.params, tc.noiseKind, tc.partitionType); (err != nil) != tc.wantErr {
			t.Errorf("With %s, got=%v error, wantErr=%t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestCountPreThresholding(t *testing.T) {
	// In this test, we set pre-threshold to 10, per-partition l1 sensitivity to 2, and:
	// - value 0 is associated with 9 privacy units, so it should be thresholded;
	// - value 1 is associated with 10 privacy units appearing twice each, so each of
	//   them should be counted twice;
	// Each privacy unit contributes to at most 1 partition.
	pairs := testutils.ConcatenatePairs(
		testutils.MakePairsWithFixedVStartingFromKey(0, 9, 0),
		testutils.MakePairsWithFixedVStartingFromKey(10, 10, 1),
		testutils.MakePairsWithFixedVStartingFromKey(10, 10, 1),
	)
	result := []testutils.PairII64{
		// Partition 0 is dropped due to the pre-threshold.
		{1, 20}, // 10*2
	}
	p, s, col, want := ptest.CreateList2(pairs, result)
	col = beam.ParDo(s, testutils.PairToKV, col)

	// ε=10⁹, δ≈1 and l0Sensitivity=2 means partitions meeting the preThreshold should be kept.
	// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
	// we can have each partition fail with 10⁻²³ probability (k=23).
	epsilon, delta, k, l1Sensitivity := 1e9, dpagg.LargestRepresentableDelta, 23.0, 2.0
	preThreshold := int64(10)
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
			PreThreshold:              preThreshold}))
	got := Count(s, pcol, CountParams{MaxValue: int64(l1Sensitivity), MaxPartitionsContributed: 1, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, testutils.PairII64ToKV, want)
	testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestCountPreThresholding: Count(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

func TestCountAllowNegativeOutputs(t *testing.T) {
	// We use public partitions to noise zeros. For a larger number of
	// public partitions, we expect that at least one noised zero is
	// negative (in case AllowNegativeOutputs is true) or that all are
	// non-negative (in case AllowNegativeOutputs is false).
	for _, tc := range []struct {
		allowNegativeOutputs bool
	}{
		{true},
		{false},
	} {
		privacySpec, err := NewPrivacySpec(PrivacySpecParams{
			AggregationEpsilon: 0.1,
		})
		if err != nil {
			t.Fatalf("Test setup failed: %v", err)
		}
		p, s, col := ptest.CreateList([]testutils.PairII{})
		col = beam.ParDo(s, testutils.PairToKV, col)
		pcol := MakePrivate(s, col, privacySpec)
		var emptyPartitions []int
		for i := 0; i < 10000; i++ {
			emptyPartitions = append(emptyPartitions, i)
		}

		countRes := Count(s, pcol,
			CountParams{
				MaxValue:                 1,
				MaxPartitionsContributed: 1,
				NoiseKind:                LaplaceNoise{},
				PublicPartitions:         emptyPartitions,
				AllowNegativeOutputs:     tc.allowNegativeOutputs,
			})

		values := beam.DropKey(s, countRes)
		if tc.allowNegativeOutputs {
			testutils.CheckAtLeastOneValueNegativeInt64(s, values)
		} else {
			beam.ParDo0(s, testutils.CheckNoNegativeValuesInt64, values)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestCountAllowNegativeOutputs: error with allowNegativeOutputs = %v: %v", tc.allowNegativeOutputs, err)
		}
	}
}
