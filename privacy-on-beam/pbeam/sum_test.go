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

	"github.com/google/differential-privacy/go/v3/dpagg"
	"github.com/google/differential-privacy/go/v3/noise"
	"github.com/google/differential-privacy/privacy-on-beam/v3/pbeam/testutils"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/core/typex"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/testing/ptest"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/stats"
)

func init() {
	beam.RegisterFunction(checkAllValuesNegativeInt64Fn)
}

// Checks that SumPerKey returns a correct answer with int values. The logic
// mirrors TestDistinctPrivacyIDNoNoise, without duplicates.
func TestSumPerKeyNoNoiseInt(t *testing.T) {
	triples := testutils.ConcatenateTriplesWithIntValue(
		testutils.MakeSampleTripleWithIntValue(7, 0),
		testutils.MakeSampleTripleWithIntValue(31, 1),
		testutils.MakeSampleTripleWithIntValue(99, 2))
	result := []testutils.PairII64{
		// The sum for partition 0 is 7, which is below the threshold of 31: so it should be dropped.
		{1, 31},
		{2, 99},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l0Sensitivity=3 gives a threshold of ≈31.
	// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
	// To see the logic and the math behind flakiness and tolerance calculation,
	// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf.
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-200, 24.0, 3.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 3, MinValue: 0.0, MaxValue: 1, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, testutils.PairII64ToKV, want)
	testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyNoNoiseInt: SumPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

// Checks that SumPerKey with partitions returns a correct answer with int values.
func TestSumPerKeyWithPartitionsNoNoiseInt(t *testing.T) {
	for _, tc := range []struct {
		minValue        float64
		maxValue        float64
		lInfSensitivity float64
		inMemory        bool
	}{
		{
			minValue:        1.0,
			maxValue:        3.0,
			lInfSensitivity: 3.0,
			inMemory:        false,
		},
		{
			minValue:        1.0,
			maxValue:        3.0,
			lInfSensitivity: 3.0,
			inMemory:        true,
		},
		{
			minValue:        0.0,
			maxValue:        2.0,
			lInfSensitivity: 2.0,
			inMemory:        false,
		},
		{
			minValue:        0.0,
			maxValue:        2.0,
			lInfSensitivity: 2.0,
			inMemory:        true,
		},
		{
			minValue:        -10.0,
			maxValue:        10.0,
			lInfSensitivity: 10.0,
			inMemory:        false,
		},
		{
			minValue:        -10.0,
			maxValue:        10.0,
			lInfSensitivity: 10.0,
			inMemory:        true,
		},
	} {
		// ID:1 contributes to 8 partitions, only 3 of which are public partitions. So none
		// should be dropped with maxPartitionsContributed=3.
		// Tests that cross-partition contribution bounding happens after non-public partitions are dropped.
		triples := testutils.ConcatenateTriplesWithIntValue(
			testutils.MakeSampleTripleWithIntValue(7, 0),
			testutils.MakeSampleTripleWithIntValue(58, 1),
			testutils.MakeSampleTripleWithIntValue(99, 2),
			testutils.MakeSampleTripleWithIntValue(1, 5),
			testutils.MakeSampleTripleWithIntValue(1, 6),
			testutils.MakeSampleTripleWithIntValue(1, 7),
			testutils.MakeSampleTripleWithIntValue(1, 8),
			testutils.MakeSampleTripleWithIntValue(1, 9))

		publicPartitionsSlice := []int{0, 2, 5, 10, 11}
		// Keep partitions 0, 2 and 5.
		// drop partition 6 to 9.
		// Add partitions 10 and 11.
		result := []testutils.PairII64{
			{0, 7},
			{2, 99},
			{5, 1},
			{10, 0},
			{11, 0},
		}

		p, s, col, want := ptest.CreateList2(triples, result)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		// We have ε=50, δ=0, and l1Sensitivity=3*lInfSensitivity, to scale the noise with different MinValues and MaxValues.
		epsilon, k, l1Sensitivity := 50.0, 25.0, 3.0*tc.lInfSensitivity
		pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))

		pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
		sumParams := SumParams{MaxPartitionsContributed: 3, MinValue: tc.minValue, MaxValue: tc.maxValue, NoiseKind: LaplaceNoise{}, PublicPartitions: publicPartitions}
		got := SumPerKey(s, pcol, sumParams)
		want = beam.ParDo(s, testutils.PairII64ToKV, want)
		testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestSumPerKeyWithPartitionsNoNoiseInt test case=+%v: SumPerKey(%v) = %v, expected %v: %v", tc, col, got, want, err)
		}
	}
}

// Checks that SumPerKey works correctly for negative bounds and negative values with int values.
func TestSumPerKeyNegativeBoundsInt(t *testing.T) {
	triples := testutils.ConcatenateTriplesWithIntValue(
		testutils.MakeTripleWithIntValue(21, 1, -1), // should be clamped down to -2
		testutils.MakeTripleWithIntValue(50, 2, -4)) // should be clamped up to -3
	result := []testutils.PairII64{
		{1, -42},
		{2, -150},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l0Sensitivity=2 gives a threshold of ≈21.
	// We have 2 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-200, 24.0, 6.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 2, MinValue: -3, MaxValue: -2, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, testutils.PairII64ToKV, want)
	testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyNegativeBoundsInt: SumPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

// Checks that SumPerKey with partitions works correctly for negative bounds and negative values with int values.
func TestSumPerKeyWithPartitionsNegativeBoundsInt(t *testing.T) {
	// We have two test cases, one for public partitions as a PCollection and one for public partitions as a slice (i.e., in-memory).
	for _, tc := range []struct {
		inMemory bool
	}{
		{true},
		{false},
	} {
		triples := testutils.ConcatenateTriplesWithIntValue(
			testutils.MakeTripleWithIntValue(21, 1, -1), // should be clamped down to -2
			testutils.MakeTripleWithIntValue(50, 2, -4)) // should be clamped up to -3
		result := []testutils.PairII64{
			{1, -42},
			{2, -150},
		}

		p, s, col, want := ptest.CreateList2(triples, result)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

		publicPartitionsSlice := []int{1, 2}
		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}
		// We have ε=50, δ=0 and l1Sensitivity=6.
		// We have 2 partitions. So, to get an overall flakiness of 10⁻²³,
		// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
		epsilon, k, l1Sensitivity := 50.0, 24.0, 6.0
		pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
		pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
		sumParams := SumParams{MaxPartitionsContributed: 2, MinValue: -3, MaxValue: -2, NoiseKind: LaplaceNoise{}, PublicPartitions: publicPartitions}
		got := SumPerKey(s, pcol, sumParams)
		want = beam.ParDo(s, testutils.PairII64ToKV, want)
		testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestSumPerKeyWithPartitionsNegativeBoundsInt in-memory=%t: SumPerKey(%v) = %v, expected %v: %v", tc.inMemory, col, got, want, err)
		}
	}
}

// Checks that SumPerKey returns a correct answer with float values. The logic
// mirrors TestDistinctPrivacyIDNoNoise, without duplicates.
func TestSumPerKeyNoNoiseFloat(t *testing.T) {
	triples := testutils.ConcatenateTriplesWithFloatValue(
		testutils.MakeSampleTripleWithFloatValue(7, 0),
		testutils.MakeSampleTripleWithFloatValue(31, 1),
		testutils.MakeSampleTripleWithFloatValue(99, 2))
	result := []testutils.PairIF64{
		// Only 7 privacy units are associated with value 0: should be thresholded.
		{1, 31},
		{2, 99},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l0Sensitivity=3 gives a threshold of ≈31.
	// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-200, 24.0, 3.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 3, MinValue: 0.0, MaxValue: 1.0, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, testutils.PairIF64ToKV, want)
	testutils.ApproxEqualsKVFloat64(t, s, got, want, testutils.LaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyNoNoiseFloat: SumPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

// Checks that SumPerKey with partitions returns a correct answer with float values.
func TestSumPerKeyWithPartitionsNoNoiseFloat(t *testing.T) {
	for _, tc := range []struct {
		minValue        float64
		maxValue        float64
		lInfSensitivity float64
		inMemory        bool
	}{
		{
			minValue:        0.0,
			maxValue:        1.0,
			lInfSensitivity: 1.0,
			inMemory:        false,
		},
		{
			minValue:        0.0,
			maxValue:        1.0,
			lInfSensitivity: 1.0,
			inMemory:        true,
		},
		{
			minValue:        3.0,
			maxValue:        10.0,
			lInfSensitivity: 10.0,
			inMemory:        false,
		},
		{
			minValue:        3.0,
			maxValue:        10.0,
			lInfSensitivity: 10.0,
			inMemory:        true,
		},
		{
			minValue:        -50.0,
			maxValue:        50.0,
			lInfSensitivity: 50.0,
			inMemory:        false,
		},
		{
			minValue:        -50.0,
			maxValue:        50.0,
			lInfSensitivity: 50.0,
			inMemory:        true,
		},
	} {
		triples := testutils.ConcatenateTriplesWithFloatValue(
			testutils.MakeSampleTripleWithFloatValue(7, 0),
			testutils.MakeSampleTripleWithFloatValue(58, 1),
			testutils.MakeSampleTripleWithFloatValue(99, 2))
		for i := 5; i < 10; i++ {
			triples = append(triples, testutils.MakeSampleTripleWithFloatValue(1, i)...)
		}
		publicPartitionsSlice := []int{0, 3, 5}
		// Keep partitions 0, 3, and 5.
		// Drop other partitions up to 10.
		result := []testutils.PairIF64{
			{0, 7},
			{3, 0},
			{5, 1},
		}

		p, s, col, want := ptest.CreateList2(triples, result)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		// We have ε=50, δ=0 and l1Sensitivity=3*tc.lInfSensitivity.
		// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
		// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
		epsilon, k, l1Sensitivity := 50.0, 24.0, 3.0*tc.lInfSensitivity
		pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
		pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		sumParams := SumParams{MaxPartitionsContributed: 3, MinValue: tc.minValue, MaxValue: tc.maxValue, NoiseKind: LaplaceNoise{}, PublicPartitions: publicPartitions}
		got := SumPerKey(s, pcol, sumParams)
		want = beam.ParDo(s, testutils.PairIF64ToKV, want)
		testutils.ApproxEqualsKVFloat64(t, s, got, want, testutils.LaplaceTolerance(k, l1Sensitivity, epsilon))
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestSumPerKeyWithPartitionsNoNoiseFloat test case=%+v: SumPerKey(%v) = %v, expected %v: %v", tc, col, got, want, err)
		}
	}
}

// Checks that SumPerKey works correctly for negative bounds and negative values with float values.
func TestSumPerKeyNegativeBoundsFloat(t *testing.T) {
	triples := testutils.ConcatenateTriplesWithFloatValue(
		testutils.MakeTripleWithFloatValue(21, 1, -1.0), // should be clamped down to -2.0
		testutils.MakeTripleWithFloatValue(50, 2, -4.0)) // should be clamped up to -3.0
	result := []testutils.PairIF64{
		{1, -42.0},
		{2, -150.0},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l0Sensitivity=2 gives a threshold of ≈21.
	// We have 2 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-200, 24.0, 6.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 2, MinValue: -3.0, MaxValue: -2.0, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, testutils.PairIF64ToKV, want)
	testutils.ApproxEqualsKVFloat64(t, s, got, want, testutils.LaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyNegativeBoundsFloat: SumPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

// Checks that SumPerKey with partitions works correctly for negative bounds and negative values with float values.
func TestSumPerKeyWithPartitionsNegativeBoundsFloat(t *testing.T) {
	// We have two test cases, one for public partitions as a PCollection and one for public partitions as a slice (i.e., in-memory).
	for _, tc := range []struct {
		inMemory bool
	}{
		{true},
		{false},
	} {
		triples := testutils.ConcatenateTriplesWithFloatValue(
			testutils.MakeTripleWithFloatValue(21, 1, -1.0), // should be clamped down to -2.0
			testutils.MakeTripleWithFloatValue(50, 2, -4.0)) // should be clamped up to -3.0
		result := []testutils.PairIF64{
			{1, -42.0},
			{2, -150.0},
		}

		p, s, col, want := ptest.CreateList2(triples, result)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		publicPartitionsSlice := []int{1, 2}
		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		// We have ε=50, δ=0 and l1Sensitivity=6.
		// We have 2 partitions. So, to get an overall flakiness of 10⁻²³,
		// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
		epsilon, k, l1Sensitivity := 50.0, 24.0, 6.0
		pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
		pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		sumParams := SumParams{MaxPartitionsContributed: 2, MinValue: -3.0, MaxValue: -2.0, NoiseKind: LaplaceNoise{}, PublicPartitions: publicPartitions}
		got := SumPerKey(s, pcol, sumParams)
		want = beam.ParDo(s, testutils.PairIF64ToKV, want)
		testutils.ApproxEqualsKVFloat64(t, s, got, want, testutils.LaplaceTolerance(k, l1Sensitivity, epsilon))
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestSumPerKeyWithPartitionsNegativeBoundsFloat in-memory=%t: SumPerKey(%v) = %v, expected %v: %v", tc.inMemory, col, got, want, err)
		}
	}
}

// Checks that SumPerKey adds noise to its output with int values. The logic
// mirrors TestDistinctPrivacyIDAddsNoise.
func TestSumPerKeyAddsNoiseInt(t *testing.T) {
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
			partitionSelectionDelta:   0.01,
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
		l0Sensitivity, minValue, maxValue := int64(1), 0.0, 1.0

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

		// triples contains {1,0,1}, {2,0,1}, …, {numIDs,0,1}.
		triples := testutils.MakeSampleTripleWithIntValue(numIDs, 0)
		p, s, col := ptest.CreateList(triples)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

		pcol := MakePrivate(s, col, privacySpec(t,
			PrivacySpecParams{
				AggregationEpsilon:        tc.aggregationEpsilon,
				AggregationDelta:          tc.aggregationDelta,
				PartitionSelectionEpsilon: tc.partitionSelectionEpsilon,
				PartitionSelectionDelta:   tc.partitionSelectionDelta,
			}))
		pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
		got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: l0Sensitivity, MinValue: minValue, MaxValue: maxValue, NoiseKind: tc.noiseKind})
		got = beam.ParDo(s, testutils.KVToPairII64, got)

		testutils.CheckInt64MetricsAreNoisy(s, got, numIDs, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("SumPerKey didn't add any noise with int inputs and %s Noise: %v", tc.name, err)
		}
	}
}

// Checks that SumPerKey with partitions adds noise to its output with int values. The logic
// mirrors TestDistinctPrivacyIDAddsNoise.
func TestSumPerKeyWithPartitionsAddsNoiseInt(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		noiseKind NoiseKind
		epsilon   float64
		delta     float64
		inMemory  bool
	}{
		// Epsilon and delta are not split because partitions are public. All of them are used for the noise.
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
			inMemory:  false,
		},
		{
			desc:      "as PCollection w/ Laplace",
			noiseKind: LaplaceNoise{},
			epsilon:   1e-15,
			delta:     0, // It is 0 because partitions are public and we are using Laplace noise.
			inMemory:  true,
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
		l0Sensitivity, minValue, maxValue := int64(1), 0.0, 1.0
		numIDs := 10

		// triples contains {1,0,1}, {2,0,1}, …, {10,0,1}.
		triples := testutils.MakeSampleTripleWithIntValue(numIDs, 0)

		p, s, col := ptest.CreateList(triples)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

		publicPartitionsSlice := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		pcol := MakePrivate(s, col, privacySpec(t,
			PrivacySpecParams{
				AggregationEpsilon: tc.epsilon,
				AggregationDelta:   tc.delta,
			}))
		pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
		sumParams := SumParams{MaxPartitionsContributed: l0Sensitivity, MinValue: minValue, MaxValue: maxValue, NoiseKind: tc.noiseKind, PublicPartitions: publicPartitions}
		got := SumPerKey(s, pcol, sumParams)
		got = beam.ParDo(s, testutils.KVToPairII64, got)

		testutils.CheckInt64MetricsAreNoisy(s, got, numIDs, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("SumPerKey with public partitions %s didn't add any noise with int inputs: %v", tc.desc, err)
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
		aggregationEpsilon        float64
		aggregationDelta          float64
		partitionSelectionEpsilon float64
		partitionSelectionDelta   float64
	}{
		{
			name:                      "Gaussian",
			noiseKind:                 GaussianNoise{},
			aggregationEpsilon:        1,
			aggregationDelta:          0.005,
			partitionSelectionEpsilon: 1,
			partitionSelectionDelta:   0.005,
		},
		{
			name:                      "Laplace",
			noiseKind:                 LaplaceNoise{},
			aggregationEpsilon:        0.1,
			partitionSelectionEpsilon: 0.1,
			partitionSelectionDelta:   0.01,
		},
	} {
		// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
		// we need to have each partition pass with 1-10⁻²³ probability (k=23).
		k := 23.0
		l0Sensitivity, lInfSensitivity := 1.0, 1.0
		l1Sensitivity := l0Sensitivity * lInfSensitivity
		tolerance := testutils.ComplementaryLaplaceTolerance(k, l1Sensitivity, tc.aggregationEpsilon)
		if tc.noiseKind == gaussianNoise {
			tolerance = testutils.ComplementaryGaussianTolerance(k, l0Sensitivity, lInfSensitivity, tc.aggregationEpsilon, tc.aggregationDelta)
		}

		// Compute the number of IDs needed to keep the partition.
		sp, err := dpagg.NewPreAggSelectPartition(
			&dpagg.PreAggSelectPartitionOptions{
				Epsilon:                  tc.partitionSelectionEpsilon,
				Delta:                    tc.partitionSelectionDelta,
				MaxPartitionsContributed: 1,
			})
		if err != nil {
			t.Fatalf("Couldn't initialize PreAggSelectPartition necessary to compute the number of IDs needed: %v", err)
		}
		numIDs, err := sp.GetHardThreshold()
		if err != nil {
			t.Fatalf("Couldn't compute hard threshold: %v", err)
		}

		// triples contains {1,0,1}, {2,0,1}, …, {numIDs,0,1}.
		triples := testutils.MakeSampleTripleWithFloatValue(numIDs, 0)
		p, s, col := ptest.CreateList(triples)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		pcol := MakePrivate(s, col, privacySpec(t,
			PrivacySpecParams{
				AggregationEpsilon:        tc.aggregationEpsilon,
				AggregationDelta:          tc.aggregationDelta,
				PartitionSelectionEpsilon: tc.partitionSelectionEpsilon,
				PartitionSelectionDelta:   tc.partitionSelectionDelta,
			}))
		pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 1, MinValue: 0.0, MaxValue: 1.0, NoiseKind: tc.noiseKind})
		got = beam.ParDo(s, testutils.KVToPairIF64, got)

		testutils.CheckFloat64MetricsAreNoisy(s, got, float64(numIDs), tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("SumPerKey didn't add any noise with float inputs and %s Noise: %v", tc.name, err)
		}
	}
}

// Checks that SumPerKey bounds cross-partition contributions correctly with int values.
// The logic mirrors TestCountCrossPartitionContributionBounding.
func TestSumPerKeyCrossPartitionContributionBoundingInt(t *testing.T) {
	// triples contains {1,0,1}, {2,0,1}, …, {50,0,1}, {1,1,1}, …, {50,1,1}, {1,2,1}, …, {50,9,1}.
	var triples []testutils.TripleWithIntValue
	for i := 0; i < 10; i++ {
		triples = append(triples, testutils.MakeSampleTripleWithIntValue(50, i)...)
	}
	result := []testutils.PairII64{
		{0, 150},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

	// ε=50, δ=0.01 and l0Sensitivity=3 gives a threshold of 3.
	// We have 10 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁴ probability (24).
	epsilon, delta, k, l1Sensitivity := 50.0, 0.01, 24.0, 3.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 3, MinValue: 0, MaxValue: 1, NoiseKind: LaplaceNoise{}})
	// With a max contribution of 3, 70% of the data should have be
	// dropped. The sum of all elements must then be 150.
	counts := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, counts)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
	want = beam.ParDo(s, testutils.PairII64ToKV, want)
	testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyCrossPartitionContributionBoundingInt: SumPerKey(%v) = %v, expected elements to sum to 150: %v", col, got, err)
	}
}

// Checks that SumPerKey with partitions bounds cross-partition contributions correctly with int values.
func TestSumPerKeyWithPartitionsCrossPartitionContributionBoundingInt(t *testing.T) {
	// We have two test cases, one for public partitions as a PCollection and one for public partitions as a slice (i.e., in-memory).
	for _, tc := range []struct {
		inMemory bool
	}{
		{true},
		{false},
	} {
		// triples contains {1,0,1}, {2,0,1}, …, {50,0,1}, {1,1,1}, …, {50,1,1}, {1,2,1}, …, {50,9,1}.
		var triples []testutils.TripleWithIntValue
		for i := 0; i < 10; i++ {
			triples = append(triples, testutils.MakeSampleTripleWithIntValue(50, i)...)
		}
		result := []testutils.PairII64{
			{0, 150},
		}

		p, s, col, want := ptest.CreateList2(triples, result)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

		publicPartitionsSlice := []int{0, 1, 2, 3, 4}
		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		// We have ε=50, δ=0.0 and l1Sensitivity=3.
		// We have 5 partitions. So, to get an overall flakiness of 10⁻²³,
		// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
		epsilon, k, l1Sensitivity := 50.0, 24.0, 3.0
		pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
		pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
		sumParams := SumParams{MaxPartitionsContributed: 3, MinValue: 0, MaxValue: 1, NoiseKind: LaplaceNoise{}, PublicPartitions: publicPartitions}
		got := SumPerKey(s, pcol, sumParams)
		// With a max contribution of 3, all of the data going to three partitions
		// should be kept. The sum of all elements must then be 150.
		counts := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, counts)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.PairII64ToKV, want)
		testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestSumPerKeyWithPartitionsCrossPartitionContributionBoundingInt in-memory=%t: SumPerKey(%v) = %v, expected elements to sum to 150: %v", tc.inMemory, col, got, err)
		}
	}
}

// Checks that SumPerKey bounds cross-partition contributions correctly with float values.
// The logic mirrors TestCountCrossPartitionContributionBounding.
func TestSumPerKeyCrossPartitionContributionBoundingFloat(t *testing.T) {
	// triples contains {1,0,1.0}, {2,0,1.0}, …, {50,0,1.0}, {1,1,1.0}, …, {50,1,1.0}, {1,2,1.0}, …, {50,9,1.0}.
	var triples []testutils.TripleWithFloatValue
	for i := 0; i < 10; i++ {
		triples = append(triples, testutils.MakeSampleTripleWithFloatValue(50, i)...)
	}
	result := []testutils.PairIF64{
		{0, 150.0},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

	// ε=50, δ=0.01 and l0Sensitivity=3 gives a threshold of 3.
	// We have 10 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
	epsilon, delta, k, l1Sensitivity := 50.0, 0.01, 24.0, 3.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 3, MinValue: 0.0, MaxValue: 1.0, NoiseKind: LaplaceNoise{}})
	// With a max contribution of 3, 70% of the data should have be
	// dropped. The sum of all elements must then be 150.
	counts := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, counts)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
	want = beam.ParDo(s, testutils.PairIF64ToKV, want)
	testutils.ApproxEqualsKVFloat64(t, s, got, want, testutils.LaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyCrossPartitionContributionBoundingFloat: SumPerKey(%v) = %v, expected elements to sum to 150.0: %v", col, got, err)
	}
}

// Checks that SumPerKey with partitions bounds per-user contributions correctly with float values.
// The logic mirrors TestCountCrossPartitionContributionBounding.
func TestSumPerKeyWithPartitionsCrossPartitionContributionBoundingFloat(t *testing.T) {
	// We have two test cases, one for public partitions as a PCollection and one for public partitions as a slice (i.e., in-memory).
	for _, tc := range []struct {
		inMemory bool
	}{
		{true},
		{false},
	} {
		// triples contains {1,0,1.0}, {2,0,1.0}, …, {50,0,1.0}, {1,1,1.0}, …, {50,1,1.0}, {1,2,1.0}, …, {50,9,1.0}.
		var triples []testutils.TripleWithFloatValue
		for i := 0; i < 10; i++ {
			triples = append(triples, testutils.MakeSampleTripleWithFloatValue(50, i)...)
		}
		result := []testutils.PairIF64{
			{0, 150.0},
		}

		p, s, col, want := ptest.CreateList2(triples, result)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		publicPartitionsSlice := []int{0, 1, 2, 3, 4}
		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		// We have ε=50, δ=0.0 and l1Sensitivity=3.
		// We have 5 partitions. So, to get an overall flakiness of 10⁻²³,
		// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
		epsilon, k, l1Sensitivity := 50.0, 24.0, 3.0
		pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
		pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		sumParams := SumParams{MaxPartitionsContributed: 3, MinValue: 0.0, MaxValue: 1.0, NoiseKind: LaplaceNoise{}, PublicPartitions: publicPartitions}
		got := SumPerKey(s, pcol, sumParams)
		// With a max contribution of 3, all of the data for three partitions should be kept.
		// The sum of all elements must then be 150.
		counts := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, counts)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.PairIF64ToKV, want)
		testutils.ApproxEqualsKVFloat64(t, s, got, want, testutils.LaplaceTolerance(k, l1Sensitivity, epsilon))
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestSumPerKeyWithPartitionsCrossPartitionContributionBoundingFloat in-memory=%t: SumPerKey(%v) = %v, expected elements to sum to 150.0: %v", tc.inMemory, col, got, err)
		}
	}
}

// Checks that SumPerKey does per-partition contribution bounding correctly for ints.
func TestSumPerKeyPerPartitionContributionBoundingInt(t *testing.T) {
	var triples []testutils.TripleWithIntValue
	for id := 1; id <= 50; id++ {
		triples = append(triples, testutils.TripleWithIntValue{id, 0, 1}) // partition 0 is associated with 50 times 1
		triples = append(triples, testutils.TripleWithIntValue{id, 1, 4}) // partition 1 is associated with 50 times 4
		// Additional values that should not influence the clamping
		triples = append(triples, testutils.TripleWithIntValue{id, 0, -17}) // should clamp to lower bound
		triples = append(triples, testutils.TripleWithIntValue{id, 1, 42})  // should clamp to upper bound
	}
	result := []testutils.PairII64{
		{0, 100}, // each aggregated record in partition 0 must be clamped to 2
		{1, 150}, // each aggregated record in partition 1 must be clamped to 3
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

	// ε=60, δ=0.01 and l0Sensitivity=2 gives a threshold of ≈2.
	// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
	epsilon, delta, k, l1Sensitivity := 60.0, 0.01, 24.0, 6.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MinValue: 2, MaxValue: 3, MaxPartitionsContributed: 2, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, testutils.PairII64ToKV, want)
	testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyPerPartitionContributionBoundingInt: SumPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

// Checks that SumPerKey does per-partition contribution bounding correctly for floats.
func TestSumPerKeyPerPartitionContributionBoundingFloat(t *testing.T) {
	var triples []testutils.TripleWithFloatValue
	for id := 1; id <= 50; id++ {
		triples = append(triples, testutils.TripleWithFloatValue{id, 0, 1}) // partition 0 is associated with 50 times 1
		triples = append(triples, testutils.TripleWithFloatValue{id, 1, 4}) // partition 1 is associated with 50 times 4
		// Additional values that are outside of range [lower, upper]
		triples = append(triples, testutils.TripleWithFloatValue{id, 0, -17}) // should clamp to lower bound
		triples = append(triples, testutils.TripleWithFloatValue{id, 1, 42})  // should clamp to upper bound
	}
	result := []testutils.PairIF64{
		{0, 100.0}, // each aggregated record in partition 0 must be clamped to 2.0
		{1, 150.0}, // each aggregated record in partition 1 must be clamped to 3.0
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

	// ε=60, δ=0.01 and l0Sensitivity=2 gives a threshold of ≈2.
	// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
	epsilon, delta, k, l1Sensitivity := 60.0, 0.01, 24.0, 6.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	got := SumPerKey(s, pcol, SumParams{MinValue: 2.0, MaxValue: 3.0, MaxPartitionsContributed: 2, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, testutils.PairIF64ToKV, want)
	testutils.ApproxEqualsKVFloat64(t, s, got, want, testutils.LaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyPerPartitionContributionBoundingFloat: SumPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

var sumPartitionSelectionTestCases = []struct {
	name                      string
	noiseKind                 NoiseKind
	aggregationEpsilon        float64
	aggregationDelta          float64
	partitionSelectionEpsilon float64
	partitionSelectionDelta   float64
	numPartitions             int
	entriesPerPartition       int
}{
	{
		name:                      "Gaussian",
		noiseKind:                 GaussianNoise{},
		aggregationEpsilon:        1,
		aggregationDelta:          0.3,
		partitionSelectionEpsilon: 1,
		partitionSelectionDelta:   0.3,
		// entriesPerPartition=1 yields a 30% chance of emitting any particular partition
		// (since δ_emit=0.3).
		entriesPerPartition: 1,
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
		// entriesPerPartition=1 yields a 30% chance of emitting any particular partition
		// (since δ_emit=0.3).
		entriesPerPartition: 1,
		numPartitions:       143,
	},
}

// Checks that SumPerKey applies partition selection for int input values.
func TestSumPartitionSelectionInt(t *testing.T) {
	for _, tc := range sumPartitionSelectionTestCases {
		t.Run(tc.name, func(t *testing.T) {
			// Verify that entriesPerPartition is sensical.
			if tc.entriesPerPartition <= 0 {
				t.Fatalf("Invalid test case: entriesPerPartition must be positive. Got: %d", tc.entriesPerPartition)
			}

			// Build up {ID, Partition, Value} pairs such that for each of the tc.numPartitions partitions,
			// tc.entriesPerPartition privacy units contribute a single value:
			//    {0, 0, 1}, {1, 0, 1}, …, {entriesPerPartition-1, 0, 1}
			//    {entriesPerPartition, 1, 1}, {entriesPerPartition+1, 1, 1}, …, {entriesPerPartition+entriesPerPartition-1, 1, 1}
			//    …
			//    {entriesPerPartition*(numPartitions-1), numPartitions-1, 1}, …, {entriesPerPartition*numPartitions-1, numPartitions-1, 1}
			var (
				triples []testutils.TripleWithIntValue
				kOffset = 0
			)
			for i := 0; i < tc.numPartitions; i++ {
				for j := 0; j < tc.entriesPerPartition; j++ {
					triples = append(triples, testutils.TripleWithIntValue{ID: kOffset + j, Partition: i, Value: 1})
				}
				kOffset += tc.entriesPerPartition
			}
			p, s, col := ptest.CreateList(triples)
			col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

			// Run SumPerKey on triples
			pcol := MakePrivate(s, col, privacySpec(t,
				PrivacySpecParams{
					AggregationEpsilon:        tc.aggregationEpsilon,
					AggregationDelta:          tc.aggregationDelta,
					PartitionSelectionEpsilon: tc.partitionSelectionEpsilon,
					PartitionSelectionDelta:   tc.partitionSelectionDelta,
				}))
			pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
			got := SumPerKey(s, pcol, SumParams{MinValue: 0, MaxValue: 1, NoiseKind: tc.noiseKind, MaxPartitionsContributed: 1})
			got = beam.ParDo(s, testutils.KVToPairII64, got)

			// Validate that partition selection is applied (i.e., some emitted and some dropped).
			testutils.CheckSomePartitionsAreDropped(s, got, tc.numPartitions)
			if err := ptest.Run(p); err != nil {
				t.Errorf("%v", err)
			}
		})
	}
}

// Checks that SumPerKey applies partition selection for float input values.
func TestSumPartitionSelectionFloat(t *testing.T) {
	for _, tc := range sumPartitionSelectionTestCases {
		t.Run(tc.name, func(t *testing.T) {
			// Verify that entriesPerPartition is sensical.
			if tc.entriesPerPartition <= 0 {
				t.Fatalf("Invalid test case: entriesPerPartition must be positive. Got: %d", tc.entriesPerPartition)
			}

			// Build up {ID, Partition, Value} pairs such that for each of the tc.numPartitions partitions,
			// tc.entriesPerPartition privacy units contribute a single value:
			//    {0, 0, 1}, {1, 0, 1}, …, {entriesPerPartition-1, 0, 1}
			//    {entriesPerPartition, 1, 1}, {entriesPerPartition+1, 1, 1}, …, {entriesPerPartition+entriesPerPartition-1, 1, 1}
			//    …
			//    {entriesPerPartition*(numPartitions-1), numPartitions-1, 1}, …, {entriesPerPartition*numPartitions-1, numPartitions-1, 1}
			var (
				triples []testutils.TripleWithFloatValue
				kOffset = 0
			)
			for i := 0; i < tc.numPartitions; i++ {
				for j := 0; j < tc.entriesPerPartition; j++ {
					triples = append(triples, testutils.TripleWithFloatValue{ID: kOffset + j, Partition: i, Value: 1.0})
				}
				kOffset += tc.entriesPerPartition
			}
			p, s, col := ptest.CreateList(triples)
			col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

			// Run SumPerKey on triples
			pcol := MakePrivate(s, col, privacySpec(t,
				PrivacySpecParams{
					AggregationEpsilon:        tc.aggregationEpsilon,
					AggregationDelta:          tc.aggregationDelta,
					PartitionSelectionEpsilon: tc.partitionSelectionEpsilon,
					PartitionSelectionDelta:   tc.partitionSelectionDelta,
				}))
			pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
			got := SumPerKey(s, pcol, SumParams{MinValue: 0.0, MaxValue: 1.0, MaxPartitionsContributed: 1, NoiseKind: tc.noiseKind})
			got = beam.ParDo(s, testutils.KVToPairIF64, got)

			// Validate that partition selection is applied (i.e., some emitted and some dropped).
			testutils.CheckSomePartitionsAreDropped(s, got, tc.numPartitions)
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
		wantConvertFn any
		wantErr       bool
	}{
		{"int", typex.New(reflect.TypeOf(int(0))), convertToInt64Fn, false},
		{"int8", typex.New(reflect.TypeOf(int8(0))), convertToInt64Fn, false},
		{"int16", typex.New(reflect.TypeOf(int16(0))), convertToInt64Fn, false},
		{"int32", typex.New(reflect.TypeOf(int32(0))), convertToInt64Fn, false},
		{"int64", typex.New(reflect.TypeOf(int64(0))), convertToInt64Fn, false},
		{"uint", typex.New(reflect.TypeOf(uint(0))), convertToInt64Fn, false},
		{"uint8", typex.New(reflect.TypeOf(uint8(0))), convertToInt64Fn, false},
		{"uint16", typex.New(reflect.TypeOf(uint16(0))), convertToInt64Fn, false},
		{"uint32", typex.New(reflect.TypeOf(uint32(0))), convertToInt64Fn, false},
		{"uint64", typex.New(reflect.TypeOf(uint64(0))), convertToInt64Fn, false},
		{"float32", typex.New(reflect.TypeOf(float32(0))), convertToFloat64Fn, false},
		{"float64", typex.New(reflect.TypeOf(float64(0))), convertToFloat64Fn, false},
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
		convertFn any
		wantKind  reflect.Kind
		wantErr   bool
	}{
		{"convertToInt64", convertToInt64Fn, reflect.Int64, false},
		{"convertToFloat64", convertToFloat64Fn, reflect.Float64, false},
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
	var triples []testutils.TripleWithFloatValue
	for key := 0; key < 100; key++ {
		triples = append(triples, testutils.TripleWithFloatValue{key, key, 0.01})
	}
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)
	// Using a low epsilon & delta for noise, a high epsilon &delta for partition selection, and a high maxValue
	// here to add a lot of noise while keeping partitions.
	lowEps, lowDel, highEps, highDelta, maxValue := 0.001, 1e-10, 10.0, 0.999, 1e8
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        lowEps,
			AggregationDelta:          lowDel,
			PartitionSelectionEpsilon: highEps,
			PartitionSelectionDelta:   highDelta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	sums := SumPerKey(s, pcol, SumParams{MinValue: 0, MaxValue: maxValue, MaxPartitionsContributed: 1, NoiseKind: GaussianNoise{}})
	values := beam.DropKey(s, sums)
	beam.ParDo0(s, testutils.CheckNoNegativeValuesFloat64, values)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyReturnsNonNegativeFloat64 returned errors: %v", err)
	}
}

// // Expect non-negative results with partitions if MinValue >= 0 for float64 values.
func TestSumPerKeyWithPartitionsReturnsNonNegativeFloat64(t *testing.T) {
	// We have two test cases, one for public partitions as a PCollection and one for public partitions as a slice (i.e., in-memory).
	for _, tc := range []struct {
		inMemory bool
	}{
		{true},
		{false},
	} {
		var triples []testutils.TripleWithFloatValue
		for key := 0; key < 100; key++ {
			triples = append(triples, testutils.TripleWithFloatValue{key, key, 0.01})
		}
		var publicPartitionsSlice []int
		for p := 0; p < 200; p++ {
			publicPartitionsSlice = append(publicPartitionsSlice, p)
		}

		p, s, col := ptest.CreateList(triples)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		// Using a low epsilon & delta and a high maxValue here to add a lot of noise.
		eps, del, maxValue := 0.001, 1e-10, 1e8
		pcol := MakePrivate(s, col, privacySpec(t,
			PrivacySpecParams{
				AggregationEpsilon: eps,
				AggregationDelta:   del,
			}))
		pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		sumParams := SumParams{MinValue: 0, MaxValue: maxValue, MaxPartitionsContributed: 1, NoiseKind: GaussianNoise{}, PublicPartitions: publicPartitions}
		sums := SumPerKey(s, pcol, sumParams)
		values := beam.DropKey(s, sums)
		beam.ParDo0(s, testutils.CheckNoNegativeValuesFloat64, values)
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestSumPerKeyWithPartitionsReturnsNonNegativeFloat64 in-memory=%t returned errors: %v", tc.inMemory, err)
		}
	}
}

// Expect non-negative results if MinValue >= 0 for int64 values.
func TestSumPerKeyReturnsNonNegativeInt64(t *testing.T) {
	var triples []testutils.TripleWithIntValue
	for key := 0; key < 100; key++ {
		triples = append(triples, testutils.TripleWithIntValue{key, key, 1})
	}
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)
	// Using a low epsilon & delta for noise, a high epsilon &delta for partition selection, and a high maxValue
	// here to add a lot of noise while keeping partitions.
	lowEps, lowDel, highEps, highDelta, maxValue := 0.001, 1e-10, 10.0, 0.999, 1e8
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        lowEps,
			AggregationDelta:          lowDel,
			PartitionSelectionEpsilon: highEps,
			PartitionSelectionDelta:   highDelta,
		}))
	pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
	sums := SumPerKey(s, pcol, SumParams{MinValue: 0, MaxValue: maxValue, MaxPartitionsContributed: 1, NoiseKind: GaussianNoise{}})
	values := beam.DropKey(s, sums)
	beam.ParDo0(s, testutils.CheckNoNegativeValuesInt64, values)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyReturnsNonNegativeInt64 returned errors: %v", err)
	}
}

// Expect non-negative results with partitions if MinValue >= 0 for int64 values.
func TestSumPerKeyWithPartitionsReturnsNonNegativeInt64(t *testing.T) {
	// We have two test cases, one for public partitions as a PCollection and one for public partitions as a slice (i.e., in-memory).
	for _, tc := range []struct {
		inMemory bool
	}{
		{true},
		{false},
	} {
		var triples []testutils.TripleWithIntValue
		for key := 0; key < 100; key++ {
			triples = append(triples, testutils.TripleWithIntValue{key, key, 1})
		}
		var publicPartitionsSlice []int
		for p := 0; p < 200; p++ {
			publicPartitionsSlice = append(publicPartitionsSlice, p)
		}

		p, s, col := ptest.CreateList(triples)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		// Using a low epsilon & delta and a high maxValue here to add a lot of noise.
		eps, del, maxValue := 0.001, 1e-10, 1e8
		pcol := MakePrivate(s, col, privacySpec(t,
			PrivacySpecParams{
				AggregationEpsilon: eps,
				AggregationDelta:   del,
			}))
		pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
		sumParams := SumParams{MinValue: 0, MaxValue: maxValue, MaxPartitionsContributed: 1, NoiseKind: GaussianNoise{}, PublicPartitions: publicPartitions}
		sums := SumPerKey(s, pcol, sumParams)
		values := beam.DropKey(s, sums)
		beam.ParDo0(s, testutils.CheckNoNegativeValuesInt64, values)
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestSumPerKeyWithPartitionsReturnsNonNegativeInt64 in-memory=%t returned errors: %v", tc.inMemory, err)
		}
	}
}

// Expect at least one negative value after post-aggregation clamping when
// MinValue < 0 for float64 values.
func TestSumPerKeyNoClampingForNegativeMinValueFloat64(t *testing.T) {
	var triples []testutils.TripleWithFloatValue
	for key := 0; key < 100; key++ {
		triples = append(triples, testutils.TripleWithFloatValue{key, key, 0})
	}
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)
	// Using `typical` privacy parameters with a high delta to keep partitions.
	epsilon, lowDelta, highDelta, minValue, maxValue := 0.1, 1e-10, 0.999, -100.0, 100.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			AggregationDelta:          lowDelta,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   highDelta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	sums := SumPerKey(s, pcol, SumParams{MinValue: minValue, MaxValue: maxValue, MaxPartitionsContributed: 1, NoiseKind: GaussianNoise{}})
	values := beam.DropKey(s, sums)
	mValue := stats.Min(s, values)
	beam.ParDo0(s, testutils.CheckAllValuesNegativeFloat64, mValue)
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
	var triples []testutils.TripleWithIntValue
	for key := 0; key < 100; key++ {
		triples = append(triples, testutils.TripleWithIntValue{key, key, 0})
	}
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)
	// Using `typical` privacy parameters with a high delta to keep partitions.
	epsilon, lowDelta, highDelta, minValue, maxValue := 0.1, 1e-10, 0.999, -100.0, 100.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			AggregationDelta:          lowDelta,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   highDelta,
		}))
	pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
	sums := SumPerKey(s, pcol, SumParams{MinValue: minValue, MaxValue: maxValue, MaxPartitionsContributed: 1, NoiseKind: GaussianNoise{}})
	values := beam.DropKey(s, sums)
	mValue := stats.Min(s, values)
	beam.ParDo0(s, checkAllValuesNegativeInt64Fn, mValue)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyNoClampingForNegativeMinValueInt64 returned errors: %v", err)
	}
}

func TestCheckSumPerKeyParams(t *testing.T) {
	_, _, publicPartitions := ptest.CreateList([]int{0, 1})
	for _, tc := range []struct {
		desc          string
		params        SumParams
		noiseKind     noise.Kind
		partitionType reflect.Type
		wantErr       bool
	}{

		{
			desc: "valid parameters",
			params: SumParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MinValue:                 -5.0,
				MaxValue:                 5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       false,
		},
		{
			desc: "PartitionSelectionParams.MaxPartitionsContributed set",
			params: SumParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5, MaxPartitionsContributed: 1},
				MaxPartitionsContributed: 1,
				MinValue:                 -5.0,
				MaxValue:                 5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "zero aggregationDelta w/ Gaussian noise",
			params: SumParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MinValue:                 -5.0,
				MaxValue:                 5.0,
			},
			noiseKind:     noise.GaussianNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "negative aggregationEpsilon",
			params: SumParams{
				AggregationEpsilon:       -1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MinValue:                 -5.0,
				MaxValue:                 5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "negative partitionSelectionEpsilon",
			params: SumParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: -1.0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MinValue:                 -5.0,
				MaxValue:                 5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "zero partitionSelectionDelta w/o public partitions",
			params: SumParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 0},
				MaxPartitionsContributed: 1,
				MinValue:                 -5.0,
				MaxValue:                 5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "zero partitionSelectionEpsilon w/o public partitions",
			params: SumParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MinValue:                 -5.0,
				MaxValue:                 5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "MaxPartitionsContributed unset",
			params: SumParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MinValue:                 -5.0,
				MaxValue:                 5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "MaxValue < MinValue",
			params: SumParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MinValue:                 6.0,
				MaxValue:                 5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "MaxValue = MinValue",
			params: SumParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MinValue:                 5.0,
				MaxValue:                 5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       false,
		},
		{
			desc: "non-zero partitionSelectionDelta w/ public partitions",
			params: SumParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 0, Delta: 1e-5},
				MaxPartitionsContributed: 1,
				MinValue:                 -5.0,
				MaxValue:                 5.0,
				PublicPartitions:         publicPartitions,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(0),
			wantErr:       true,
		},
		{
			desc: "non-zero partitionSelectionEpsilon w/ public partitions",
			params: SumParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1.0, Delta: 0},
				MaxPartitionsContributed: 1,
				MinValue:                 -5.0,
				MaxValue:                 5.0,
				PublicPartitions:         publicPartitions,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(0),
			wantErr:       true,
		},
		{
			desc: "wrong partition type w/ public partitions as beam.PCollection",
			params: SumParams{
				AggregationEpsilon:       1.0,
				MaxPartitionsContributed: 1,
				MinValue:                 -5.0,
				MaxValue:                 5.0,
				PublicPartitions:         publicPartitions,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
		{
			desc: "wrong partition type w/ public partitions as slice",
			params: SumParams{
				AggregationEpsilon:       1.0,
				MaxPartitionsContributed: 1,
				MinValue:                 -5.0,
				MaxValue:                 5.0,
				PublicPartitions:         []int{0},
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
		{
			desc: "wrong partition type w/ public partitions as array",
			params: SumParams{
				AggregationEpsilon:       1.0,
				MaxPartitionsContributed: 1,
				MinValue:                 -5.0,
				MaxValue:                 5.0,
				PublicPartitions:         [1]int{0},
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
		{
			desc: "public partitions as something other than beam.PCollection, slice or array",
			params: SumParams{
				AggregationEpsilon:       1.0,
				MaxPartitionsContributed: 1,
				MinValue:                 -5.0,
				MaxValue:                 5.0,
				PublicPartitions:         "",
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
	} {
		if err := checkSumPerKeyParams(tc.params, tc.noiseKind, tc.partitionType); (err != nil) != tc.wantErr {
			t.Errorf("With %s, got=%v, wantErr=%t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestSumPerKeyPreThresholdingInt(t *testing.T) {
	// Arrange
	// ε=10⁹, δ≈1 and l0Sensitivity=1 means partitions meeting the preThreshold should be kept.
	// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²³ probability (k=23).
	epsilon, delta, k, l1Sensitivity := 1e9, dpagg.LargestRepresentableDelta, 23.0, 1.0
	spec := privacySpec(t, PrivacySpecParams{
		AggregationEpsilon:        epsilon,
		PartitionSelectionEpsilon: epsilon,
		PartitionSelectionDelta:   delta,
		PreThreshold:              10,
	})
	triples := testutils.ConcatenateTriplesWithIntValue(
		testutils.MakeSampleTripleWithIntValue(9, 0),
		testutils.MakeTripleWithIntValueStartingFromKey(9, 10, 1, 1))
	result := []testutils.PairII64{
		// The privacy ID count for partition 0 is 9, which is below the pre-threshold of 10: so it should be dropped.
		{1, 10},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	want = beam.ParDo(s, testutils.PairII64ToKV, want)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

	pcol := MakePrivate(s, col, spec)
	pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)

	// Act
	got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 1, MinValue: 0, MaxValue: 1, NoiseKind: LaplaceNoise{}})

	// Assert
	testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyPreThresholdingInt: SumPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

func TestSumPerKeyPreThresholdingFloat(t *testing.T) {
	// Arrange
	// ε=10⁹, δ≈1 and l0Sensitivity=1 means partitions meeting the preThreshold should be kept.
	// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²³ probability (k=23).
	epsilon, delta, k, l1Sensitivity := 1e9, dpagg.LargestRepresentableDelta, 23.0, 1.0
	spec := privacySpec(t, PrivacySpecParams{
		AggregationEpsilon:        epsilon,
		PartitionSelectionEpsilon: epsilon,
		PartitionSelectionDelta:   delta,
		PreThreshold:              10,
	})
	triples := testutils.ConcatenateTriplesWithFloatValue(
		testutils.MakeSampleTripleWithFloatValue(9, 0),
		testutils.MakeTripleWithFloatValueStartingFromKey(9, 10, 1, 1))
	result := []testutils.PairIF64{
		// The privacy ID count for partition 0 is 9, which is below the pre-threshold of 10: so it should be dropped.
		{1, 10},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	want = beam.ParDo(s, testutils.PairIF64ToKV, want)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)
	pcol := MakePrivate(s, col, spec)
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)

	// Act
	got := SumPerKey(s, pcol, SumParams{MaxPartitionsContributed: 1, MinValue: 0, MaxValue: 1, NoiseKind: LaplaceNoise{}})

	// Assert
	testutils.ApproxEqualsKVFloat64(t, s, got, want, testutils.LaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestSumPerKeyPreThresholdingFloat: SumPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}
