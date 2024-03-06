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
	"math"
	"reflect"
	"testing"

	"github.com/google/differential-privacy/go/v3/dpagg"
	"github.com/google/differential-privacy/go/v3/noise"
	"github.com/google/differential-privacy/privacy-on-beam/v3/pbeam/testutils"
	testpb "github.com/google/differential-privacy/privacy-on-beam/v3/testdata"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/testing/passert"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/testing/ptest"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/stats"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/protobuf/proto"
)

var (
	ln3 = math.Log(3)
)

func init() {
	beam.RegisterType(reflect.TypeOf((*testpb.TestAnon)(nil)))
	beam.RegisterType(reflect.TypeOf((*checkNothingBelowThresholdFn)(nil)))
}

// Simply checks that running an aggregation step from a PrivatePCollection
// generated from MakeDistinctPrivacyIDFromProto does not fail.
func TestProtoAggregation(t *testing.T) {
	values := []*testpb.TestAnon{
		{Foo: proto.Int64(42), Bar: proto.String("fourty-two")},
		{Foo: proto.Int64(17), Bar: proto.String("seventeen")},
		{Bar: proto.String("zero")},
	}
	p, s, col := ptest.CreateList(values)
	pcol := MakePrivateFromProto(
		s,
		col,
		privacySpec(t,
			PrivacySpecParams{
				AggregationEpsilon:        1,
				PartitionSelectionEpsilon: 1,
				PartitionSelectionDelta:   1e-10,
			}),
		"foo",
	)
	got := DistinctPrivacyID(s, pcol, DistinctPrivacyIDParams{MaxPartitionsContributed: 1, NoiseKind: LaplaceNoise{}})
	// All values are distinct and should be thresholded.
	passert.Empty(s, got)
	if err := ptest.Run(p); err != nil {
		t.Errorf("proto aggregation failed: %v", err)
	}
}

// Checks that DistinctPrivacyID returns a correct answer, in particular that keys
// are correctly counted (without duplicates).
func TestDistinctPrivacyIDNoNoise(t *testing.T) {
	// pairs{privacy_id, partition_key} contain input data belonging to partitions 0, 1, and 2.
	pairs := testutils.ConcatenatePairs(
		testutils.MakePairsWithFixedV(7, 0),
		testutils.MakePairsWithFixedV(52, 1),
		testutils.MakePairsWithFixedV(99, 2),
		testutils.MakePairsWithFixedV(7, 0)) // duplicated values should have no influence.
	result := []testutils.PairII64{
		// Only 7 privacy units are associated with value 0: should be thresholded.
		{1, 52},
		{2, 99},
	}
	p, s, col, want := ptest.CreateList2(pairs, result)
	col = beam.ParDo(s, testutils.PairToKV, col)

	// ε=50, δ=10⁻²⁰⁰ and l1Sensitivity=4 gives a post-aggregation threshold of 38.
	// We have 4 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
	// To see the logic and the math behind flakiness and tolerance calculation,
	// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf.
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-200, 24.0, 4.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:      epsilon,
			PartitionSelectionDelta: delta,
		}))
	got := DistinctPrivacyID(s, pcol, DistinctPrivacyIDParams{MaxPartitionsContributed: 4, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, testutils.PairII64ToKV, want)
	testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestDistinctPrivacyIDNoNoise: DistinctPrivacyID(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

// Checks that DistinctPrivacyID with partitions returns a correct answer, in particular that keys
// are correctly counted (without duplicates).
func TestDistinctPrivacyIDWithPartitionsNoNoise(t *testing.T) {
	// We have two test cases, one for public partitions as a PCollection and one for public partitions as a slice (i.e., in-memory).
	for _, tc := range []struct {
		inMemory bool
	}{
		{true},
		{false},
	} {
		// pairs{privacy_id, partition_key} contain input data belonging to partitions 0, 1, 2 and 3.
		pairs := testutils.ConcatenatePairs(
			testutils.MakePairsWithFixedV(7, 0),
			testutils.MakePairsWithFixedV(52, 1),
			testutils.MakePairsWithFixedV(99, 2),
			testutils.MakePairsWithFixedV(7, 0), // duplicated values should have no influence.
			testutils.MakePairsWithFixedV(20, 3))
		result := []testutils.PairII64{
			// Public partitions include 0, which would otherwise be thresholded.
			{0, 7},
			{1, 52},
			// Drop non-public partition 2.
			{3, 20},
			{4, 0}, // Add public partition 4.
		}

		p, s, col, want := ptest.CreateList2(pairs, result)
		col = beam.ParDo(s, testutils.PairToKV, col)

		publicPartitionsSlice := []int{0, 1, 3, 4}
		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		// We have ε=500, δ=0, and l1Sensitivity=4.
		// We have 4 partitions. So, to get an overall flakiness of 10⁻²³,
		// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
		epsilon, delta, k, l1Sensitivity := 500.0, 0.0, 24.0, 4.0
		pcol := MakePrivate(s, col, privacySpec(t,
			PrivacySpecParams{
				AggregationEpsilon:      epsilon,
				PartitionSelectionDelta: delta,
			}))
		distinctPrivacyIDParams := DistinctPrivacyIDParams{MaxPartitionsContributed: 4, NoiseKind: LaplaceNoise{}, PublicPartitions: publicPartitions}
		got := DistinctPrivacyID(s, pcol, distinctPrivacyIDParams)
		want = beam.ParDo(s, testutils.PairII64ToKV, want)
		testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestDistinctPrivacyIDWithPartitionsNoNoise in-memory=%t: DistinctPrivacyID(%v) = %v, expected %v: %v", tc.inMemory, col, got, want, err)
		}
	}
}

type distinctThresholdTestCase struct {
	name                    string
	noiseKind               NoiseKind
	epsilon                 float64
	aggregationDelta        float64
	partitionSelectionDelta float64
	numPartitions           int
	minAllowedValue         int
}

func computeGaussianThreshold(l0Sensitivity int64, lInfSensitivity, epsilon, noiseDelta, thresholdDelta float64) float64 {
	res, _ := noise.Gaussian().Threshold(l0Sensitivity, lInfSensitivity, epsilon, noiseDelta, thresholdDelta)
	return res
}

func computeLaplaceThreshold(l0Sensitivity int64, lInfSensitivity, epsilon, noiseDelta, thresholdDelta float64) float64 {
	res, _ := noise.Laplace().Threshold(l0Sensitivity, lInfSensitivity, epsilon, noiseDelta, thresholdDelta)
	return res
}

var distinctThresholdTestCases = []distinctThresholdTestCase{
	{
		name:                    "Gaussian",
		noiseKind:               GaussianNoise{},
		epsilon:                 1,
		aggregationDelta:        0.005,
		partitionSelectionDelta: 0.005,
		numPartitions:           25,
		minAllowedValue:         int(computeGaussianThreshold(25, 1, 1, 0.005, 0.005)),
	},
	{
		name:                    "Laplace",
		noiseKind:               LaplaceNoise{},
		epsilon:                 1,
		aggregationDelta:        0,
		partitionSelectionDelta: 0.005,
		numPartitions:           25,
		minAllowedValue:         int(computeLaplaceThreshold(25, 1, 1, 0, 0.01)),
	},
}

type checkNothingBelowThresholdFn struct {
	Threshold int // Exported in order to be usable by Beam.
}

func (fn *checkNothingBelowThresholdFn) ProcessElement(c testutils.PairII64) error {
	if c.Value < int64(fn.Threshold) {
		return fmt.Errorf("found a count of %d<%d for key %d", c.Value, fn.Threshold, c.Key)
	}
	return nil
}

func buildDistinctPrivacyIDThresholdPipeline(t *testing.T, tc distinctThresholdTestCase) (p *beam.Pipeline, s beam.Scope, col beam.PCollection, got beam.PCollection) {
	t.Helper()
	// pairs contains {1,0}, {2,0}, …, {minAllowedValue,0}, {1,1}, …, {minAllowedValue,1}, {1,2}, …, {minAllowedValue,9}.
	var pairs []testutils.PairII
	for i := 0; i < tc.numPartitions; i++ {
		// We add minAllowedValue privacy keys per value to place each of the values
		// right next to the distribution's Threshold.
		pairs = append(pairs, testutils.MakePairsWithFixedV(tc.minAllowedValue, i)...)
	}
	p, s, col = ptest.CreateList(pairs)
	col = beam.ParDo(s, testutils.PairToKV, col)

	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:      tc.epsilon,
			AggregationDelta:        tc.aggregationDelta,
			PartitionSelectionDelta: tc.partitionSelectionDelta,
		}))
	got = DistinctPrivacyID(s, pcol, DistinctPrivacyIDParams{MaxPartitionsContributed: int64(tc.numPartitions), NoiseKind: tc.noiseKind})
	got = beam.ParDo(s, testutils.KVToPairII64, got)
	return p, s, col, got
}

// Checks that DistinctPrivacyID correctly removes partitions under the threshold.
func TestDistinctPrivacyIDThresholdsSmallEntries(t *testing.T) {
	for _, tc := range distinctThresholdTestCases {
		t.Run(tc.name, func(t *testing.T) {
			// Verify that minAllowedValue is sensical.
			if tc.minAllowedValue <= 0 {
				t.Errorf("Invalid test case: minAllowedValue must be positive. Got: %d", tc.minAllowedValue)
			}

			p, s, col, got := buildDistinctPrivacyIDThresholdPipeline(t, tc)
			beam.ParDo0(s, &checkNothingBelowThresholdFn{tc.minAllowedValue}, got)
			if err := ptest.Run(p); err != nil {
				t.Errorf("%s: DistinctPrivacyID(%v) = %v, found an unexpected value below minAllowedValue: %v", tc.name, col, got, err)
			}
		})
	}
}

// Checks that DistinctPrivacyID does not remove all data (partitions above the
// threshold should be maintained).
func TestDistinctPrivacyIDThresholdLeavesSomeEntries(t *testing.T) {
	for _, tc := range distinctThresholdTestCases {
		t.Run(tc.name, func(t *testing.T) {
			// Verify that minAllowedValue is sensical.
			if tc.minAllowedValue <= 0 {
				t.Errorf("Invalid test case: minAllowedValue must be positive. Got: %d", tc.minAllowedValue)
			}

			p, s, col, got := buildDistinctPrivacyIDThresholdPipeline(t, tc)
			passert.Empty(s, got) // We want this to be an error.
			if err := ptest.Run(p); err == nil {
				t.Errorf("%s: DistinctPrivacyID(%v) returned an empty result.", tc.name, col)
			}
		})
	}
}

// Checks that DistinctPrivacyID adds noise to its output.
func TestDistinctPrivacyIDAddsNoise(t *testing.T) {
	for _, tc := range []struct {
		name      string
		noiseKind NoiseKind
		// Differential privacy params used.
		aggregationEpsilon      float64
		aggregationDelta        float64
		partitionSelectionDelta float64
	}{
		{
			name:                    "Gaussian",
			noiseKind:               GaussianNoise{},
			aggregationEpsilon:      2 * 1e-5,
			aggregationDelta:        1e-5,
			partitionSelectionDelta: 1e-5,
		},
		{
			name:                    "Laplace",
			noiseKind:               LaplaceNoise{},
			aggregationEpsilon:      4 * 1e-5,
			partitionSelectionDelta: 0.5,
		},
	} {
		// Because this is an integer aggregation, we can't use the regular complementary
		// tolerance computations. Instead, we do the following:
		//
		// If generated noise is between -0.5 and 0.5, it will be rounded to 0 and the
		// test will fail. For Laplace, this will happen with probability
		//   P ~= Laplace_CDF(0.5) - Laplace_CDF(-0.5).
		// Given that Laplace scale = l1_sensitivity / ε = 10⁵ / 4, P ~= 2.9e-5.
		// For Gaussian, this will happen with probability
		//	 P ~= Gaussian_CDF(0.5) - Gaussian_CDF(-0.5).
		// For given ε=2e-5, δ=1e-5 => sigma = 21824, P ~= 1.8e-5.
		//
		// We want to keep numIDs low (otherwise the tests take a long time) while
		// also keeping P low. This means we can't have a tiny ε & δ.
		tolerance := 0.0
		k := 5.0 // k leads to 1e-5 and both P's are close to 1e-5.
		l0Sensitivity, lInfSensitivity := 1.0, 1.0
		l1Sensitivity := l0Sensitivity * lInfSensitivity
		thresholdTolerance := testutils.LaplaceTolerance(k, l1Sensitivity, tc.aggregationEpsilon)
		numIDs := int(math.Ceil(computeLaplaceThreshold(int64(l0Sensitivity), lInfSensitivity, tc.aggregationEpsilon, tc.aggregationDelta, tc.partitionSelectionDelta) + thresholdTolerance))
		if tc.noiseKind == gaussianNoise {
			thresholdTolerance = testutils.GaussianTolerance(k, l0Sensitivity, lInfSensitivity, tc.aggregationEpsilon, tc.aggregationDelta)
			numIDs = int(math.Ceil(computeGaussianThreshold(int64(l0Sensitivity), lInfSensitivity, tc.aggregationEpsilon, tc.aggregationDelta, tc.partitionSelectionDelta) + thresholdTolerance))
		}
		// pairs{privacy_id, partition_key} contains {1,0}, {2,0}, …, {numIDs,0}.
		pairs := testutils.MakePairsWithFixedV(numIDs, 0)
		p, s, col := ptest.CreateList(pairs)
		col = beam.ParDo(s, testutils.PairToKV, col)

		pcol := MakePrivate(s, col, privacySpec(t,
			PrivacySpecParams{
				AggregationEpsilon:      tc.aggregationEpsilon,
				AggregationDelta:        tc.aggregationDelta,
				PartitionSelectionDelta: tc.partitionSelectionDelta,
			}))
		got := DistinctPrivacyID(s, pcol, DistinctPrivacyIDParams{MaxPartitionsContributed: int64(lInfSensitivity), NoiseKind: tc.noiseKind})
		got = beam.ParDo(s, testutils.KVToPairII64, got)

		testutils.CheckInt64MetricsAreNoisy(s, got, numIDs, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("DistinctPrivacyID didn't add any %s noise: %v", tc.name, err)
		}
	}
}

// Checks that DistinctPrivacyID with partitions adds noise to its output.
func TestDistinctPrivacyIDWithPartitionsAddsNoise(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		noiseKind NoiseKind
		// Differential privacy params used. The test assumes sensitivities of 1.
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
		l0Sensitivity := 1.
		numIDs := 10
		// pairs{privacy_id, partition_key} contains {1,0}, {2,0}, …, {numIDs,0}.
		pairs := testutils.MakePairsWithFixedV(numIDs, 0)
		p, s, col := ptest.CreateList(pairs)
		col = beam.ParDo(s, testutils.PairToKV, col)

		pcol := MakePrivate(s, col, privacySpec(t,
			PrivacySpecParams{
				AggregationEpsilon: tc.epsilon,
				AggregationDelta:   tc.delta,
			}))
		publicPartitionsSlice := []int{0}
		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}
		distinctPrivacyIDParams := DistinctPrivacyIDParams{MaxPartitionsContributed: int64(l0Sensitivity), NoiseKind: tc.noiseKind, PublicPartitions: publicPartitions}
		got := DistinctPrivacyID(s, pcol, distinctPrivacyIDParams)
		got = beam.ParDo(s, testutils.KVToPairII64, got)

		testutils.CheckInt64MetricsAreNoisy(s, got, numIDs, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("DistinctPrivacyID with public partitions %s didn't add any noise: %v", tc.desc, err)
		}
	}
}

// Checks that DistinctPrivacyID bounds cross-partition contributions correctly.
// The logic mirrors TestCountCrossPartitionContributionBounding.
func TestDistinctPrivacyIDCrossPartitionContributionBounding(t *testing.T) {
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

	// ε=50, δ=0.01 and l0Sensitivity=3 gives a post-aggregation threshold of 2.
	// We have 10 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
	epsilon, delta, k, l1Sensitivity := 50.0, 0.01, 24.0, 3.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:      epsilon,
			PartitionSelectionDelta: delta,
		}))
	got := DistinctPrivacyID(s, pcol, DistinctPrivacyIDParams{MaxPartitionsContributed: 3, NoiseKind: LaplaceNoise{}})
	// With a max contribution of 3, 70% of the data should be
	// dropped. The sum of all elements must then be 150.
	counts := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, counts)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
	want = beam.ParDo(s, testutils.PairII64ToKV, want)
	testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestDistinctPrivacyIDCrossPartitionContributionBounding: DistinctPrivacyID(%v) = %v, expected elements to sum to 150: %v", col, got, err)
	}
}

// Checks that DistinctPrivacyID bounds cross-partition contributions correctly.
// The logic mirrors TestCountCrossPartitionContributionBounding.
func TestDistinctPrivacyIDWithPartitionsCrossPartitionContributionBounding(t *testing.T) {
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

		// We have ε=50, δ=0 and l1Sensitivity=3.
		// We have 5 partitions. So, to get an overall flakiness of 10⁻²³,
		// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
		epsilon, delta, k, l1Sensitivity := 50.0, 0.0, 24.0, 3.0
		pcol := MakePrivate(s, col, privacySpec(t,
			PrivacySpecParams{
				AggregationEpsilon: epsilon,
				AggregationDelta:   delta,
			}))
		distinctPrivacyIDParams := DistinctPrivacyIDParams{MaxPartitionsContributed: 3, NoiseKind: LaplaceNoise{}, PublicPartitions: publicPartitions}
		got := DistinctPrivacyID(s, pcol, distinctPrivacyIDParams)
		// With a max contribution of 3, 40% of the public partitions should be dropped.
		// The sum of all elements must then be 150.
		counts := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, counts)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.PairII64ToKV, want)
		testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestDistinctPrivacyIDWithPartitionsCrossPartitionContributionBounding in-memory=%t: DistinctPrivacyID(%v) = %v, expected elements to sum to 150: %v", tc.inMemory, col, got, err)
		}
	}
}

// Check that no negative values are returned from DistinctPrivacyID.
func TestDistinctPrivacyIDReturnsNonNegative(t *testing.T) {
	var pairs []testutils.PairII
	for i := 0; i < 100; i++ {
		pairs = append(pairs, testutils.PairII{i, i})
	}
	p, s, col := ptest.CreateList(pairs)
	col = beam.ParDo(s, testutils.PairToKV, col)
	// Using a low epsilon adds a lot of noise and using a high delta keeps
	// many partitions.
	epsilon, delta := 0.001, 0.999
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:      epsilon,
			PartitionSelectionDelta: delta,
		}))
	counts := DistinctPrivacyID(s, pcol, DistinctPrivacyIDParams{MaxPartitionsContributed: 1, NoiseKind: LaplaceNoise{}})
	values := beam.DropKey(s, counts)
	// Check if we have negative elements.
	beam.ParDo0(s, testutils.CheckNoNegativeValuesInt64, values)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestCountReturnsNonNegative returned errors: %v", err)
	}
}

// Check that no negative values are returned from DistinctPrivacyID with partitions.
func TestDistinctPrivacyIDWithPartitionsReturnsNonNegative(t *testing.T) {
	// We have two test cases, one for public partitions as a PCollection and one for public partitions as a slice (i.e., in-memory).
	for _, tc := range []struct {
		inMemory bool
	}{
		{true},
		{false},
	} {
		var pairs []testutils.PairII
		for i := 0; i < 100; i++ {
			pairs = append(pairs, testutils.PairII{i, i})
		}
		p, s, col := ptest.CreateList(pairs)
		col = beam.ParDo(s, testutils.PairToKV, col)

		var publicPartitionsSlice []int
		for i := 0; i < 200; i++ {
			publicPartitionsSlice = append(publicPartitionsSlice, i)
		}
		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		// Using a low epsilon adds a lot of noise.
		epsilon := 0.001
		pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
		distinctPrivacyIDParams := DistinctPrivacyIDParams{MaxPartitionsContributed: 1, NoiseKind: LaplaceNoise{}, PublicPartitions: publicPartitions}
		counts := DistinctPrivacyID(s, pcol, distinctPrivacyIDParams)
		values := beam.DropKey(s, counts)
		// Check if we have negative elements.
		beam.ParDo0(s, testutils.CheckNoNegativeValuesInt64, values)
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestCountWithPartitionsReturnsNonNegative in-memory=%t returned errors: %v", tc.inMemory, err)
		}
	}
}

// Checks that DistinctPrivacyID deduplicates KV pairs *before* bounding
// contribution. To test that, we set the contribution to exactly the number of
// distinct values per key, and we duplicate many values: if contribution
// bounding isn't optimized, we should lose values.
func TestDistinctPrivacyIDOptimizedContrib(t *testing.T) {
	// pairs{privacy_id, partition_key} contain input data belonging to partitions 0, 1, 2 and 3.
	pairs := testutils.ConcatenatePairs(
		testutils.MakePairsWithFixedV(50, 0),
		testutils.MakePairsWithFixedV(50, 1),
		testutils.MakePairsWithFixedV(50, 2),
		testutils.MakePairsWithFixedV(50, 3),
		testutils.MakePairsWithFixedV(50, 0),
		testutils.MakePairsWithFixedV(50, 1),
		testutils.MakePairsWithFixedV(50, 2),
		testutils.MakePairsWithFixedV(50, 3))
	result := []testutils.PairII64{
		{0, 50},
		{1, 50},
		{2, 50},
		{3, 50},
	}
	p, s, col, want := ptest.CreateList2(pairs, result)
	col = beam.ParDo(s, testutils.PairToKV, col)

	// ε=50, δ=10⁻²⁰⁰ and l1Sensitivity=4 gives a post-aggregation threshold of 38.
	// We have 4 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁴ probability (k=24).
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-200, 24.0, 4.0
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:      epsilon,
			PartitionSelectionDelta: delta,
		}))
	got := DistinctPrivacyID(s, pcol, DistinctPrivacyIDParams{MaxPartitionsContributed: 4, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, testutils.PairII64ToKV, want)
	testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestDistinctPrivacyIDOptimizedContrib: DistinctPrivacyID(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

func TestNewCountFn(t *testing.T) {
	for _, tc := range []struct {
		desc                    string
		noiseKind               noise.Kind
		aggregationEpsilon      float64
		aggregationDelta        float64
		partitionSelectionDelta float64
		preThreshold            int64
		want                    *countFn
	}{
		{"Laplace noise kind", noise.LaplaceNoise, 1.0, 0.0, 1e-5, 0,
			&countFn{
				Epsilon:                  1.0,
				NoiseDelta:               0,
				ThresholdDelta:           1e-5,
				MaxPartitionsContributed: 17,
				NoiseKind:                noise.LaplaceNoise,
			}},
		{"Gaussian noise kind", noise.GaussianNoise, 1.0, 0.0, 1e-5, 0,
			&countFn{
				Epsilon:                  1.0,
				NoiseDelta:               0,
				ThresholdDelta:           1e-5,
				MaxPartitionsContributed: 17,
				NoiseKind:                noise.GaussianNoise,
			}},
		{"PreThreshold set", noise.LaplaceNoise, 1.0, 0.0, 1e-5, 10,
			&countFn{
				Epsilon:                  1.0,
				NoiseDelta:               0,
				ThresholdDelta:           1e-5,
				MaxPartitionsContributed: 17,
				PreThreshold:             10,
				NoiseKind:                noise.LaplaceNoise,
			}},
	} {
		got, err := newCountFn(PrivacySpec{preThreshold: tc.preThreshold, testMode: TestModeDisabled},
			DistinctPrivacyIDParams{
				AggregationEpsilon:       tc.aggregationEpsilon,
				AggregationDelta:         tc.aggregationDelta,
				PartitionSelectionDelta:  tc.partitionSelectionDelta,
				MaxPartitionsContributed: 17,
			}, tc.noiseKind, false)
		if err != nil {
			t.Fatalf("Couldn't get countFn: %v", err)
		}
		if diff := cmp.Diff(tc.want, got, cmpopts.IgnoreUnexported(countFn{})); diff != "" {
			t.Errorf("newCountFn mismatch for '%s' (-want +got):\n%s", tc.desc, diff)
		}
	}
}

func TestCountFnSetup(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		noiseKind noise.Kind
		wantNoise any
	}{
		{"Laplace noise kind", noise.LaplaceNoise, noise.Laplace()},
		{"Gaussian noise kind", noise.GaussianNoise, noise.Gaussian()},
	} {
		spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1, PartitionSelectionDelta: 1e-5})
		got, err := newCountFn(*spec, DistinctPrivacyIDParams{MaxPartitionsContributed: 17}, tc.noiseKind, false)
		if err != nil {
			t.Fatalf("Couldn't get countFn: %v", err)
		}
		got.Setup()
		if !cmp.Equal(tc.wantNoise, got.noise) {
			t.Errorf("Setup: for %s got %v, want %v", tc.desc, got.noise, tc.wantNoise)
		}
	}
}

func TestCountFnAddInput(t *testing.T) {
	cf := countFn{
		// Use ε=math.MaxFloat64 to deterministically not add any noise when calling ExtractOutput.
		Epsilon:    math.MaxFloat64,
		NoiseDelta: 0,
		// Use δ≈1 to make certain we do not threshold anything when calling ExtractOutput.
		ThresholdDelta:           1 - 1e-15,
		MaxPartitionsContributed: 1,
		NoiseKind:                noise.LaplaceNoise,
		noise:                    noise.Laplace(),
	}

	accum, err := cf.CreateAccumulator()
	if err != nil {
		t.Fatalf("Couldn't create accum: %v", err)
	}
	cf.AddInput(accum, 1)
	cf.AddInput(accum, 1)

	got, err := cf.ExtractOutput(accum)
	if err != nil {
		t.Fatalf("Couldn't extract output: %v", err)
	}
	want := testutils.Int64Ptr(2)
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("unexpected output (-want +got):\n%s", diff)
	}
}

func TestCountFnMergeAccumulators(t *testing.T) {
	cf := countFn{
		// Use ε=math.MaxFloat64 to deterministically not add any noise when calling ExtractOutput.
		Epsilon:    math.MaxFloat64,
		NoiseDelta: 0,
		// Use δ≈1 to make certain we do not threshold anything when calling ExtractOutput.
		ThresholdDelta:           1 - 1e-15,
		MaxPartitionsContributed: 1,
		NoiseKind:                noise.LaplaceNoise,
		noise:                    noise.Laplace(),
	}

	accum1, err := cf.CreateAccumulator()
	if err != nil {
		t.Fatalf("Couldn't create accum1: %v", err)
	}
	cf.AddInput(accum1, 1)
	cf.AddInput(accum1, 1)
	accum2, err := cf.CreateAccumulator()
	if err != nil {
		t.Fatalf("Couldn't create accum2: %v", err)
	}
	cf.AddInput(accum2, 1)
	cf.MergeAccumulators(accum1, accum2)

	got, err := cf.ExtractOutput(accum1)
	if err != nil {
		t.Fatalf("Couldn't extract output: %v", err)
	}
	want := testutils.Int64Ptr(3)
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("unexpected output (-want +got):\n%s", diff)
	}
}

func TestCountFnExtractOutputReturnsNilForSmallPartitions(t *testing.T) {
	// The laplace threshold for ε=ln3, δ=10⁻²⁰⁰, lInfSensitivity = 1 is ~ 420.
	// The raw count after adding 10 elements is equal to 10.
	// The laplace tolerance is equal to 48 for ε=ln3, l1Sensitivity=maxPartitionsContributed=1 and flakiness of 10⁻²³.
	// The rawCount + laplaceTolerance is less than the threshold with flakiness of 10⁻²³ for chosen parameters: 10 + 48 < 420.
	fn := countFn{
		Epsilon:                  ln3,
		NoiseDelta:               0,
		ThresholdDelta:           1e-200,
		MaxPartitionsContributed: 1,
		NoiseKind:                noise.LaplaceNoise,
	}

	fn.Setup()
	accum, err := fn.CreateAccumulator()
	if err != nil {
		t.Fatalf("Couldn't create accum: %v", err)
	}
	for i := 0; i < 10; i++ {
		fn.AddInput(accum, 1)
	}

	got, err := fn.ExtractOutput(accum)
	if err != nil {
		t.Fatalf("Couldn't extract output: %v", err)
	}

	// Should return nil output for small partitions.
	if got != nil {
		t.Errorf("ExtractOutput: for 10 added values got: %d, want nil", *got)
	}
}

func TestCountFnExtractOutputDoesNotReturnNilIfPartitionsPublic(t *testing.T) {
	// Thresholding does not occur because partitions are public.
	fn := countFn{
		Epsilon:                  ln3,
		NoiseDelta:               0,
		ThresholdDelta:           1e-200,
		MaxPartitionsContributed: 1,
		NoiseKind:                noise.LaplaceNoise,
		PublicPartitions:         true,
	}

	fn.Setup()
	accum, err := fn.CreateAccumulator()
	if err != nil {
		t.Fatalf("Couldn't create accum: %v", err)
	}
	fn.AddInput(accum, 1)

	got, err := fn.ExtractOutput(accum)
	if err != nil {
		t.Fatalf("Couldn't extract output: %v", err)
	}

	// Should not return nil output for small partitions, since partitions are public.
	if got == nil {
		t.Errorf("ExtractOutput: with public partitions for 1 added value got nil, want non-nil")
	}
}

func TestCheckDistinctPrivacyIDParams(t *testing.T) {
	_, _, partitions := ptest.CreateList([]int{0})
	for _, tc := range []struct {
		desc          string
		params        DistinctPrivacyIDParams
		noiseKind     noise.Kind
		partitionType reflect.Type
		wantErr       bool
	}{
		{
			desc: "valid parameters w/o public partitions",
			params: DistinctPrivacyIDParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionDelta:  1e-5,
				MaxPartitionsContributed: 1,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       false,
		},
		{
			desc: "valid parameters w/ gaussian noise w/o public partitions",
			params: DistinctPrivacyIDParams{
				AggregationEpsilon:       1.0,
				AggregationDelta:         1e-5,
				PartitionSelectionDelta:  1e-5,
				MaxPartitionsContributed: 1,
			},
			noiseKind:     noise.GaussianNoise,
			partitionType: nil,
			wantErr:       false,
		},
		{
			desc: "zero aggregationDelta w/ gaussian noise w/o public partitions",
			params: DistinctPrivacyIDParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionDelta:  1e-5,
				MaxPartitionsContributed: 1,
			},
			noiseKind:     noise.GaussianNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "valid parameters w/ public partitions",
			params: DistinctPrivacyIDParams{
				AggregationEpsilon:       1.0,
				PublicPartitions:         []int{0},
				MaxPartitionsContributed: 1,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(0),
			wantErr:       false,
		},
		{
			desc: "negative epsilon",
			params: DistinctPrivacyIDParams{
				AggregationEpsilon:       -1.0,
				PartitionSelectionDelta:  1e-5,
				MaxPartitionsContributed: 1,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "zero partitionSelectionDelta w/o public partitions",
			params: DistinctPrivacyIDParams{
				AggregationEpsilon:       1.0,
				MaxPartitionsContributed: 1,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "non-zero partitionSelectionDelta w/ laplace noise",
			params: DistinctPrivacyIDParams{
				AggregationEpsilon:       1.0,
				PartitionSelectionDelta:  1e-5,
				MaxPartitionsContributed: 1,
				PublicPartitions:         []int{},
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(0),
			wantErr:       true,
		},
		{
			desc: "unset MaxPartitionsContributed",
			params: DistinctPrivacyIDParams{
				AggregationEpsilon:      1.0,
				PartitionSelectionDelta: 1e-5,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "wrong partition type w/ public partitions as beam.PCollection",
			params: DistinctPrivacyIDParams{
				AggregationEpsilon:       1.0,
				MaxPartitionsContributed: 1,
				PublicPartitions:         partitions,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
		{
			desc: "wrong partition type w/ public partitions as slice",
			params: DistinctPrivacyIDParams{
				AggregationEpsilon:       1.0,
				MaxPartitionsContributed: 1,
				PublicPartitions:         []int{0},
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
		{
			desc: "wrong partition type w/ public partitions as array",
			params: DistinctPrivacyIDParams{
				AggregationEpsilon:       1.0,
				MaxPartitionsContributed: 1,
				PublicPartitions:         [1]int{0},
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
		{
			desc: "public partitions as something other than beam.PCollection, slice or array",
			params: DistinctPrivacyIDParams{
				AggregationEpsilon:       1.0,
				MaxPartitionsContributed: 1,
				PublicPartitions:         "",
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
	} {
		if err := checkDistinctPrivacyIDParams(tc.params, tc.noiseKind, tc.partitionType); (err != nil) != tc.wantErr {
			t.Errorf("With %s, got=%v error, wantErr=%t", tc.desc, err, tc.wantErr)
		}
	}
}

func TestDistinctPrivacyIDPreThresholding(t *testing.T) {
	// In this test, we set pre-threshold to 10, per-partition l1 sensitivity to 2, and:
	// - value 0 is associated with 9 privacy units, so it should be thresholded;
	// - value 1 is associated with 11 privacy units appearing twice each;
	// Each privacy unit contributes to at most 1 partition.
	pairs := testutils.ConcatenatePairs(
		testutils.MakePairsWithFixedVStartingFromKey(0, 9, 0),
		testutils.MakePairsWithFixedVStartingFromKey(10, 11, 1),
		testutils.MakePairsWithFixedVStartingFromKey(10, 11, 1),
	)
	result := []testutils.PairII64{
		// Partition 0 is dropped due to the pre-threshold.
		{1, 11},
	}
	p, s, col, want := ptest.CreateList2(pairs, result)
	col = beam.ParDo(s, testutils.PairToKV, col)

	// ε=10⁹, δ≈1 and l0Sensitivity=1 means partitions meeting the preThreshold should be kept.
	// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
	// we can have each partition fail with 10⁻²³ probability (k=23).
	epsilon, delta, k, l1Sensitivity := 1e9, dpagg.LargestRepresentableDelta, 23.0, 1.0
	preThreshold := int64(10)
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
			PreThreshold:              preThreshold}))
	got := DistinctPrivacyID(s, pcol, DistinctPrivacyIDParams{MaxPartitionsContributed: 1, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, testutils.PairII64ToKV, want)
	testutils.ApproxEqualsKVInt64(t, s, got, want, testutils.RoundedLaplaceTolerance(k, l1Sensitivity, epsilon))
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestDistinctPrivacyIDPreThresholding: Count(%v) = %v, expected %v: %v", col, got, want, err)
	}
}
