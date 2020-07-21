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

	"github.com/google/differential-privacy/go/noise"
	testpb "github.com/google/differential-privacy/privacy-on-beam/testdata"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/testing/passert"
	"github.com/apache/beam/sdks/go/pkg/beam/testing/ptest"
	"github.com/apache/beam/sdks/go/pkg/beam/transforms/stats"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/protobuf/proto"
)

var (
	ln3 = math.Log(3)
)

func init() {
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
	pcol := MakePrivateFromProto(s, col, NewPrivacySpec(1, 1e-10), "foo")
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
	pairs := concatenatePairs(
		makePairsWithFixedV(7, 0),
		makePairsWithFixedV(52, 1),
		makePairsWithFixedV(99, 2),
		makePairsWithFixedV(7, 0)) // duplicated values should have no influence.
	result := []testInt64Metric{
		// Only 7 users are associated to value 0: should be thresholded.
		{1, 52},
		{2, 99},
	}
	p, s, col, want := ptest.CreateList2(pairs, result)
	col = beam.ParDo(s, pairToKV, col)

	// ε=50, δ=10⁻²⁰⁰ and l1Sensitivity=4 gives a post-aggregation threshold of 37.
	// We have 4 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁵ probability (k=25).
	// To see the logic and the math behind flakiness and tolerance calculation,
	// See https://github.com/google/differential-privacy/blob/master/privacy-on-beam/docs/Tolerance_Calculation.pdf.
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-200, 25.0, 4.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	got := DistinctPrivacyID(s, pcol, DistinctPrivacyIDParams{MaxPartitionsContributed: 4, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, int64MetricToKV, want)
	if err := approxEqualsKVInt64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestDistinctPrivacyIDNoNoise: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestDistinctPrivacyIDNoNoise: DistinctPrivacyID(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

type distinctThresholdTestCase struct {
	name            string
	noiseKind       NoiseKind
	epsilon         float64
	delta           float64
	numValues       int
	minAllowedValue int
}

var distinctThresholdTestCases = []distinctThresholdTestCase{
	{
		name:      "Gaussian",
		noiseKind: GaussianNoise{},
		epsilon:   1,
		delta:     0.01,
		numValues: 25,
		// We use δ = 0.005 in these calculations since the δ = 0.01 budget is split
		// in half (50% for adding noise, 50% for thresholding).
		minAllowedValue: int(noise.Gaussian().Threshold(25, 1, 1, 0.005, 0.005)),
	},
	{
		name:            "Laplace",
		noiseKind:       LaplaceNoise{},
		epsilon:         1,
		delta:           0.01,
		numValues:       25,
		minAllowedValue: int(noise.Laplace().Threshold(25, 1, 1, 0, 0.01)),
	},
}

func buildDistinctPrivacyIDThresholdPipeline(tc distinctThresholdTestCase) (p *beam.Pipeline, s beam.Scope, col beam.PCollection, got beam.PCollection) {
	// pairs contains {1,0}, {2,0}, …, {minAllowedValue,0}, {1,1}, …, {minAllowedValue,1}, {1,2}, …, {minAllowedValue,9}.
	var pairs []pairII
	for i := 0; i < tc.numValues; i++ {
		// We use minAllowedValue keys per value to place each of the 25 values
		// right next to the distribution's Threshold.
		pairs = append(pairs, makePairsWithFixedV(tc.minAllowedValue, i)...)
	}
	p, s, col = ptest.CreateList(pairs)
	col = beam.ParDo(s, pairToKV, col)

	pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
	got = DistinctPrivacyID(s, pcol, DistinctPrivacyIDParams{MaxPartitionsContributed: int64(tc.numValues), NoiseKind: tc.noiseKind})
	got = beam.ParDo(s, kvToInt64Metric, got)
	return p, s, col, got
}

// Checks that DistinctPrivacyID correctly removes data under the threshold.
func TestDistinctPrivacyIDThresholdsSmallEntries(t *testing.T) {
	for _, tc := range distinctThresholdTestCases {
		t.Run(tc.name, func(t *testing.T) {
			// Sanity check that the minAllowedValue is sensical.
			if tc.minAllowedValue <= 0 {
				t.Errorf("Invalid test case: minAllowedValue must be positive. Got: %d", tc.minAllowedValue)
			}

			p, s, col, got := buildDistinctPrivacyIDThresholdPipeline(tc)
			beam.ParDo0(s, &checkNothingBelowThresholdFn{tc.minAllowedValue}, got)
			if err := ptest.Run(p); err != nil {
				t.Errorf("%s: DistinctPrivacyID(%v) = %v, found an unexpected value below minAllowedValue: %v", tc.name, col, got, err)
			}
		})
	}
}

type checkNothingBelowThresholdFn struct {
	Threshold int
}

func (fn *checkNothingBelowThresholdFn) ProcessElement(c testInt64Metric) error {
	if c.Metric < int64(fn.Threshold) {
		return fmt.Errorf("found a count of %d<%d for value %d", c.Metric, fn.Threshold, c.Value)
	}
	return nil
}

// Checks that DistinctPrivacyID does not remove all data (portion above the
// threshold should be maintained).
func TestDistinctPrivacyIDThresholdLeavesSomeEntries(t *testing.T) {
	for _, tc := range distinctThresholdTestCases {
		t.Run(tc.name, func(t *testing.T) {
			// Sanity check that the minAllowedValue is sensical.
			if tc.minAllowedValue <= 0 {
				t.Errorf("Invalid test case: minAllowedValue must be positive. Got: %d", tc.minAllowedValue)
			}

			p, s, col, got := buildDistinctPrivacyIDThresholdPipeline(tc)
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
		// Differential privacy params used. The test assumes sensitivities of 1.
		epsilon float64
		delta   float64
	}{
		{
			name:      "Gaussian",
			noiseKind: GaussianNoise{},
			epsilon:   1,
			delta:     0.01, // It is split by 2: 0.005 for the noise and 0.005 for the partition selection
		},
		{

			name:      "Laplace",
			noiseKind: LaplaceNoise{},
			epsilon:   0.1,
			delta:     0.01,
		},
	} {
		// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
		// we need to have each partition pass with 1-10⁻²³ probability (k=23).
		epsilonNoise, deltaNoise := tc.epsilon, 0.0
		k := 23.0
		l0Sensitivity, lInfSensitivity := 1.0, 1.0
		epsilonPartition, deltaPartition := tc.epsilon, tc.delta
		l1Sensitivity := l0Sensitivity * lInfSensitivity
		tolerance := complementaryLaplaceTolerance(k, l1Sensitivity, epsilonNoise)
		numIDs := int(noise.Laplace().Threshold(1, 1, epsilonPartition, deltaNoise, deltaPartition) + tolerance)
		if tc.noiseKind == gaussianNoise {
			deltaNoise = tc.delta / 2
			deltaPartition = tc.delta / 2
			tolerance = complementaryGaussianTolerance(k, l0Sensitivity, lInfSensitivity, epsilonNoise, deltaNoise)
			numIDs = int(noise.Gaussian().Threshold(1, 1, epsilonNoise, deltaNoise, deltaPartition) + tolerance)
		}
		// pairs contains {1,0}, {2,0}, …, {numIDs,0}.
		pairs := makePairsWithFixedV(numIDs, 0)
		p, s, col := ptest.CreateList(pairs)
		col = beam.ParDo(s, pairToKV, col)

		pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
		got := DistinctPrivacyID(s, pcol, DistinctPrivacyIDParams{MaxPartitionsContributed: 1, NoiseKind: tc.noiseKind})
		got = beam.ParDo(s, kvToInt64Metric, got)

		checkInt64MetricsAreNoisy(s, got, numIDs, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("DistinctPrivacyID didn't add any noise: %v", err)
		}
	}
}

// Checks that DistinctPrivacyID bounds per-user contributions correctly.
// The logic mirrors TestCountCrossPartitionContributionBounding.
func TestDistinctPrivacyIDCrossPartitionContributionBounding(t *testing.T) {
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

	// ε=50, δ=0.01 and l1Sensitivity=3 gives a post-aggregation threshold of 1.
	// We have 10 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁵ probability (k=25).
	epsilon, delta, k, l1Sensitivity := 50.0, 0.01, 25.0, 3.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	got := DistinctPrivacyID(s, pcol, DistinctPrivacyIDParams{MaxPartitionsContributed: 3, NoiseKind: LaplaceNoise{}})
	// With a max contribution of 3, 70% of the data should have be
	// dropped. The sum of all elements must then be 150.
	counts := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, counts)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
	want = beam.ParDo(s, int64MetricToKV, want)
	if err := approxEqualsKVInt64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestDistinctPrivacyIDCrossPartitionContributionBounding: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestDistinctPrivacyIDCrossPartitionContributionBounding: DistinctPrivacyID(%v) = %v, expected elements to sum to 150: %v", col, got, err)
	}
}

// Checks that DistinctPrivacyID deduplicates KV pairs *before* bounding
// contribution. To test that, we set the contribution to exactly the number of
// distinct values per key, and we duplicate many values: if contribution
// bounding isn't optimized, we should lose values.
func TestDistinctPrivacyIDOptimizedContrib(t *testing.T) {
	pairs := concatenatePairs(
		makePairsWithFixedV(50, 0),
		makePairsWithFixedV(50, 1),
		makePairsWithFixedV(50, 2),
		makePairsWithFixedV(50, 3),
		makePairsWithFixedV(50, 0),
		makePairsWithFixedV(50, 1),
		makePairsWithFixedV(50, 2),
		makePairsWithFixedV(50, 3))
	result := []testInt64Metric{
		{0, 50},
		{1, 50},
		{2, 50},
		{3, 50},
	}
	p, s, col, want := ptest.CreateList2(pairs, result)
	col = beam.ParDo(s, pairToKV, col)

	// ε=50, δ=10⁻²⁰⁰ and l1Sensitivity=4 gives a post-aggregation threshold of 37.
	// We have 4 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁵ probability (k=25).
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-200, 25.0, 4.0
	pcol := MakePrivate(s, col, NewPrivacySpec(epsilon, delta))
	got := DistinctPrivacyID(s, pcol, DistinctPrivacyIDParams{MaxPartitionsContributed: 4, NoiseKind: LaplaceNoise{}})
	want = beam.ParDo(s, int64MetricToKV, want)
	if err := approxEqualsKVInt64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestDistinctPrivacyIDOptimizedContrib: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestDistinctPrivacyIDOptimizedContrib: DistinctPrivacyID(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

func TestNewCountFn(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		noiseKind noise.Kind
		want      *countFn
	}{
		{"Laplace", noise.LaplaceNoise,
			&countFn{
				Epsilon:                  1,
				DeltaNoise:               0,
				DeltaThreshold:           1e-5,
				MaxPartitionsContributed: 17,
				NoiseKind:                noise.LaplaceNoise,
			}},
		{"Gaussian", noise.GaussianNoise,
			&countFn{
				Epsilon:                  1,
				DeltaNoise:               5e-6,
				DeltaThreshold:           5e-6,
				MaxPartitionsContributed: 17,
				NoiseKind:                noise.GaussianNoise,
			}},
	} {
		got := newCountFn(1, 1e-5, 17, tc.noiseKind)
		if diff := cmp.Diff(tc.want, got, cmpopts.IgnoreUnexported(countFn{})); diff != "" {
			t.Errorf("newCountFn mismatch for '%s' (-want +got):\n%s", tc.desc, diff)
		}
	}
}

func TestCountFnSetup(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		noiseKind noise.Kind
		wantNoise interface{}
	}{
		{"Laplace noise kind", noise.LaplaceNoise, noise.Laplace()},
		{"Gaussian noise kind", noise.GaussianNoise, noise.Gaussian()}} {
		got := newCountFn(1, 1e-5, 17, tc.noiseKind)
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
		DeltaNoise: 0,
		// Use δ=1 to make certain we do not threshold anything when calling ExtractOutput.
		DeltaThreshold:           1,
		MaxPartitionsContributed: 1,
		NoiseKind:                noise.LaplaceNoise,
		noise:                    noise.Laplace(),
	}

	accum1 := cf.CreateAccumulator()
	cf.AddInput(accum1, 1)
	cf.AddInput(accum1, 1)

	got := cf.ExtractOutput(accum1)
	want := int64Ptr(2)
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("unexpected output (-want +got):\n%s", diff)
	}
}

func TestCountFnMergeAccumulators(t *testing.T) {
	cf := countFn{
		// Use ε=math.MaxFloat64 to deterministically not add any noise when calling ExtractOutput.
		Epsilon:    math.MaxFloat64,
		DeltaNoise: 0,
		// Use δ=1 to make certain we do not threshold anything when calling ExtractOutput.
		DeltaThreshold:           1,
		MaxPartitionsContributed: 1,
		NoiseKind:                noise.LaplaceNoise,
		noise:                    noise.Laplace(),
	}

	accum1 := cf.CreateAccumulator()
	cf.AddInput(accum1, 1)
	cf.AddInput(accum1, 1)
	accum2 := cf.CreateAccumulator()
	cf.AddInput(accum2, 1)
	cf.MergeAccumulators(accum1, accum2)

	got := cf.ExtractOutput(accum1)
	want := int64Ptr(3)
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
		DeltaNoise:               0,
		DeltaThreshold:           1e-200,
		MaxPartitionsContributed: 1,
		NoiseKind:                noise.LaplaceNoise,
	}

	fn.Setup()
	accum := fn.CreateAccumulator()
	for i := 0; i < 10; i++ {
		fn.AddInput(accum, 1)
	}

	got := fn.ExtractOutput(accum)

	// Should return nil output for small partitions.
	if got != nil {
		t.Errorf("ExtractOutput: for 10 added values got: %d, want nil", *got)
	}
}
