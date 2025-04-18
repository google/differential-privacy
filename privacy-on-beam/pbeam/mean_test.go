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
	"strings"
	"testing"

	"github.com/google/differential-privacy/go/v4/dpagg"
	"github.com/google/differential-privacy/go/v4/noise"
	"github.com/google/differential-privacy/privacy-on-beam/v4/pbeam/testutils"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/register"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/testing/ptest"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/stats"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

type msSliceIter = func(*MeanStatistics) bool

func init() {
	register.DoFn1x1[pairIntMS, error](&checkMeanStatisticsAreNoisyFn{})
	register.DoFn2x1[int, MeanStatistics, error](&checkMSIsInRange{})
	register.DoFn3x1[int, msSliceIter, msSliceIter, string](&diffMeanStatisticsFn{})
}

func approxEqualMeanStatistics(
	t *testing.T,
	got MeanStatistics,
	want,
	tolerance testutils.VarianceStatistics,
) {
	t.Helper()
	w := testUtilsVSToPbeamMS(want)
	tol := testUtilsVSToPbeamMS(tolerance)
	if diff := approxDiffMS(got, w, tol); diff != "" {
		t.Errorf("MeanStatistics diff (-got +want):\n%s", diff)
	}
}

func approxDiffMS(got, want, tolerance MeanStatistics) string {
	var b strings.Builder
	if !cmp.Equal(got.Count, want.Count, cmpopts.EquateApprox(0, tolerance.Count)) {
		fmt.Fprintf(&b, "- \tCount: %f\n", got.Count)
		fmt.Fprintf(&b, "+ \tCount: %f\n", want.Count)
		fmt.Fprintf(&b, "  \tCount Tolerance: %f\n", tolerance.Count)
	}
	if !cmp.Equal(got.Mean, want.Mean, cmpopts.EquateApprox(0, tolerance.Mean)) {
		fmt.Fprintf(&b, "- \tMean: %f\n", got.Mean)
		fmt.Fprintf(&b, "+ \tMean: %f\n", want.Mean)
		fmt.Fprintf(&b, "  \tMean Tolerance: %f\n", tolerance.Mean)
	}
	return b.String()
}

func testUtilsVSToPbeamMS(vs testutils.VarianceStatistics) MeanStatistics {
	return MeanStatistics{
		Count: vs.Count,
		Mean:  vs.Mean,
	}
}

type pairIntMS struct {
	K  int
	MS MeanStatistics
}

func kMSToPairIntMS(k int, ms MeanStatistics) pairIntMS {
	return pairIntMS{K: k, MS: ms}
}

func pairIntMSToKMS(p pairIntMS) (int, MeanStatistics) {
	return p.K, p.MS
}

// checkMSsAreNoisy checks that the got MeanStatistics are noisy compared to the exactStat.
// That is, the differences in all fields exceed the given tolerance.
//
// got is a PCollection<K,MeanStatistics>.
func checkMSsAreNoisy(
	s beam.Scope,
	got beam.PCollection,
	want testutils.VarianceStatistics,
	tolerance MeanStatistics,
) {
	fn := &checkMeanStatisticsAreNoisyFn{
		ExactStat: testUtilsVSToPbeamMS(want),
		Tolerance: tolerance,
	}
	// PCollection<K,MeanStatistics> -> PCollection<pairIntMS>.
	got = beam.ParDo(s, kMSToPairIntMS, got)

	// PCollection<pairIntMS> -> N/A.
	beam.ParDo0(s, fn, got)
}

type checkMeanStatisticsAreNoisyFn struct {
	ExactStat MeanStatistics
	Tolerance MeanStatistics
}

func (fn *checkMeanStatisticsAreNoisyFn) ProcessElement(p pairIntMS) error {
	exact, tol := &fn.ExactStat, &fn.Tolerance
	if cmp.Equal(p.MS.Count, exact.Count, cmpopts.EquateApprox(0, tol.Count)) {
		return fmt.Errorf("found a non-noisy count of %f for (key, exact, tolerance)=(%d, %f, %f)",
			p.MS.Count, p.K, exact.Count, tol.Count)
	}
	if cmp.Equal(p.MS.Mean, exact.Mean, cmpopts.EquateApprox(0, tol.Mean)) {
		return fmt.Errorf("found a non-noisy mean of %f for (key, exact, tolerance)=(%d, %f, %f)",
			p.MS.Mean, p.K, exact.Mean, tol.Mean)
	}
	return nil
}

// checkMSsAreInRange checks that the given PCollection<K,MeanStatistics> has values in the
// expected range.
//   - The mean is in range [minValue, maxValue].
//
// The input got is a PCollection<K, MeanStatistics>.
func checkMSsAreInRange(s beam.Scope, got beam.PCollection, minValue, maxValue float64) {
	fn := &checkMSIsInRange{MinValue: minValue, MaxValue: maxValue}
	beam.ParDo0(s, fn, got)
}

type checkMSIsInRange struct {
	MinValue float64
	MaxValue float64
}

func (fn *checkMSIsInRange) ProcessElement(k int, ms MeanStatistics) error {
	if ms.Mean < fn.MinValue || ms.Mean > fn.MaxValue {
		return fmt.Errorf("partition %d: found a mean of %f outside of the expected range [%f, %f]",
			k, ms.Mean, fn.MinValue, fn.MaxValue)
	}
	return nil
}

// approxEqualsKMS checks that the got PCollection<K, MeanStatistics> and want
// PCollection<K, MeanStatistics> have the same key set and that all values of the same key
// has only a single MeanStatistics and are approximately equal (within the given tolerance).
//
//   - got: PCollection<K, MeanStatistics>
//   - want: PCollection<K, MeanStatistics>
func approxEqualsKMS(
	s beam.Scope, got, want beam.PCollection, tolerance testutils.VarianceStatistics,
) {
	fn := &diffMeanStatisticsFn{Tolerance: testUtilsVSToPbeamMS(tolerance)}

	// PCollection<K, []MeanStatistics, []MeanStatistics>
	grouped := beam.CoGroupByKey(s, got, want)

	// PCollection<K, string>
	diffPerKey := beam.ParDo(s, fn, grouped)

	// PCollection<string>
	wholeDiff := beam.Combine(s, testutils.CombineDiffs, diffPerKey)

	// Report error if there is any non-empty diff.
	beam.ParDo0(s, testutils.ReportDiffs, wholeDiff)
}

type diffMeanStatisticsFn struct {
	Tolerance MeanStatistics
}

func (fn *diffMeanStatisticsFn) ProcessElement(
	k int, gotSliceIter, wantSliceIter msSliceIter,
) string {
	gotSlice := testutils.SliceFromIter(gotSliceIter)
	wantSlice := testutils.SliceFromIter(wantSliceIter)
	if len(gotSlice) != 1 || len(wantSlice) != 1 {
		return fmt.Sprintf(
			"MeanStatistics of key %d: got has %d values, want has %d values",
			k, len(gotSlice), len(wantSlice),
		)
	}
	if diff := approxDiffMS(gotSlice[0], wantSlice[0], fn.Tolerance); diff != "" {
		return fmt.Sprintf("MeanStatistics of key %d diff (-got +want):\n%s", k, diff)
	}
	return "" // No diff for key k.
}

func TestNewBoundedMeanFn(t *testing.T) {
	opts := []cmp.Option{
		cmpopts.EquateApprox(0, 1e-10),
		cmpopts.IgnoreUnexported(boundedMeanFn{}),
	}
	for _, tc := range []struct {
		desc                      string
		noiseKind                 noise.Kind
		aggregationEpsilon        float64
		aggregationDelta          float64
		partitionSelectionEpsilon float64
		partitionSelectionDelta   float64
		preThreshold              int64
		want                      *boundedMeanFn
	}{
		{"Laplace noise kind", noise.LaplaceNoise, 1.0, 0, 1.0, 1e-5, 0,
			&boundedMeanFn{
				NoiseEpsilon:                 1.0,
				NoiseDelta:                   0,
				PartitionSelectionEpsilon:    1.0,
				PartitionSelectionDelta:      1e-5,
				MaxPartitionsContributed:     17,
				MaxContributionsPerPartition: 5,
				Lower:                        0,
				Upper:                        10,
				NoiseKind:                    noise.LaplaceNoise,
			}},
		{"Gaussian noise kind", noise.GaussianNoise, 1.0, 1e-5, 1.0, 1e-5, 0,
			&boundedMeanFn{
				NoiseEpsilon:                 1.0,
				NoiseDelta:                   1e-5,
				PartitionSelectionEpsilon:    1.0,
				PartitionSelectionDelta:      1e-5,
				MaxPartitionsContributed:     17,
				MaxContributionsPerPartition: 5,
				Lower:                        0,
				Upper:                        10,
				NoiseKind:                    noise.GaussianNoise,
			}},
		{"PreThreshold set", noise.GaussianNoise, 1.0, 1e-5, 1.0, 1e-5, 10,
			&boundedMeanFn{
				NoiseEpsilon:                 1.0,
				NoiseDelta:                   1e-5,
				PartitionSelectionEpsilon:    1.0,
				PartitionSelectionDelta:      1e-5,
				PreThreshold:                 10,
				MaxPartitionsContributed:     17,
				MaxContributionsPerPartition: 5,
				Lower:                        0,
				Upper:                        10,
				NoiseKind:                    noise.GaussianNoise,
			}},
	} {
		got, err := newBoundedMeanFn(PrivacySpec{preThreshold: tc.preThreshold, testMode: TestModeDisabled},
			MeanParams{
				AggregationEpsilon:           tc.aggregationEpsilon,
				AggregationDelta:             tc.aggregationDelta,
				PartitionSelectionParams:     PartitionSelectionParams{Epsilon: tc.partitionSelectionEpsilon, Delta: tc.partitionSelectionDelta},
				MaxPartitionsContributed:     17,
				MaxContributionsPerPartition: 5,
				MinValue:                     0,
				MaxValue:                     10,
			}, tc.noiseKind, false, false)
		if err != nil {
			t.Fatalf("Couldn't get newBoundedMeanFn: %v", err)
		}
		if diff := cmp.Diff(tc.want, got, opts...); diff != "" {
			t.Errorf("newBoundedMeanFn: for %q (-want +got):\n%s", tc.desc, diff)
		}
	}
}

func TestBoundedMeanFnSetup(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		noiseKind noise.Kind
		wantNoise any
	}{
		{"Laplace noise kind", noise.LaplaceNoise, noise.Laplace()},
		{"Gaussian noise kind", noise.GaussianNoise, noise.Gaussian()}} {
		spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1, PartitionSelectionEpsilon: 1, PartitionSelectionDelta: 1e-5})
		got, err := newBoundedMeanFn(*spec, MeanParams{
			AggregationEpsilon:           1,
			PartitionSelectionParams:     PartitionSelectionParams{Epsilon: 1, Delta: 1e-5},
			MaxPartitionsContributed:     17,
			MaxContributionsPerPartition: 5,
			MinValue:                     0,
			MaxValue:                     10,
		}, tc.noiseKind, false, false)
		if err != nil {
			t.Fatalf("Couldn't get newBoundedMeanFn: %v", err)
		}
		got.Setup()
		if !cmp.Equal(tc.wantNoise, got.noise) {
			t.Errorf("Setup: for %s got %v, want %v", tc.desc, got.noise, tc.wantNoise)
		}
	}
}

func TestBoundedMeanFnAddInput(t *testing.T) {
	// δ=10⁻²³, ε=1e100 and l0Sensitivity=1 gives a threshold of =2.
	// Since ε=1e100, the noise is added with probability in the order of exp(-1e100).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 1e100
	delta := 1e-23
	minValue := 0.0
	maxValue := 5.0
	spec := privacySpec(t, PrivacySpecParams{
		AggregationEpsilon:        epsilon,
		PartitionSelectionEpsilon: epsilon,
		PartitionSelectionDelta:   delta,
	})
	fn, err := newBoundedMeanFn(*spec, MeanParams{
		AggregationEpsilon:           epsilon,
		PartitionSelectionParams:     PartitionSelectionParams{Epsilon: epsilon, Delta: delta},
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
	}, noise.LaplaceNoise, false, false)
	if err != nil {
		t.Fatalf("Couldn't get newBoundedMeanFn: %v", err)
	}
	fn.Setup()

	accum, err := fn.CreateAccumulator()
	if err != nil {
		t.Fatalf("Couldn't create accum: %v", err)
	}
	fn.AddInput(accum, []float64{2.0})
	fn.AddInput(accum, []float64{4.0})

	got, err := fn.ExtractOutput(accum)
	if err != nil {
		t.Fatalf("Couldn't extract output: %v", err)
	}
	if got == nil {
		t.Fatal("ExtractOutput: got nil, want non-nil")
	}
	want := testutils.VarianceStatistics{
		Count:         2.0,
		NormalizedSum: 1.,
	}
	want.ComputeMean(minValue, maxValue)
	tolerance, err := testutils.LaplaceToleranceForMean(
		23, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed, epsilon, want)
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	approxEqualMeanStatistics(t, *got, want, tolerance)
}

func TestBoundedMeanFnMergeAccumulators(t *testing.T) {
	// δ=10⁻²³, ε=1e100 and l0Sensitivity=1 gives a threshold of =2.
	// Since ε=1e100, the noise is added with probability in the order of exp(-1e100).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 1e100
	delta := 1e-23
	minValue := 0.0
	maxValue := 5.0
	spec := privacySpec(t, PrivacySpecParams{
		AggregationEpsilon:        epsilon,
		PartitionSelectionEpsilon: epsilon,
		PartitionSelectionDelta:   delta,
	})
	fn, err := newBoundedMeanFn(*spec, MeanParams{
		AggregationEpsilon:           epsilon,
		PartitionSelectionParams:     PartitionSelectionParams{Epsilon: epsilon, Delta: delta},
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
	}, noise.LaplaceNoise, false, false)
	if err != nil {
		t.Fatalf("Couldn't get newBoundedMeanFn: %v", err)
	}
	fn.Setup()

	accum1, err := fn.CreateAccumulator()
	if err != nil {
		t.Fatalf("Couldn't create accum1: %v", err)
	}
	fn.AddInput(accum1, []float64{2.0})
	fn.AddInput(accum1, []float64{3.0})
	fn.AddInput(accum1, []float64{1.0})
	accum2, err := fn.CreateAccumulator()
	if err != nil {
		t.Fatalf("Couldn't create accum2: %v", err)
	}
	fn.AddInput(accum2, []float64{4.0})
	fn.MergeAccumulators(accum1, accum2)

	got, err := fn.ExtractOutput(accum1)
	if err != nil {
		t.Fatalf("Couldn't extract output: %v", err)
	}
	if got == nil {
		t.Fatal("ExtractOutput: got nil, want non-nil")
	}
	want := testutils.VarianceStatistics{
		Count:         4.0,
		NormalizedSum: 0,
	}
	want.ComputeMean(minValue, maxValue)
	tolerance, err := testutils.LaplaceToleranceForMean(
		23, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed, epsilon, want)
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	approxEqualMeanStatistics(t, *got, want, tolerance)
}

func TestBoundedMeanFnExtractOutputReturnsNilForSmallPartitions(t *testing.T) {
	for _, tc := range []struct {
		desc                     string
		inputSize                int
		datapointsPerPrivacyUnit int
	}{
		// It's a special case for partition selection in which the algorithm should always eliminate the partition.
		{"Empty input", 0, 0},
		{"Input with 1 privacy unit with 1 contribution", 1, 1},
	} {
		// The choice of ε=1e100, δ=10⁻²³, and l0Sensitivity=1 gives a threshold of =2.
		spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1e100, PartitionSelectionEpsilon: 1e100, PartitionSelectionDelta: 1e-23})
		fn, err := newBoundedMeanFn(*spec, MeanParams{
			AggregationEpsilon:           1e100,
			PartitionSelectionParams:     PartitionSelectionParams{Epsilon: 1e100, Delta: 1e-23},
			MaxPartitionsContributed:     1,
			MaxContributionsPerPartition: 1,
			MinValue:                     0,
			MaxValue:                     10,
		}, noise.LaplaceNoise, false, false)
		if err != nil {
			t.Fatalf("Couldn't get newBoundedMeanFn: %v", err)
		}
		fn.Setup()
		accum, err := fn.CreateAccumulator()
		if err != nil {
			t.Fatalf("Couldn't create accum: %v", err)
		}
		for i := 0; i < tc.inputSize; i++ {
			values := make([]float64, tc.datapointsPerPrivacyUnit)
			for i := 0; i < tc.datapointsPerPrivacyUnit; i++ {
				values[i] = 1.0
			}
			fn.AddInput(accum, values)
		}

		got, err := fn.ExtractOutput(accum)
		if err != nil {
			t.Fatalf("Couldn't extract output: %v", err)
		}

		// Should return nil output for small partitions.
		if got != nil {
			t.Errorf("ExtractOutput: for %s got: %f, want nil", tc.desc, *got)
		}
	}
}

func TestBoundedMeanFnWithPartitionsExtractOutputDoesNotReturnNilForSmallPartitions(t *testing.T) {
	for _, tc := range []struct {
		desc              string
		inputSize         int
		datapointsPerUser int
	}{
		{"Empty input", 0, 0},
		{"Input with 1 user with 1 contribution", 1, 1},
	} {
		spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1e100})
		fn, err := newBoundedMeanFn(*spec, MeanParams{
			AggregationEpsilon:           1e100,
			MaxPartitionsContributed:     1,
			MaxContributionsPerPartition: 1,
			MinValue:                     0,
			MaxValue:                     10,
		}, noise.LaplaceNoise, true, false)
		if err != nil {
			t.Fatalf("Couldn't get newBoundedMeanFn: %v", err)
		}
		fn.Setup()
		accum, err := fn.CreateAccumulator()
		if err != nil {
			t.Fatalf("Couldn't create accum: %v", err)
		}
		for i := 0; i < tc.inputSize; i++ {
			values := make([]float64, tc.datapointsPerUser)
			for i := 0; i < tc.datapointsPerUser; i++ {
				values[i] = 1.0
			}
			fn.AddInput(accum, values)
		}

		got, err := fn.ExtractOutput(accum)
		if err != nil {
			t.Fatalf("Couldn't extract output: %v", err)
		}

		// Should not return nil output for small partitions in the case of public partitions.
		if got == nil {
			t.Errorf("ExtractOutput for %s thresholded with public partitions when it shouldn't", tc.desc)
		}
	}
}

// Checks that MeanStatisticsPerKey adds noise to its output.
func TestMeanStatisticsPerKeyAddsNoise(t *testing.T) {
	params := &testutils.MeanVarianceAPIAddsNoiseTestParams
	for _, tc := range []struct {
		name      string
		noiseKind NoiseKind
		// Differential privacy params used
		aggregationEpsilon        float64
		aggregationDelta          float64
		partitionSelectionEpsilon float64
		partitionSelectionDelta   float64
	}{
		{
			name:                      "Gaussian",
			noiseKind:                 GaussianNoise{},
			aggregationEpsilon:        params.AggEpsGaussian,
			aggregationDelta:          params.AggDelGaussian,
			partitionSelectionEpsilon: 1,
			partitionSelectionDelta:   0.005,
		},
		{
			name:                      "Laplace",
			noiseKind:                 LaplaceNoise{},
			aggregationEpsilon:        params.AggEpsLaplace,
			partitionSelectionEpsilon: 0.1,
			partitionSelectionDelta:   0.01,
		},
	} {
		minValue := 0.0
		maxValue := 3.0
		maxPartitionsContributed, maxContributionsPerPartition := int64(1), int64(1)

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
		expected := testutils.PerPartitionVarianceStatistics(minValue, maxValue, triples)[0]
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
		got := MeanStatisticsPerKey(s, pcol, MeanParams{
			MaxPartitionsContributed:     maxPartitionsContributed,
			MaxContributionsPerPartition: maxContributionsPerPartition,
			MinValue:                     minValue,
			MaxValue:                     maxValue,
			NoiseKind:                    tc.noiseKind,
		})

		// We check that any noise is added, hence tolerance is 0.0.
		// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf.
		checkMSsAreNoisy(s, got, expected, MeanStatistics{})
		if err := ptest.Run(p); err != nil {
			t.Errorf("MeanStatisticsPerKey didn't add any noise with float inputs and %s Noise: %v", tc.name, err)
		}
	}
}

// Checks that MeanStatisticsPerKey with partitions adds noise to its output.
func TestMeanStatisticsPerKeyWithPartitionsAddsNoise(t *testing.T) {
	params := &testutils.MeanVarianceAPIAddsNoiseTestParams
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
			epsilon:   params.AggEpsGaussian,
			delta:     params.AggDelGaussian,
			inMemory:  false,
		},
		{
			desc:      "as slice w/ Gaussian",
			noiseKind: GaussianNoise{},
			epsilon:   params.AggEpsGaussian,
			delta:     params.AggDelGaussian,
			inMemory:  true,
		},
		{
			desc:      "as PCollection w/ Laplace",
			noiseKind: LaplaceNoise{},
			epsilon:   params.AggEpsLaplace,
			delta:     0, // It is 0 because partitions are public and we are using Laplace noise.
			inMemory:  false,
		},
		{
			desc:      "as PCollection w/ Laplace",
			noiseKind: LaplaceNoise{},
			epsilon:   params.AggEpsLaplace,
			delta:     0, // It is 0 because partitions are public and we are using Laplace noise.
			inMemory:  true,
		},
	} {
		minValue := 0.0
		maxValue := 3.0
		maxPartitionsContributed, maxContributionsPerPartition := int64(1), int64(1)

		numIDs := 10
		// triples contains {1,0,1}, {2,0,1}, …, {numIDs,0,1}.
		triples := testutils.MakeSampleTripleWithFloatValue(numIDs, 0)
		expected := testutils.PerPartitionVarianceStatistics(minValue, maxValue, triples)[0]

		p, s, col := ptest.CreateList(triples)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		publicPartitionsSlice := []int{0}
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
		pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		meanParams := MeanParams{
			MaxPartitionsContributed:     maxPartitionsContributed,
			MaxContributionsPerPartition: maxContributionsPerPartition,
			MinValue:                     minValue,
			MaxValue:                     maxValue,
			NoiseKind:                    tc.noiseKind,
			PublicPartitions:             publicPartitions,
		}
		got := MeanStatisticsPerKey(s, pcol, meanParams)

		// We check that any noise is added, hence tolerance is 0.0.
		// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf.
		checkMSsAreNoisy(s, got, expected, MeanStatistics{})
		if err := ptest.Run(p); err != nil {
			t.Errorf("MeanStatisticsPerKey with partitions %s didn't add any noise with float inputs: %v", tc.desc, err)
		}
	}
}

// Checks that MeanStatisticsPerKey returns a correct answer for float input values.
func TestMeanStatisticsPerKeyNoNoiseFloat(t *testing.T) {
	// Arrange
	triples := testutils.ConcatenateTriplesWithFloatValue(
		testutils.MakeTripleWithFloatValue(7, 0, 2.0),
		testutils.MakeTripleWithFloatValueStartingFromKey(7, 100, 1, 1.3),
		testutils.MakeTripleWithFloatValueStartingFromKey(107, 150, 1, 2.5))

	minValue := 1.0
	maxValue := 3.0

	// Only get statistics for partition 1 because partition 0 will be dropped due to thresholding.
	stat := testutils.PerPartitionVarianceStatistics(minValue, maxValue, triples)[1]
	result := []pairIntMS{
		{K: 1, MS: testUtilsVSToPbeamMS(stat)},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l0Sensitivity=1 gives a threshold of =11.
	// We have 2 partitions. So, to get an overall flakiness of 10⁻²³,
	// we can have each partition fail with 10⁻²⁴ probability (k=24).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 50.0
	delta := 1e-200

	// Act
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	got := MeanStatisticsPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
		NoiseKind:                    LaplaceNoise{},
	})

	// Assert
	want = beam.ParDo(s, pairIntMSToKMS, want)
	tolerance, err := testutils.LaplaceToleranceForMean(
		24, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed, epsilon, stat)
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	approxEqualsKMS(s, got, want, tolerance)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanStatisticsPerKeyNoNoiseFloat: MeanStatisticsPerKey(%v) = %v, want %v, error %v",
			col, got, want, err)
	}
}

// Checks that MeanStatisticsPerKey returns a correct answer for int input values.
// They should be correctly converted to float64 and then correct result
// with float statistic should be computed.
func TestMeanStatisticsPerKeyNoNoiseInt(t *testing.T) {
	// Arrange
	triples := testutils.ConcatenateTriplesWithIntValue(
		testutils.MakeTripleWithIntValue(7, 0, 2),
		testutils.MakeTripleWithIntValueStartingFromKey(7, 100, 1, 1),
		testutils.MakeTripleWithIntValueStartingFromKey(107, 150, 1, 2))

	minValue := 0.0
	maxValue := 2.0
	// Only get statistics for partition 1 because partition 0 will be dropped due to thresholding.
	stat := testutils.PerPartitionVarianceStatisticsInt(minValue, maxValue, triples)[1]
	result := []pairIntMS{
		{K: 1, MS: testUtilsVSToPbeamMS(stat)},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l0Sensitivity=1 gives a threshold of =11.
	// We have 2 partitions. So, to get an overall flakiness of 10⁻²³,
	// we can have each partition fail with 10⁻²⁴ probability (k=24).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 50.0
	delta := 1e-200

	// Act
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
	got := MeanStatisticsPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
		NoiseKind:                    LaplaceNoise{},
	})

	// Assert
	want = beam.ParDo(s, pairIntMSToKMS, want)
	tolerance, err := testutils.LaplaceToleranceForMean(
		24, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed, epsilon, stat)
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	approxEqualsKMS(s, got, want, tolerance)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanStatisticsPerKeyNoNoiseInt: MeanStatisticsPerKey(%v) = %v, want %v, error %v",
			col, got, want, err)
	}
}

// Checks that MeanStatisticsPerKey with partitions returns a correct answer for float input values.
func TestMeanStatisticsPerKeyWithPartitionsNoNoiseFloat(t *testing.T) {
	for _, tc := range []struct {
		minValue float64
		maxValue float64
		inMemory bool
	}{
		{
			minValue: 1.0,
			maxValue: 3.0,
			inMemory: false,
		},
		{
			minValue: 1.0,
			maxValue: 3.0,
			inMemory: true,
		},
		{
			minValue: 0.0,
			maxValue: 2.0,
			inMemory: false,
		},
		{
			minValue: 0.0,
			maxValue: 2.0,
			inMemory: true,
		},
		{
			minValue: -10.0,
			maxValue: 10.0,
			inMemory: false,
		},
		{
			minValue: -10.0,
			maxValue: 10.0,
			inMemory: true,
		},
	} {
		// Arrange
		triples := testutils.ConcatenateTriplesWithFloatValue(
			testutils.MakeTripleWithFloatValue(7, 0, 2),
			testutils.MakeTripleWithFloatValueStartingFromKey(7, 100, 1, 1))

		stat := testutils.PerPartitionVarianceStatistics(tc.minValue, tc.maxValue, triples)[0]
		result := []pairIntMS{
			{K: 0, MS: testUtilsVSToPbeamMS(stat)},
			// Partition 1 will be dropped because it's not in the list of public partitions.
		}
		publicPartitionsSlice := []int{0}

		p, s, col, want := ptest.CreateList2(triples, result)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		// We have ε=50, δ=0 and l0Sensitivity=1. No thresholding is done because partitions are public.
		// We have 2 partitions. So, to get an overall flakiness of 10⁻²³,
		// we can have each partition fail with 10⁻²⁴ probability (k=24).
		maxContributionsPerPartition := int64(1)
		maxPartitionsContributed := int64(1)
		epsilon := 50.0

		// Act
		pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
		pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		meanParams := MeanParams{
			MaxPartitionsContributed:     maxPartitionsContributed,
			MaxContributionsPerPartition: maxContributionsPerPartition,
			MinValue:                     tc.minValue,
			MaxValue:                     tc.maxValue,
			NoiseKind:                    LaplaceNoise{},
			PublicPartitions:             publicPartitions,
		}
		got := MeanStatisticsPerKey(s, pcol, meanParams)

		// Assert
		want = beam.ParDo(s, pairIntMSToKMS, want)
		tolerance, err := testutils.LaplaceToleranceForMean(
			24, tc.minValue, tc.maxValue, maxContributionsPerPartition, maxPartitionsContributed,
			epsilon, stat)
		if err != nil {
			t.Fatalf("LaplaceToleranceForMean test case=%+v: got error %v", tc, err)
		}
		approxEqualsKMS(s, got, want, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestMeanStatisticsPerKeyWithPartitionsNoNoiseFloat test case=%+v: "+
				"MeanStatisticsPerKey(%v) = %v, want %v, error %v", tc, col, got, want, err)
		}
	}
}

// Checks that MeanStatisticsPerKey with public partitions returns a correct answer for int input values.
// They should be correctly converted to float64 and then correct result
// with float statistic should be computed.
func TestMeanStatisticsPerKeyWithPartitionsNoNoiseInt(t *testing.T) {
	for _, tc := range []struct {
		minValue float64
		maxValue float64
		inMemory bool
	}{
		{
			minValue: 1.0,
			maxValue: 3.0,
			inMemory: false,
		},
		{
			minValue: 1.0,
			maxValue: 3.0,
			inMemory: true,
		},
		{
			minValue: 0.0,
			maxValue: 2.0,
			inMemory: false,
		},
		{
			minValue: 0.0,
			maxValue: 2.0,
			inMemory: true,
		},
		{
			minValue: -10.0,
			maxValue: 10.0,
			inMemory: false,
		},
		{
			minValue: -10.0,
			maxValue: 10.0,
			inMemory: true,
		},
	} {
		triples := testutils.ConcatenateTriplesWithIntValue(
			testutils.MakeTripleWithIntValue(7, 0, 2),
			testutils.MakeTripleWithIntValueStartingFromKey(7, 100, 1, 1),
			testutils.MakeTripleWithIntValueStartingFromKey(107, 150, 1, 2),
		)
		stat := testutils.PerPartitionVarianceStatisticsInt(tc.minValue, tc.maxValue, triples)[1]

		// We have ε=50, δ=0 and l0Sensitivity=1.
		// We do not use thresholding because partitions are public.
		// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
		// we can have each partition fail with 10⁻²³ probability (k=23).
		maxContributionsPerPartition := int64(1)
		maxPartitionsContributed := int64(1)
		epsilon := 50.0

		result := []pairIntMS{
			// Partition 0 will be dropped because it's not in the list of public partitions.
			{K: 1, MS: testUtilsVSToPbeamMS(stat)},
		}
		publicPartitionsSlice := []int{1}

		p, s, col, want := ptest.CreateList2(triples, result)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
		pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
		meanParams := MeanParams{
			MaxPartitionsContributed:     maxPartitionsContributed,
			MaxContributionsPerPartition: maxContributionsPerPartition,
			MinValue:                     tc.minValue,
			MaxValue:                     tc.maxValue,
			NoiseKind:                    LaplaceNoise{},
			PublicPartitions:             publicPartitions,
		}
		got := MeanStatisticsPerKey(s, pcol, meanParams)
		want = beam.ParDo(s, pairIntMSToKMS, want)

		tolerance, err := testutils.LaplaceToleranceForMean(
			23, tc.minValue, tc.maxValue, maxContributionsPerPartition, maxPartitionsContributed,
			epsilon, stat)
		if err != nil {
			t.Fatalf("LaplaceToleranceForMean: test case=%+v got error %v", tc, err)
		}
		approxEqualsKMS(s, got, want, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestMeanStatisticsPerKeyWithPartitionsNoNoiseInt test case=%+v: "+
				"MeanStatisticsPerKey(%v) = %v, want %v, error %v", tc, col, got, want, err)
		}
	}
}

// Checks that MeanStatisticsPerKey does partition selection correctly by counting privacy IDs correctly,
// which means if the privacy unit has > 1 contributions to a partition the algorithm will not consider them as new privacy IDs.
func TestMeanStatisticsPerKeyCountsPrivacyUnitIDsWithMultipleContributionsCorrectly(t *testing.T) {
	triples := testutils.ConcatenateTriplesWithFloatValue(
		testutils.MakeTripleWithFloatValue(7, 0, 2.0),
		testutils.MakeTripleWithFloatValueStartingFromKey(7, 11, 1, 1.3),
		// We have a total of 42 contributions to partition 2, but privacy units with ID 18 and 19 contribute 21 times each.
		// So the actual count of privacy IDs in partition 2 is equal to 2, not 42.
		// And the threshold is equal to 11, so the partition 2 should be eliminated,
		// because the probability of keeping the partition with 2 elements is negligible, ≈5.184e-179.
		testutils.MakeTripleWithFloatValueStartingFromKey(18, 2, 2, 0))

	minValue := 1.0
	maxValue := 3.0
	// Only get statistics for partition 1, since both partition 0 and 2 will be dropped due to thresholding.
	stat := testutils.PerPartitionVarianceStatistics(minValue, maxValue, triples)[1]

	// Duplicated contribution to partition 2 from privacy unit 18 and 19, each duplicated 20 times.
	// While these duplicates are kept for computing the mean (due to maxContributionsPerPartition=20),
	// they are not considered for partition selection.
	// There are only two privacy units in partition 2, which is smaller than the threshold and should be dropped.
	for i := 0; i < 20; i++ {
		triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(18, 2, 2, 1)...)
	}

	// The threshold is equal to 11.
	// The probability to keep partition 0 (count = 7) is 1.942e-70.
	// The probability to keep partition 2 (count = 2) is 5.184e-179.
	// So only partition 1 is kept.
	result := []pairIntMS{
		{K: 1, MS: testUtilsVSToPbeamMS(stat)},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l0Sensitivity=1 gives a threshold of =11.
	// We have 2 partitions. So, to get an overall flakiness of 10⁻²³,
	// we can have each partition fail with 10⁻²⁴ probability (k=24).
	maxContributionsPerPartition := int64(20)
	maxPartitionsContributed := int64(1)
	epsilon := 50.0
	delta := 1e-200

	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	got := MeanStatisticsPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
		NoiseKind:                    LaplaceNoise{},
	})

	want = beam.ParDo(s, pairIntMSToKMS, want)
	tolerance, err := testutils.LaplaceToleranceForMean(
		24, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
		epsilon, stat)
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	approxEqualsKMS(s, got, want, tolerance)
	if err := ptest.Run(p); err != nil {
		t.Errorf("MeanStatisticsPerKey: for %v got %v, want %v, error %v", col, got, want, err)
	}
}

// Checks that MeanStatisticsPerKey applies partition selection.
func TestMeanStatisticsPerKeyPartitionSelection(t *testing.T) {
	testCases := []struct {
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
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Verify that entriesPerPartition is sensical.
			if tc.entriesPerPartition <= 0 {
				t.Fatalf("Invalid test case: entriesPerPartition must be positive. Got: %d",
					tc.entriesPerPartition)
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
					triples = append(triples, testutils.TripleWithFloatValue{
						ID: kOffset + j, Partition: i, Value: 1.0})
				}
				kOffset += tc.entriesPerPartition
			}
			p, s, col := ptest.CreateList(triples)
			col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

			// Run MeanStatisticsPerKey on triples
			pcol := MakePrivate(s, col, privacySpec(t,
				PrivacySpecParams{
					AggregationEpsilon:        tc.aggregationEpsilon,
					AggregationDelta:          tc.aggregationDelta,
					PartitionSelectionEpsilon: tc.partitionSelectionEpsilon,
					PartitionSelectionDelta:   tc.partitionSelectionDelta,
				}))
			pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
			got := MeanStatisticsPerKey(s, pcol, MeanParams{
				MinValue:                     0.0,
				MaxValue:                     1.0,
				MaxContributionsPerPartition: int64(tc.entriesPerPartition),
				MaxPartitionsContributed:     1,
				NoiseKind:                    tc.noiseKind,
			})
			got = beam.ParDo(s, extractMean, got)
			got = beam.ParDo(s, testutils.KVToPairIF64, got)

			// Validate that partition selection is applied (i.e., some emitted and some dropped).
			testutils.CheckSomePartitionsAreDropped(s, got, tc.numPartitions)
			if err := ptest.Run(p); err != nil {
				t.Errorf("%v", err)
			}
		})
	}
}

// Checks that MeanStatisticsPerKey works correctly for negative bounds and negative float values.
func TestMeanStatisticsKeyNegativeBounds(t *testing.T) {
	triples := testutils.ConcatenateTriplesWithFloatValue(
		testutils.MakeTripleWithFloatValue(100, 1, -5.0),
		testutils.MakeTripleWithFloatValueStartingFromKey(100, 150, 1, -1.0))

	minValue := -6.0
	maxValue := -2.0
	stat := testutils.PerPartitionVarianceStatistics(minValue, maxValue, triples)[1]
	result := []pairIntMS{
		{K: 1, MS: testUtilsVSToPbeamMS(stat)},
	}

	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

	// ε=50, δ=10⁻²⁰⁰ and l0Sensitivity=1 gives a threshold of =11.
	// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
	// we can have each partition fail with 10⁻²³ probability (k=23).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 50.0
	delta := 1e-200

	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	got := MeanStatisticsPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
		NoiseKind:                    LaplaceNoise{},
	})

	want = beam.ParDo(s, pairIntMSToKMS, want)
	tolerance, err := testutils.LaplaceToleranceForMean(
		23, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed, epsilon, stat)
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	approxEqualsKMS(s, got, want, tolerance)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanStatisticsPerKeyNegativeBounds: MeanStatisticsPerKey(%v) = %v, want %v, error %v",
			col, got, want, err)
	}
}

// Checks that MeanStatisticsPerKey does cross-partition contribution bounding correctly.
func TestMeanStatisticsPerKeyCrossPartitionContributionBounding(t *testing.T) {
	minValue := 0.0
	maxValue := 150.0
	var triples []testutils.TripleWithFloatValue

	// id 0 contributes to partition 0 and 1 with value 150.0.
	// ids [1, 4] each contributes to partition 0 with value 0.0.
	// ids [5, 8] each contributes to partition 1 with value 0.0.
	triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 150)...)

	triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(1, 4, 0, 0)...)
	triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(5, 4, 1, 0)...)

	statMap := testutils.PerPartitionVarianceStatistics(minValue, maxValue, triples)
	// stat1 represents the partition with an extra contribution with value 150.0.
	// stat2 represents the partition without an extra contribution.
	stat1, stat2 := statMap[0], statMap[1]
	triples = append(triples, testutils.MakeTripleWithFloatValue(1, 1, 150)...)

	// MaxPartitionContributed = 1, but id = 0 contributes to 2 partitions (0 and 1).
	// There will be cross-partition contribution bounding stage.
	// In this stage the algorithm will randomly keep either partition 0 or partition 1 for id 0.
	// The sum of 2 means should be equal to 150/5 + 0/4 = 30 in both cases
	// (unlike 150/5 + 150/5 = 60, if no cross-partition contribution bounding is done).
	// The difference between these numbers is 30 (60-30), and the sum of two tolerances (see below)
	// is ≈ 26.6277 (11.4598 + 15.1679),
	// so the test should fail if there was no cross-partition contribution bounding.
	result := []testutils.PairIF64{
		{Key: 0, Value: stat1.Mean + stat2.Mean},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

	// ε=10000, δ=0.01 and l0Sensitivity=1 gives a threshold of =2.
	// We have 2 partitions. So, to get an overall flakiness of 10⁻²³,
	// we can have each partition fail with 10⁻²⁴ probability (k=24).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 1e4
	delta := 0.01

	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	got := MeanStatisticsPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
		NoiseKind:                    LaplaceNoise{},
	})
	got = beam.ParDo(s, extractMean, got)

	means := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, means)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.

	want = beam.ParDo(s, testutils.PairIF64ToKV, want)

	// Tolerance for the partition with an extra contribution with value 150.0.
	tolerance1, err := testutils.LaplaceToleranceForMean(
		24, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
		epsilon, stat1) // ≈15.1679
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	// Tolerance for the partition without an extra contribution.
	tolerance2, err := testutils.LaplaceToleranceForMean(
		24, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
		epsilon, stat2) // ≈1.074
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}

	meanDiffTolerance := tolerance1.Mean + tolerance2.Mean
	testutils.ApproxEqualsKVFloat64(t, s, got, want, meanDiffTolerance)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanStatisticsPerKeyCrossPartitionContributionBounding: MeanStatisticsPerKey(%v) = %v, "+
			"want %v, error %v", col, got, want, err)
	}
}

// Checks that MeanStatisticsPerKey does per-partition contribution bounding correctly.
func TestMeanStatisticsPerKeyPerPartitionContributionBounding(t *testing.T) {
	minValue := 0.0
	maxValue := 100.0
	var triples []testutils.TripleWithFloatValue
	triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 50)...)
	triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(1, 50, 0, 0)...)

	// MaxContributionsPerPartition = 1, but id = 0 contributes 3 times to partition 0.
	// There will be per-partition contribution bounding stage.
	// In this stage the algorithm will randomly chose one of these 3 contributions.
	// The mean should be equal to 50/51 = 0.98 (not 150/53 ≈ 2.83, if no per-partition contribution bounding is done).
	// The difference between these numbers ≈ 1,85 and the tolerance (see below) is ≈0.92, so the test should catch if there was no per-partition contribution bounding.

	// stat represents the statistics result with per-partition contribution bonding.
	stat := testutils.PerPartitionVarianceStatistics(minValue, maxValue, triples)[0]
	result := []testutils.PairIF64{
		{Key: 0, Value: stat.Mean},
	}

	triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 50)...)
	triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 50)...)
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

	// ε=60, δ=0.01 and l0Sensitivity=1 gives a threshold of =2.
	// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
	// we can have that partition fail with 10⁻²³ probability (k=23).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 1000.0
	delta := 0.01

	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	got := MeanStatisticsPerKey(s, pcol, MeanParams{
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
		MaxPartitionsContributed:     1,
		NoiseKind:                    LaplaceNoise{},
	})
	got = beam.ParDo(s, extractMean, got)

	means := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, means)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.

	want = beam.ParDo(s, testutils.PairIF64ToKV, want)
	tolerance, err := testutils.LaplaceToleranceForMean(
		23, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
		epsilon, stat) // ≈0.92
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	testutils.ApproxEqualsKVFloat64(t, s, got, want, tolerance.Mean)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanStatisticsPerKeyPerPartitionContributionBounding: MeanStatisticsPerKey(%v) = %v, "+
			"want %v, error %v", col, got, want, err)
	}
}

// Make sure that the result values are in expected range. See checkMSsAreInRange for details.
func TestMeanStatisticsPerKeyReturnsSensibleValue(t *testing.T) {
	var triples []testutils.TripleWithFloatValue
	for key := 0; key < 100; key++ {
		triples = append(triples, testutils.TripleWithFloatValue{
			ID: key, Partition: key, Value: 0.})
	}
	testCases := []struct {
		name     string
		minValue float64
		maxValue float64
		epsilon  float64
		delta    float64
	}{
		{
			name:     "no negative noisy mean",
			minValue: 0.0,
			maxValue: 1e8,
			epsilon:  0.001, // ε=0.001. δ=0.999 and l0Sensitivity=1 gives a threshold of =2.
			delta:    0.999,
		},
		// The probability that any given partition has a negative noisy mean is 1/2 * 0.999.
		// The probability of none of the partitions having a noisy negative mean is 1 - (1/2 * 0.999)^100, which is negligible.
		{
			name:     "should have at least one negative noisy mean",
			minValue: -100.0,
			maxValue: 100.0,
			epsilon:  0.1, // ε=0.1. δ=0.999 and l0Sensitivity=1 gives a threshold of =2.
			delta:    0.999,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			p, s, col := ptest.CreateList(triples)
			col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

			maxContributionsPerPartition := int64(1)
			pcol := MakePrivate(s, col, privacySpec(t,
				PrivacySpecParams{
					AggregationEpsilon:        tc.epsilon,
					PartitionSelectionEpsilon: tc.epsilon,
					PartitionSelectionDelta:   tc.delta,
				}))
			pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
			got := MeanStatisticsPerKey(s, pcol, MeanParams{
				MaxContributionsPerPartition: maxContributionsPerPartition,
				MinValue:                     tc.minValue,
				MaxValue:                     tc.maxValue,
				MaxPartitionsContributed:     1,
				NoiseKind:                    LaplaceNoise{},
			})
			checkMSsAreInRange(s, got, tc.minValue, tc.maxValue)
			if err := ptest.Run(p); err != nil {
				t.Errorf("TestMeanStatisticsPerKeyReturnsSensibleValue returned error: %v", err)
			}
		})
	}
}

// Make sure that the result values are in expected range. See checkMSsAreInRange for details.
func TestMeanStatisticsPerKeyWithPartitionsReturnsSensibleValue(t *testing.T) {
	// We have two test cases, one for public partitions as a PCollection and one for public partitions as a slice (i.e., in-memory).
	for _, tc := range []struct {
		inMemory bool
	}{
		{true},
		{false},
	} {
		var triples []testutils.TripleWithFloatValue
		for key := 0; key < 100; key++ {
			triples = append(triples, testutils.TripleWithFloatValue{
				ID: key, Partition: key, Value: 0.01})
		}
		p, s, col := ptest.CreateList(triples)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		var publicPartitionsSlice []int
		for p := 0; p < 200; p++ {
			publicPartitionsSlice = append(publicPartitionsSlice, p)
		}
		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		// Using a low ε and a high maxValue to add a lot of noise.
		maxContributionsPerPartition := int64(1)
		epsilon := 0.001
		minValue := 0.0
		maxValue := 1e8

		pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
		pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		meanParams := MeanParams{
			MaxContributionsPerPartition: maxContributionsPerPartition,
			MinValue:                     minValue,
			MaxValue:                     maxValue,
			MaxPartitionsContributed:     1,
			NoiseKind:                    LaplaceNoise{},
			PublicPartitions:             publicPartitions,
		}
		got := MeanStatisticsPerKey(s, pcol, meanParams)
		checkMSsAreInRange(s, got, minValue, maxValue)
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestMeanStatisticsPerKeyWithPartitionsReturnsSensibleValue in-memory=%t "+
				"returned error: %v", tc.inMemory, err)
		}
	}
}

// Checks that MeanStatisticsPerKey with public partitions does cross-partition contribution bounding correctly.
func TestMeanStatisticsPerKeyWithPartitionsCrossPartitionContributionBounding(t *testing.T) {
	// We have two test cases, one for public partitions as a PCollection and one for public partitions as a slice (i.e., in-memory).
	for _, tc := range []struct {
		inMemory bool
	}{
		{true},
		{false},
	} {
		minValue := 0.0
		maxValue := 150.0

		// id 0 contributes to partition 0 and 1 with value 150.0.
		// ids [1, 4] each contributes to partition 0 with value 0.0.
		// ids [5, 8] each contributes to partition 1 with value 0.0.
		var triples []testutils.TripleWithFloatValue
		triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 150)...)
		triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(1, 4, 0, 0)...)
		triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(5, 8, 1, 0)...)

		statMap := testutils.PerPartitionVarianceStatistics(minValue, maxValue, triples)
		// stat1 represents the partition with an extra contribution of value 150.
		// stat2 represents the partition without the extra contribution.
		stat1, stat2 := statMap[0], statMap[1]
		triples = append(triples, testutils.MakeTripleWithFloatValue(1, 1, 150)...)

		// MaxPartitionContributed = 1, but id = 0 contributes to 2 partitions (0 and 1).
		// There will be cross-partition contribution bounding stage.
		// In this stage the algorithm the algorithm will randomly keep either partition 0 or partition 1 for id 0.
		// The sum of 2 means should be equal to 150/5 + 0/4 = 30 in both cases
		// (unlike 150/5 + 150/5 = 60, if no cross-partition contribution bounding is done).
		// The difference between these numbers is 30 (60-30), and the sum of two tolerances (see below)
		// is ≈ 26.6433 (11.4685 + 15.1748),
		// so the test should fail if there was no cross-partition contribution bounding.
		result := []testutils.PairIF64{
			{Key: 0, Value: stat1.Mean + stat2.Mean},
		}
		publicPartitionsSlice := []int{0, 1}

		p, s, col, want := ptest.CreateList2(triples, result)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		maxContributionsPerPartition := int64(1)
		maxPartitionsContributed := int64(1)
		epsilon := 1e4

		// ε is not split, because partitions are public.
		pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
		pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		meanParams := MeanParams{
			MaxPartitionsContributed:     maxPartitionsContributed,
			MaxContributionsPerPartition: maxContributionsPerPartition,
			MinValue:                     minValue,
			MaxValue:                     maxValue,
			NoiseKind:                    LaplaceNoise{},
			PublicPartitions:             publicPartitions,
		}
		got := MeanStatisticsPerKey(s, pcol, meanParams)
		got = beam.ParDo(s, extractMean, got)

		means := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, means)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.

		want = beam.ParDo(s, testutils.PairIF64ToKV, want)

		// Tolerance for the partition with an extra contribution which is equal to 150.
		tolerance1, err := testutils.LaplaceToleranceForMean(
			25, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
			epsilon, stat1) // ≈11.4685
		if err != nil {
			t.Fatalf("LaplaceToleranceForMean in-memory=%t: got error %v", tc.inMemory, err)
		}
		// Tolerance for the partition without an extra contribution.
		tolerance2, err := testutils.LaplaceToleranceForMean(
			25, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
			epsilon, stat2) // ≈15.1748
		if err != nil {
			t.Fatalf("LaplaceToleranceForMean in-memory=%t: got error %v", tc.inMemory, err)
		}
		meanDiffTolerance := tolerance1.Mean + tolerance2.Mean
		testutils.ApproxEqualsKVFloat64(t, s, got, want, meanDiffTolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestMeanStatisticsPerKeyWithPartitionsPerPartitionContributionBounding in-memory=%t: "+
				"MeanStatisticsPerKey(%v) = %v, want %v, error %v", tc.inMemory, col, got, want, err)
		}
	}
}

// Checks that MeanStatisticsPerKey with empty public partitions returns a correct answer.
func TestMeanStatisticsPerKeyWithEmptyPartitionsNoNoise(t *testing.T) {
	for _, tc := range []struct {
		minValue float64
		maxValue float64
		inMemory bool
	}{
		{
			minValue: 1.0,
			maxValue: 3.0,
			inMemory: false,
		},
		{
			minValue: 1.0,
			maxValue: 3.0,
			inMemory: true,
		},
		{
			minValue: 0.0,
			maxValue: 2.0,
			inMemory: false,
		},
		{
			minValue: 0.0,
			maxValue: 2.0,
			inMemory: true,
		},
		{
			minValue: -10.0,
			maxValue: 10.0,
			inMemory: false,
		},
		{
			minValue: -10.0,
			maxValue: 10.0,
			inMemory: true,
		},
	} {
		triples := testutils.MakeTripleWithIntValue(7, 0, 2)

		emptyPartitionStat := testutils.VarianceStatistics{
			Count:         0,
			NormalizedSum: 0,
		}
		emptyPartitionStat.ComputeMean(tc.minValue, tc.maxValue)
		pbeamEmptyPartitionStat := testUtilsVSToPbeamMS(emptyPartitionStat)

		// We have ε=50, δ=0 and l0Sensitivity=1.
		// We do not use thresholding because partitions are public.
		// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
		// we can have each partition fail with 10⁻²⁴ probability (k=24).
		maxContributionsPerPartition := int64(1)
		maxPartitionsContributed := int64(1)
		epsilon := 50.0

		result := []pairIntMS{
			{K: 1, MS: pbeamEmptyPartitionStat},
			{K: 2, MS: pbeamEmptyPartitionStat},
			{K: 3, MS: pbeamEmptyPartitionStat},
		}
		publicPartitionsSlice := []int{1, 2, 3}

		p, s, col, want := ptest.CreateList2(triples, result)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

		var publicPartitions any
		if tc.inMemory {
			publicPartitions = publicPartitionsSlice
		} else {
			publicPartitions = beam.CreateList(s, publicPartitionsSlice)
		}

		pcol := MakePrivate(s, col, privacySpec(t, PrivacySpecParams{AggregationEpsilon: epsilon}))
		pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
		meanParams := MeanParams{
			MaxPartitionsContributed:     maxPartitionsContributed,
			MaxContributionsPerPartition: maxContributionsPerPartition,
			MinValue:                     tc.minValue,
			MaxValue:                     tc.maxValue,
			NoiseKind:                    LaplaceNoise{},
			PublicPartitions:             publicPartitions,
		}
		got := MeanStatisticsPerKey(s, pcol, meanParams)
		want = beam.ParDo(s, pairIntMSToKMS, want)

		tolerance, err := testutils.LaplaceToleranceForMean(
			24, tc.minValue, tc.maxValue, maxContributionsPerPartition, maxPartitionsContributed,
			epsilon, emptyPartitionStat)
		if err != nil {
			t.Fatalf("LaplaceToleranceForMean test case=%+v: got error %v", tc, err)
		}
		approxEqualsKMS(s, got, want, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestMeanStatisticsPerKeyWithEmptyPartitionsNoNoise test case=%+v: MeanStatisticsPerKey(%v) = %v, "+
				"want %v, error %v", tc, col, got, want, err)
		}
	}
}

func TestCheckMeanPerKeyParams(t *testing.T) {
	_, _, publicPartitions := ptest.CreateList([]int{0, 1})
	for _, tc := range []struct {
		desc          string
		params        MeanParams
		noiseKind     noise.Kind
		partitionType reflect.Type
		wantErr       bool
	}{
		{
			desc: "valid parameters",
			params: MeanParams{
				AggregationEpsilon:           1.0,
				PartitionSelectionParams:     PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed:     1,
				MaxContributionsPerPartition: 1,
				MinValue:                     -5.0,
				MaxValue:                     5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       false,
		},
		{
			desc: "PartitionSelectionParams.MaxPartitionsContributed set",
			params: MeanParams{
				AggregationEpsilon:           1.0,
				PartitionSelectionParams:     PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5, MaxPartitionsContributed: 1},
				MaxPartitionsContributed:     1,
				MaxContributionsPerPartition: 1,
				MinValue:                     -5.0,
				MaxValue:                     5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "negative aggregationEpsilon",
			params: MeanParams{
				AggregationEpsilon:           -1.0,
				PartitionSelectionParams:     PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed:     1,
				MaxContributionsPerPartition: 1,
				MinValue:                     -5.0,
				MaxValue:                     5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "negative partitionSelectionEpsilon",
			params: MeanParams{
				AggregationEpsilon:           1.0,
				PartitionSelectionParams:     PartitionSelectionParams{Epsilon: -1.0, Delta: 1e-5},
				MaxPartitionsContributed:     1,
				MaxContributionsPerPartition: 1,
				MinValue:                     -5.0,
				MaxValue:                     5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "zero partitionSelectionDelta w/o public partitions",
			params: MeanParams{
				AggregationEpsilon:           1.0,
				PartitionSelectionParams:     PartitionSelectionParams{Epsilon: 1.0, Delta: 0},
				MaxPartitionsContributed:     1,
				MaxContributionsPerPartition: 1,
				MinValue:                     -5.0,
				MaxValue:                     5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "zero partitionSelectionEpsilon w/o public partitions",
			params: MeanParams{
				AggregationEpsilon:           1.0,
				PartitionSelectionParams:     PartitionSelectionParams{Epsilon: 0, Delta: 1e-5},
				MaxPartitionsContributed:     1,
				MaxContributionsPerPartition: 1,
				MinValue:                     -5.0,
				MaxValue:                     5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "MaxValue < MinValue",
			params: MeanParams{
				AggregationEpsilon:           1.0,
				PartitionSelectionParams:     PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed:     1,
				MaxContributionsPerPartition: 1,
				MinValue:                     6.0,
				MaxValue:                     5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "MaxValue = MinValue",
			params: MeanParams{
				AggregationEpsilon:           1.0,
				PartitionSelectionParams:     PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxPartitionsContributed:     1,
				MaxContributionsPerPartition: 1,
				MinValue:                     5.0,
				MaxValue:                     5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "zero MaxContributionsPerPartition",
			params: MeanParams{
				AggregationEpsilon:       1.0,
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
			desc: "zero MaxPartitionsContributed",
			params: MeanParams{
				AggregationEpsilon:           1.0,
				PartitionSelectionParams:     PartitionSelectionParams{Epsilon: 1.0, Delta: 1e-5},
				MaxContributionsPerPartition: 1,
				MinValue:                     -5.0,
				MaxValue:                     5.0,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: nil,
			wantErr:       true,
		},
		{
			desc: "non-zero partitionSelectionDelta w/ public partitions",
			params: MeanParams{
				AggregationEpsilon:           1.0,
				PartitionSelectionParams:     PartitionSelectionParams{Epsilon: 0, Delta: 1e-5},
				MaxPartitionsContributed:     1,
				MaxContributionsPerPartition: 1,
				MinValue:                     -5.0,
				MaxValue:                     5.0,
				PublicPartitions:             publicPartitions,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(0),
			wantErr:       true,
		},
		{
			desc: "non-zero partitionSelectionEpsilon w/ public partitions",
			params: MeanParams{
				AggregationEpsilon:           1.0,
				PartitionSelectionParams:     PartitionSelectionParams{Epsilon: 1.0, Delta: 0},
				MaxPartitionsContributed:     1,
				MaxContributionsPerPartition: 1,
				MinValue:                     -5.0,
				MaxValue:                     5.0,
				PublicPartitions:             publicPartitions,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(0),
			wantErr:       true,
		},
		{
			desc: "wrong partition type w/ public partitions as beam.PCollection",
			params: MeanParams{
				AggregationEpsilon:           1.0,
				MaxPartitionsContributed:     1,
				MaxContributionsPerPartition: 1,
				MinValue:                     -5.0,
				MaxValue:                     5.0,
				PublicPartitions:             publicPartitions,
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
		{
			desc: "wrong partition type w/ public partitions as slice",
			params: MeanParams{
				AggregationEpsilon:           1.0,
				MaxPartitionsContributed:     1,
				MaxContributionsPerPartition: 1,
				MinValue:                     -5.0,
				MaxValue:                     5.0,
				PublicPartitions:             []int{0},
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
		{
			desc: "wrong partition type w/ public partitions as array",
			params: MeanParams{
				AggregationEpsilon:           1.0,
				MaxPartitionsContributed:     1,
				MaxContributionsPerPartition: 1,
				MinValue:                     -5.0,
				MaxValue:                     5.0,
				PublicPartitions:             [1]int{0},
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
		{
			desc: "public partitions as something other than beam.PCollection, slice or array",
			params: MeanParams{
				AggregationEpsilon:           1.0,
				MaxPartitionsContributed:     1,
				MaxContributionsPerPartition: 1,
				MinValue:                     -5.0,
				MaxValue:                     5.0,
				PublicPartitions:             "",
			},
			noiseKind:     noise.LaplaceNoise,
			partitionType: reflect.TypeOf(""),
			wantErr:       true,
		},
	} {
		if err := checkMeanPerKeyParams(tc.params, tc.noiseKind, tc.partitionType); (err != nil) != tc.wantErr {
			t.Errorf("With %s, got=%v, wantErr=%t", tc.desc, err, tc.wantErr)
		}
	}
}
