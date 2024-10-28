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
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

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
	exactSum := 6.0
	exactCount := 2.0
	exactMean := exactSum / exactCount
	want := testutils.Float64Ptr(exactMean)
	tolerance, err := testutils.LaplaceToleranceForMean(23, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed, epsilon, 1, exactCount, exactMean)
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	if !cmp.Equal(want, got, cmpopts.EquateApprox(0, tolerance)) {
		t.Errorf("AddInput: for 2 values got: %f, want %f", *got, *want)
	}
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
	exactSum := 10.0
	exactCount := 4.0
	exactMean := exactSum / exactCount
	want := testutils.Float64Ptr(exactMean)
	tolerance, err := testutils.LaplaceToleranceForMean(
		23, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
		epsilon, 0, exactCount, exactMean)
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	if !cmp.Equal(want, got, cmpopts.EquateApprox(0, tolerance)) {
		t.Errorf("MergeAccumulators: when merging 2 instances of boundedMeanAccum got: %f, want %f",
			*got, *want)
	}
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

// Checks that MeanPerKey adds noise to its output.
func TestMeanPerKeyAddsNoise(t *testing.T) {
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
		got := MeanPerKey(s, pcol, MeanParams{
			MaxPartitionsContributed:     maxPartitionsContributed,
			MaxContributionsPerPartition: maxContributionsPerPartition,
			MinValue:                     minValue,
			MaxValue:                     maxValue,
			NoiseKind:                    tc.noiseKind,
		})
		got = beam.ParDo(s, testutils.KVToPairIF64, got)

		// We check that any noise is added, hence tolerance is 0.0.
		// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf.
		testutils.CheckFloat64MetricsAreNoisy(s, got, 1.0, 0.0)
		if err := ptest.Run(p); err != nil {
			t.Errorf("MeanPerKey didn't add any noise with float inputs and %s Noise: %v", tc.name, err)
		}
	}
}

// Checks that MeanPerKey with partitions adds noise to its output.
func TestMeanPerKeyWithPartitionsAddsNoise(t *testing.T) {
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
			epsilon:   1,
			delta:     0.005,
			inMemory:  false,
		},
		{
			desc:      "as slice w/ Gaussian",
			noiseKind: GaussianNoise{},
			epsilon:   1,
			delta:     0.005,
			inMemory:  true,
		},
		{
			desc:      "as PCollection w/ Laplace",
			noiseKind: LaplaceNoise{},
			epsilon:   0.1,
			delta:     0, // It is 0 because partitions are public and we are using Laplace noise.
			inMemory:  false,
		},
		{
			desc:      "as PCollection w/ Laplace",
			noiseKind: LaplaceNoise{},
			epsilon:   0.1,
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
		got := MeanPerKey(s, pcol, meanParams)
		got = beam.ParDo(s, testutils.KVToPairIF64, got)

		// We check that any noise is added, hence tolerance is 0.0.
		// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf.
		testutils.CheckFloat64MetricsAreNoisy(s, got, 1.0, 0.0)
		if err := ptest.Run(p); err != nil {
			t.Errorf("MeanPerKey with partitions %s didn't add any noise with float inputs: %v", tc.desc, err)
		}
	}
}

// Checks that MeanPerKey returns a correct answer for float input values.
func TestMeanPerKeyNoNoiseFloat(t *testing.T) {
	// Arrange
	triples := testutils.ConcatenateTriplesWithFloatValue(
		testutils.MakeTripleWithFloatValue(7, 0, 2.0),
		testutils.MakeTripleWithFloatValueStartingFromKey(7, 100, 1, 1.3),
		testutils.MakeTripleWithFloatValueStartingFromKey(107, 150, 1, 2.5))

	exactCount := 250.0
	exactMean := (1.3*100 + 2.5*150) / exactCount
	result := []testutils.PairIF64{
		{Key: 1, Value: exactMean},
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
	minValue := 1.0
	maxValue := 3.0

	// Act
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	got := MeanPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
		NoiseKind:                    LaplaceNoise{},
	})

	// Assert
	want = beam.ParDo(s, testutils.PairIF64ToKV, want)
	tolerance, err := testutils.LaplaceToleranceForMean(24, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed, epsilon, 5, exactCount, exactMean)
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	testutils.ApproxEqualsKVFloat64(t, s, got, want, tolerance)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanPerKeyNoNoiseFloat: MeanPerKey(%v) = %v, want %v, error %v", col, got, want, err)
	}
}

// Checks that MeanPerKey returns a correct answer for int input values.
// They should be correctly converted to float64 and then correct result
// with float statistic should be computed.
func TestMeanPerKeyNoNoiseInt(t *testing.T) {
	// Arrange
	triples := testutils.ConcatenateTriplesWithIntValue(
		testutils.MakeTripleWithIntValue(7, 0, 2),
		testutils.MakeTripleWithIntValueStartingFromKey(7, 100, 1, 1),
		testutils.MakeTripleWithIntValueStartingFromKey(107, 150, 1, 2))

	exactCount := 250.0
	exactMean := (100.0 + 2.0*150.0) / exactCount
	result := []testutils.PairIF64{
		{Key: 1, Value: exactMean},
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
	minValue := 0.0
	maxValue := 2.0

	// Act
	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithIntValueToKV, pcol)
	got := MeanPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
		NoiseKind:                    LaplaceNoise{},
	})

	// Assert
	want = beam.ParDo(s, testutils.PairIF64ToKV, want)
	tolerance, err := testutils.LaplaceToleranceForMean(
		24, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
		epsilon, 150.0, exactCount, exactMean)
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	testutils.ApproxEqualsKVFloat64(t, s, got, want, tolerance)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanPerKeyNoNoiseInt: MeanPerKey(%v) = %v, want %v, error %v",
			col, got, want, err)
	}
}

// Checks that MeanPerKey with partitions returns a correct answer for float input values.
func TestMeanPerKeyWithPartitionsNoNoiseFloat(t *testing.T) {
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

		exactCount := 7.0
		exactMean := 14.0 / exactCount
		result := []testutils.PairIF64{
			{Key: 0, Value: exactMean},
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
		got := MeanPerKey(s, pcol, meanParams)

		// Assert
		want = beam.ParDo(s, testutils.PairIF64ToKV, want)
		exactNormalizedSum := (2.0 - (tc.maxValue+tc.minValue)/2) * exactCount
		tolerance, err := testutils.LaplaceToleranceForMean(
			24, tc.minValue, tc.maxValue, maxContributionsPerPartition, maxPartitionsContributed,
			epsilon, exactNormalizedSum, exactCount, exactMean)
		if err != nil {
			t.Fatalf("LaplaceToleranceForMean test case=%+v: got error %v", tc, err)
		}
		testutils.ApproxEqualsKVFloat64(t, s, got, want, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestMeanPerKeyWithPartitionsNoNoiseFloat test case=%+v: "+
				"MeanPerKey(%v) = %v, want %v, error %v", tc, col, got, want, err)
		}
	}
}

// Checks that MeanPerKey with public partitions returns a correct answer for int input values.
// They should be correctly converted to float64 and then correct result
// with float statistic should be computed.
func TestMeanPerKeyWithPartitionsNoNoiseInt(t *testing.T) {
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

		exactCount := 250.0
		exactMean := (100.0 + 2.0*150.0) / exactCount

		// We have ε=50, δ=0 and l0Sensitivity=1.
		// We do not use thresholding because partitions are public.
		// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
		// we can have each partition fail with 10⁻²³ probability (k=23).
		maxContributionsPerPartition := int64(1)
		maxPartitionsContributed := int64(1)
		epsilon := 50.0

		result := []testutils.PairIF64{
			// Partition 0 will be dropped because it's not in the list of public partitions.
			{Key: 1, Value: exactMean},
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
		got := MeanPerKey(s, pcol, meanParams)
		want = beam.ParDo(s, testutils.PairIF64ToKV, want)

		exactNormalizedSum := (1.0-(tc.maxValue+tc.minValue)/2)*100 + (2.0-(tc.maxValue+tc.minValue)/2)*150
		tolerance, err := testutils.LaplaceToleranceForMean(
			23, tc.minValue, tc.maxValue, maxContributionsPerPartition, maxPartitionsContributed,
			epsilon, exactNormalizedSum, exactCount, exactMean)
		if err != nil {
			t.Fatalf("LaplaceToleranceForMean: test case=%+v got error %v", tc, err)
		}
		testutils.ApproxEqualsKVFloat64(t, s, got, want, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestMeanPerKeyWithPartitionsNoNoiseInt test case=%+v: "+
				"MeanPerKey(%v) = %v, want %v, error %v", tc, col, got, want, err)
		}
	}
}

// Checks that MeanPerKey does partition selection correctly by counting privacy IDs correctly,
// which means if the privacy unit has > 1 contributions to a partition the algorithm will not consider them as new privacy IDs.
func TestMeanPerKeyCountsPrivacyUnitIDsWithMultipleContributionsCorrectly(t *testing.T) {
	triples := testutils.ConcatenateTriplesWithFloatValue(
		testutils.MakeTripleWithFloatValue(7, 0, 2.0),
		testutils.MakeTripleWithFloatValueStartingFromKey(7, 11, 1, 1.3),
		// We have a total of 42 contributions to partition 2, but privacy units with ID 18 and 19 contribute 21 times each.
		// So the actual count of privacy IDs in partition 2 is equal to 2, not 42.
		// And the threshold is equal to 11, so the partition 2 should be eliminated,
		// because the probability of keeping the partition with 2 elements is negligible, ≈5.184e-179.
		testutils.MakeTripleWithFloatValueStartingFromKey(18, 2, 2, 0))

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
	exactCount := 11.0
	exactMean := 1.3
	result := []testutils.PairIF64{
		{Key: 1, Value: exactMean},
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
	minValue := 1.0
	maxValue := 3.0

	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	got := MeanPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
		NoiseKind:                    LaplaceNoise{},
	})

	want = beam.ParDo(s, testutils.PairIF64ToKV, want)
	tolerance, err := testutils.LaplaceToleranceForMean(
		24, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
		epsilon, -7.7, exactCount, exactMean)
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	testutils.ApproxEqualsKVFloat64(t, s, got, want, tolerance)
	if err := ptest.Run(p); err != nil {
		t.Errorf("MeanPerKey: for %v got %v, want %v, error %v", col, got, want, err)
	}
}

var meanPartitionSelectionTestCases = []struct {
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

// Checks that MeanPerKey applies partition selection.
func TestMeanPartitionSelection(t *testing.T) {
	for _, tc := range meanPartitionSelectionTestCases {
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

			// Run MeanPerKey on triples
			pcol := MakePrivate(s, col, privacySpec(t,
				PrivacySpecParams{
					AggregationEpsilon:        tc.aggregationEpsilon,
					AggregationDelta:          tc.aggregationDelta,
					PartitionSelectionEpsilon: tc.partitionSelectionEpsilon,
					PartitionSelectionDelta:   tc.partitionSelectionDelta,
				}))
			pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
			got := MeanPerKey(s, pcol, MeanParams{
				MinValue:                     0.0,
				MaxValue:                     1.0,
				MaxContributionsPerPartition: int64(tc.entriesPerPartition),
				MaxPartitionsContributed:     1,
				NoiseKind:                    tc.noiseKind,
			})
			got = beam.ParDo(s, testutils.KVToPairIF64, got)

			// Validate that partition selection is applied (i.e., some emitted and some dropped).
			testutils.CheckSomePartitionsAreDropped(s, got, tc.numPartitions)
			if err := ptest.Run(p); err != nil {
				t.Errorf("%v", err)
			}
		})
	}
}

// Checks that MeanPerKey works correctly for negative bounds and negative float values.
func TestMeanKeyNegativeBounds(t *testing.T) {
	triples := testutils.ConcatenateTriplesWithFloatValue(
		testutils.MakeTripleWithFloatValue(100, 1, -5.0),
		testutils.MakeTripleWithFloatValueStartingFromKey(100, 150, 1, -1.0))

	exactCount := 250.0
	exactMean := (-5.0*100 - 2.0*150) / exactCount // 1.0 is clamped down to -2.0
	result := []testutils.PairIF64{
		{Key: 1, Value: exactMean},
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
	minValue := -6.0
	maxValue := -2.0

	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	got := MeanPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
		NoiseKind:                    LaplaceNoise{},
	})

	want = beam.ParDo(s, testutils.PairIF64ToKV, want)
	tolerance, err := testutils.LaplaceToleranceForMean(
		23, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
		epsilon, 200.0, exactCount, exactMean)
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	testutils.ApproxEqualsKVFloat64(t, s, got, want, tolerance)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanPerKeyNegativeBounds: MeanPerKey(%v) = %v, want %v, error %v",
			col, got, want, err)
	}
}

// Checks that MeanPerKey does cross-partition contribution bounding correctly.
func TestMeanPerKeyCrossPartitionContributionBounding(t *testing.T) {
	var triples []testutils.TripleWithFloatValue

	// id 0 contributes to partition 0 and 1 with value 150.0.
	// ids [1, 4] each contributes to partition 0 with value 0.0.
	// ids [5, 8] each contributes to partition 1 with value 0.0.
	triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 150)...)
	triples = append(triples, testutils.MakeTripleWithFloatValue(1, 1, 150)...)

	triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(1, 4, 0, 0)...)
	triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(5, 4, 1, 0)...)

	// MaxPartitionContributed = 1, but id = 0 contributes to 2 partitions (0 and 1).
	// There will be cross-partition contribution bounding stage.
	// In this stage the algorithm will randomly keep either partition 0 or partition 1 for id 0.
	// The sum of 2 means should be equal to 150/5 + 0/4 = 30 in both cases
	// (unlike 150/5 + 150/5 = 60, if no cross-partition contribution bounding is done).
	// The difference between these numbers is 30 (60-30), and the sum of two tolerances (see below)
	// is ≈ 26.6277 (11.4598 + 15.1679),
	// so the test should fail if there was no cross-partition contribution bounding.
	minValue := 0.0
	maxValue := 150.0
	midValue := (minValue + maxValue) / 2
	count2 := 4.0
	count1 := count2 + 1
	normalizedSum2 := (0.0 - midValue) * count2
	normalizedSum1 := normalizedSum2 + (150.0 - midValue)
	normalizedMean2 := normalizedSum2 / count2
	normalizedMean1 := normalizedSum1 / count1
	mean1, mean2 := normalizedMean1+midValue, normalizedMean2+midValue
	result := []testutils.PairIF64{
		{Key: 0, Value: mean1 + mean2},
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
	got := MeanPerKey(s, pcol, MeanParams{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
		NoiseKind:                    LaplaceNoise{},
	})

	means := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, means)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.

	want = beam.ParDo(s, testutils.PairIF64ToKV, want)

	// Tolerance for the partition with an extra contribution with value 150.0.
	tolerance1, err := testutils.LaplaceToleranceForMean(
		24, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
		epsilon, normalizedSum1, count1, mean1) // ≈15.1679
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	// Tolerance for the partition without an extra contribution.
	tolerance2, err := testutils.LaplaceToleranceForMean(
		24, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
		epsilon, normalizedSum2, count2, mean2) // ≈1.074
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}

	testutils.ApproxEqualsKVFloat64(t, s, got, want, tolerance1+tolerance2)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanPerKeyCrossPartitionContributionBounding: MeanPerKey(%v) = %v, "+
			"want %v, error %v", col, got, want, err)
	}
}

// Checks that MeanPerKey does per-partition contribution bounding correctly.
func TestMeanPerKeyPerPartitionContributionBounding(t *testing.T) {
	var triples []testutils.TripleWithFloatValue
	triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 50)...)
	triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 50)...)
	triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 50)...)
	triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(1, 50, 0, 0)...)

	// MaxContributionsPerPartition = 1, but id = 0 contributes 3 times to partition 0.
	// There will be per-partition contribution bounding stage.
	// In this stage the algorithm will randomly chose one of these 3 contributions.
	// The mean should be equal to 50/51 = 0.98 (not 150/53 ≈ 2.83, if no per-partition contribution bounding is done).
	// The difference between these numbers ≈ 1,85 and the tolerance (see below) is ≈0.92, so the test should catch if there was no per-partition contribution bounding.
	exactCount := 51.0
	exactMean := 50.0 / exactCount
	result := []testutils.PairIF64{
		{Key: 0, Value: exactMean},
	}

	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

	// ε=60, δ=0.01 and l0Sensitivity=1 gives a threshold of =2.
	// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
	// we can have that partition fail with 10⁻²³ probability (k=23).
	maxContributionsPerPartition := int64(1)
	maxPartitionsContributed := int64(1)
	epsilon := 1000.0
	delta := 0.01
	minValue := 0.0
	maxValue := 100.0

	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	got := MeanPerKey(s, pcol, MeanParams{
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
		MaxPartitionsContributed:     1,
		NoiseKind:                    LaplaceNoise{},
	})

	means := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, means)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.

	want = beam.ParDo(s, testutils.PairIF64ToKV, want)
	tolerance, err := testutils.LaplaceToleranceForMean(
		23, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
		epsilon, -2500.0, exactCount, exactMean) // ≈0.92
	if err != nil {
		t.Fatalf("LaplaceToleranceForMean: got error %v", err)
	}
	testutils.ApproxEqualsKVFloat64(t, s, got, want, tolerance)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanPerKeyPerPartitionContributionBounding: MeanPerKey(%v) = %v, "+
			"want %v, error %v", col, got, want, err)
	}
}

// Expect non-negative results if MinValue >= 0.
func TestMeanPerKeyReturnsNonNegative(t *testing.T) {
	var triples []testutils.TripleWithFloatValue
	for key := 0; key < 100; key++ {
		triples = append(triples, testutils.TripleWithFloatValue{
			ID: key, Partition: key, Value: 0.01})
	}
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)
	// Using a low ε, a high δ, and a high maxValue here to add a
	// lot of noise while keeping partitions.
	// ε=0.001. δ=0.999 and l0Sensitivity=1 gives a threshold of =2.
	maxContributionsPerPartition := int64(1)
	epsilon := 0.001
	delta := 0.999
	minValue := 0.0
	maxValue := 1e8

	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	means := MeanPerKey(s, pcol, MeanParams{
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
		MaxPartitionsContributed:     1,
		NoiseKind:                    LaplaceNoise{},
	})
	values := beam.DropKey(s, means)
	beam.ParDo0(s, testutils.CheckNoNegativeValuesFloat64, values)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanPerKeyReturnsNonNegativeFloat64 returned errors: %v", err)
	}
}

// Expect non-negative results with partitions if MinValue >= 0.
func TestMeanPerKeyWithPartitionsReturnsNonNegative(t *testing.T) {
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
		means := MeanPerKey(s, pcol, meanParams)
		values := beam.DropKey(s, means)
		beam.ParDo0(s, testutils.CheckNoNegativeValuesFloat64, values)
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestMeanPerKeyWithPartitionsReturnsNonNegativeFloat64 in-memory=%t "+
				"returned errors: %v", tc.inMemory, err)
		}
	}
}

// Expect at least one negative value after post-aggregation clamping when MinValue < 0.
func TestMeanPerKeyNoClampingForNegativeMinValue(t *testing.T) {
	var triples []testutils.TripleWithFloatValue
	// The probability that any given partition has a negative noisy mean is 1/2 * 0.999.
	// The probability of none of the partitions having a noisy negative mean is 1 - (1/2 * 0.999)^100, which is negligible.
	for key := 0; key < 100; key++ {
		triples = append(triples, testutils.TripleWithFloatValue{
			ID: key, Partition: key, Value: 0})
	}
	p, s, col := ptest.CreateList(triples)
	col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

	// ε=0.1. δ=0.999 and l0Sensitivity=1 gives a threshold of =2.
	maxContributionsPerPartition := int64(1)
	epsilon := 0.1
	delta := 0.999
	minValue := -100.0
	maxValue := 100.0

	pcol := MakePrivate(s, col, privacySpec(t,
		PrivacySpecParams{
			AggregationEpsilon:        epsilon,
			PartitionSelectionEpsilon: epsilon,
			PartitionSelectionDelta:   delta,
		}))
	pcol = ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
	means := MeanPerKey(s, pcol, MeanParams{
		MaxContributionsPerPartition: maxContributionsPerPartition,
		MinValue:                     minValue,
		MaxValue:                     maxValue,
		MaxPartitionsContributed:     1,
		NoiseKind:                    LaplaceNoise{},
	})
	values := beam.DropKey(s, means)
	mValue := stats.Min(s, values)
	beam.ParDo0(s, testutils.CheckAllValuesNegativeFloat64, mValue)
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestMeanPerKeyNoClampingForNegativeMinValueFloat64 returned errors: %v", err)
	}
}

// Checks that MeanPerKey with public partitions does cross-partition contribution bounding correctly.
func TestMeanPerKeyWithPartitionsCrossPartitionContributionBounding(t *testing.T) {
	// We have two test cases, one for public partitions as a PCollection and one for public partitions as a slice (i.e., in-memory).
	for _, tc := range []struct {
		inMemory bool
	}{
		{true},
		{false},
	} {
		// id 0 contributes to partition 0 and 1 with value 150.0.
		// ids [1, 4] each contributes to partition 0 with value 0.0.
		// ids [5, 8] each contributes to partition 1 with value 0.0.
		var triples []testutils.TripleWithFloatValue
		triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 150)...)
		triples = append(triples, testutils.MakeTripleWithFloatValue(1, 1, 150)...)
		triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(1, 4, 0, 0)...)
		triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(5, 8, 1, 0)...)

		// MaxPartitionContributed = 1, but id = 0 contributes to 2 partitions (0 and 1).
		// There will be cross-partition contribution bounding stage.
		// In this stage the algorithm the algorithm will randomly keep either partition 0 or partition 1 for id 0.
		// The sum of 2 means should be equal to 150/5 + 0/4 = 30 in both cases
		// (unlike 150/5 + 150/5 = 60, if no cross-partition contribution bounding is done).
		// The difference between these numbers is 30 (60-30), and the sum of two tolerances (see below)
		// is ≈ 26.6433 (11.4685 + 15.1748),
		// so the test should fail if there was no cross-partition contribution bounding.
		minValue := 0.0
		maxValue := 150.0
		midValue := (minValue + maxValue) / 2
		count2 := 4.0
		count1 := count2 + 1
		normalizedSum2 := (0.0 - midValue) * count2
		normalizedSum1 := normalizedSum2 + (150.0 - midValue)
		normalizedMean2 := normalizedSum2 / count2
		normalizedMean1 := normalizedSum1 / count1
		mean1, mean2 := normalizedMean1+midValue, normalizedMean2+midValue
		result := []testutils.PairIF64{
			{Key: 0, Value: mean1 + mean2},
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
		got := MeanPerKey(s, pcol, meanParams)

		means := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, means)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.

		want = beam.ParDo(s, testutils.PairIF64ToKV, want)

		// Tolerance for the partition with an extra contribution which is equal to 150.
		tolerance1, err := testutils.LaplaceToleranceForMean(
			25, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
			epsilon, normalizedSum1, count1, mean1) // ≈11.4685
		if err != nil {
			t.Fatalf("LaplaceToleranceForMean in-memory=%t: got error %v", tc.inMemory, err)
		}
		// Tolerance for the partition without an extra contribution.
		tolerance2, err := testutils.LaplaceToleranceForMean(
			25, minValue, maxValue, maxContributionsPerPartition, maxPartitionsContributed,
			epsilon, normalizedSum2, count2, mean2) // ≈15.1748
		if err != nil {
			t.Fatalf("LaplaceToleranceForMean in-memory=%t: got error %v", tc.inMemory, err)
		}
		testutils.ApproxEqualsKVFloat64(t, s, got, want, tolerance1+tolerance2)
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestMeanPerKeyWithPartitionsPerPartitionContributionBounding in-memory=%t: "+
				"MeanPerKey(%v) = %v, want %v, error %v", tc.inMemory, col, got, want, err)
		}
	}
}

// Checks that MeanPerKey with empty public partitions returns a correct answer.
func TestMeanPerKeyWithEmptyPartitionsNoNoise(t *testing.T) {
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

		midpoint := tc.minValue + (tc.maxValue-tc.minValue)/2.0
		exactCount := 0.0
		exactMean := midpoint // Mean of 0 elements is midpoint.

		// We have ε=50, δ=0 and l0Sensitivity=1.
		// We do not use thresholding because partitions are public.
		// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
		// we can have each partition fail with 10⁻²⁴ probability (k=24).
		maxContributionsPerPartition := int64(1)
		maxPartitionsContributed := int64(1)
		epsilon := 50.0

		result := []testutils.PairIF64{
			{Key: 1, Value: midpoint},
			{Key: 2, Value: midpoint},
			{Key: 3, Value: midpoint},
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
		got := MeanPerKey(s, pcol, meanParams)
		want = beam.ParDo(s, testutils.PairIF64ToKV, want)

		tolerance, err := testutils.LaplaceToleranceForMean(
			24, tc.minValue, tc.maxValue, maxContributionsPerPartition, maxPartitionsContributed,
			epsilon, 0.0, exactCount, exactMean)
		if err != nil {
			t.Fatalf("LaplaceToleranceForMean test case=%+v: got error %v", tc, err)
		}
		testutils.ApproxEqualsKVFloat64(t, s, got, want, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("TestMeanPerKeyWithEmptyPartitionsNoNoise test case=%+v: MeanPerKey(%v) = %v, "+
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
