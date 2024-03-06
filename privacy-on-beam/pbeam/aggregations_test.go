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

	"github.com/google/differential-privacy/go/v3/noise"
	"github.com/google/differential-privacy/privacy-on-beam/v3/pbeam/testutils"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/core/typex"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestNewBoundedSumFn(t *testing.T) {
	opts := []cmp.Option{
		cmpopts.EquateApprox(0, 1e-10),
		cmpopts.IgnoreUnexported(boundedSumFloat64Fn{}, boundedSumInt64Fn{}),
	}
	for _, tc := range []struct {
		desc                      string
		noiseKind                 noise.Kind
		vKind                     reflect.Kind
		aggregationEpsilon        float64
		aggregationDelta          float64
		partitionSelectionEpsilon float64
		partitionSelectionDelta   float64
		preThreshold              int64
		lower                     float64
		upper                     float64
		wantErr                   bool
		want                      any
	}{
		{"Laplace Float64", noise.LaplaceNoise, reflect.Float64, 0.5, 0, 0.5, 1e-5, 0, 0, 10, false,
			&boundedSumFloat64Fn{
				NoiseEpsilon:              0.5,
				NoiseDelta:                0,
				PartitionSelectionEpsilon: 0.5,
				PartitionSelectionDelta:   1e-5,
				MaxPartitionsContributed:  17,
				Lower:                     0,
				Upper:                     10,
				NoiseKind:                 noise.LaplaceNoise,
				PublicPartitions:          false,
			}},
		{"Gaussian Float64", noise.GaussianNoise, reflect.Float64, 0.5, 1e-5, 0.5, 1e-5, 0, 0, 10, false,
			&boundedSumFloat64Fn{
				NoiseEpsilon:              0.5,
				NoiseDelta:                1e-5,
				PartitionSelectionEpsilon: 0.5,
				PartitionSelectionDelta:   1e-5,
				MaxPartitionsContributed:  17,
				Lower:                     0,
				Upper:                     10,
				NoiseKind:                 noise.GaussianNoise,
				PublicPartitions:          false,
			}},
		{"Laplace Int64", noise.LaplaceNoise, reflect.Int64, 0.5, 0, 0.5, 1e-5, 0, 0, 10, false,
			&boundedSumInt64Fn{
				NoiseEpsilon:              0.5,
				NoiseDelta:                0,
				PartitionSelectionEpsilon: 0.5,
				PartitionSelectionDelta:   1e-5,
				MaxPartitionsContributed:  17,
				Lower:                     0,
				Upper:                     10,
				NoiseKind:                 noise.LaplaceNoise,
				PublicPartitions:          false,
			}},
		{"Gaussian Int64", noise.GaussianNoise, reflect.Int64, 0.5, 1e-5, 0.5, 1e-5, 0, 0, 10, false,
			&boundedSumInt64Fn{
				NoiseEpsilon:              0.5,
				NoiseDelta:                1e-5,
				PartitionSelectionEpsilon: 0.5,
				PartitionSelectionDelta:   1e-5,
				MaxPartitionsContributed:  17,
				Lower:                     0,
				Upper:                     10,
				NoiseKind:                 noise.GaussianNoise,
				PublicPartitions:          false,
			}},
		{"PreThreshold set Int64", noise.GaussianNoise, reflect.Int64, 0.5, 1e-5, 0.5, 1e-5, 10, 0, 10, false,
			&boundedSumInt64Fn{
				NoiseEpsilon:              0.5,
				NoiseDelta:                1e-5,
				PartitionSelectionEpsilon: 0.5,
				PartitionSelectionDelta:   1e-5,
				PreThreshold:              10,
				MaxPartitionsContributed:  17,
				Lower:                     0,
				Upper:                     10,
				NoiseKind:                 noise.GaussianNoise,
				PublicPartitions:          false,
			}},
		{"PreThreshold set Float64", noise.GaussianNoise, reflect.Float64, 0.5, 1e-5, 0.5, 1e-5, 10, 0, 10, false,
			&boundedSumFloat64Fn{
				NoiseEpsilon:              0.5,
				NoiseDelta:                1e-5,
				PartitionSelectionEpsilon: 0.5,
				PartitionSelectionDelta:   1e-5,
				PreThreshold:              10,
				MaxPartitionsContributed:  17,
				Lower:                     0,
				Upper:                     10,
				NoiseKind:                 noise.GaussianNoise,
				PublicPartitions:          false,
			}},
		{"lower > upper", noise.GaussianNoise, reflect.Int64, 0.5, 1e-5, 0.5, 1e-5, 0, 10, 0, true, nil},
		{"Float64 bounds that overflow when converted to int64", noise.GaussianNoise, reflect.Int64, 0.5, 1e-5, 0.5, 1e-5, 0, 0, 1e100, true, nil},
	} {
		got, err := newBoundedSumFn(PrivacySpec{preThreshold: tc.preThreshold, testMode: TestModeDisabled},
			SumParams{
				AggregationEpsilon:       tc.aggregationEpsilon,
				AggregationDelta:         tc.aggregationDelta,
				PartitionSelectionParams: PartitionSelectionParams{Epsilon: tc.partitionSelectionEpsilon, Delta: tc.partitionSelectionDelta},
				MaxPartitionsContributed: 17,
				MinValue:                 tc.lower,
				MaxValue:                 tc.upper,
			}, tc.noiseKind, tc.vKind, false)
		if (err != nil) != tc.wantErr {
			t.Fatalf("With %s, got=%v, wantErr=%t", tc.desc, err, tc.wantErr)
		}
		if diff := cmp.Diff(tc.want, got, opts...); diff != "" {
			t.Errorf("newBoundedSumFn mismatch for '%s' (-want +got):\n%s", tc.desc, diff)
		}
	}
}

func TestBoundedSumFloat64FnSetup(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		noiseKind noise.Kind
		wantNoise any
	}{
		{"Laplace noise kind", noise.LaplaceNoise, noise.Laplace()},
		{"Gaussian noise kind", noise.GaussianNoise, noise.Gaussian()}} {
		spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 0.5, PartitionSelectionEpsilon: 0.5, PartitionSelectionDelta: 1e-5})
		got, err := newBoundedSumFloat64Fn(
			*spec,
			SumParams{AggregationEpsilon: 0.5, PartitionSelectionParams: PartitionSelectionParams{Epsilon: 0.5, Delta: 1e-5}, MaxPartitionsContributed: 17, MinValue: 0, MaxValue: 10},
			tc.noiseKind,
			false,
		)
		if err != nil {
			t.Fatalf("Couldn't get boundedSumFloat64Fn: %v", err)
		}
		got.Setup()
		if !cmp.Equal(tc.wantNoise, got.noise) {
			t.Errorf("Setup: for %s got %v, want %v", tc.desc, got.noise, tc.wantNoise)
		}
	}
}

func TestBoundedSumInt64FnSetup(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		noiseKind noise.Kind
		wantNoise any
	}{
		{"Laplace noise kind", noise.LaplaceNoise, noise.Laplace()},
		{"Gaussian noise kind", noise.GaussianNoise, noise.Gaussian()},
	} {
		spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 0.5, PartitionSelectionEpsilon: 0.5, PartitionSelectionDelta: 1e-5})
		got, err := newBoundedSumInt64Fn(
			*spec,
			SumParams{AggregationEpsilon: 0.5, PartitionSelectionParams: PartitionSelectionParams{Epsilon: 0.5, Delta: 1e-5}, MaxPartitionsContributed: 17, MinValue: 0, MaxValue: 10},
			tc.noiseKind,
			false,
		)
		if err != nil {
			t.Fatalf("Couldn't get boundedSumInf64Fn: %v", err)
		}
		got.Setup()
		if !cmp.Equal(tc.wantNoise, got.noise) {
			t.Errorf("Setup: for %s got %v, want %v", tc.desc, got.noise, tc.wantNoise)
		}
	}
}

func TestBoundedSumInt64FnAddInput(t *testing.T) {
	// Since δ=0.5 and 2 entries are added, PreAggPartitionSelection always emits.
	// Since AggregationEpsilon=1e50, the noise is added with probability in the order of exp(-1e50),
	// which means we don't have to worry about tolerance/flakiness calculations.
	spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1e50, PartitionSelectionEpsilon: 1e50, PartitionSelectionDelta: 0.5})
	fn, err := newBoundedSumInt64Fn(
		*spec,
		SumParams{AggregationEpsilon: 1e50, PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1e50, Delta: 0.5}, MaxPartitionsContributed: 1, MinValue: 0, MaxValue: 2},
		noise.LaplaceNoise,
		false,
	)
	if err != nil {
		t.Fatalf("Couldn't get boundedSumInt64Fn: %v", err)
	}
	fn.Setup()

	accum, err := fn.CreateAccumulator()
	if err != nil {
		t.Fatalf("Couldn't create accum: %v", err)
	}
	fn.AddInput(accum, 2)
	fn.AddInput(accum, 2)

	got, err := fn.ExtractOutput(accum)
	if err != nil {
		t.Fatalf("Couldn't extract output: %v", err)
	}
	want := testutils.Int64Ptr(4)
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("unexpected output (-want +got):\n%s", diff)
	}
}

func TestBoundedSumInt64FnMergeAccumulators(t *testing.T) {
	// We use δ=0.5 so that partition selection is non-deterministic with 1 input
	// and deterministic with 2 inputs. This is used to verify that merging
	// accumulators is also affecting our partition selection outcome.
	//
	// Since AggregationEpsilon=1e50, the noise is added with probability in the order of exp(-1e50),
	// which means we don't have to worry about tolerance/flakiness calculations.
	spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1e50, PartitionSelectionEpsilon: 1e50, PartitionSelectionDelta: 0.5})
	fn, err := newBoundedSumInt64Fn(
		*spec,
		SumParams{AggregationEpsilon: 1e50, PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1e50, Delta: 0.5}, MaxPartitionsContributed: 1, MinValue: 0, MaxValue: 2},
		noise.LaplaceNoise,
		false,
	)
	if err != nil {
		t.Fatalf("Couldn't get boundedSumInt64Fn: %v", err)
	}
	fn.Setup()

	accum1, err := fn.CreateAccumulator()
	if err != nil {
		t.Fatalf("Couldn't create accum1: %v", err)
	}
	fn.AddInput(accum1, 2)
	accum2, err := fn.CreateAccumulator()
	if err != nil {
		t.Fatalf("Couldn't create accum2: %v", err)
	}
	fn.AddInput(accum2, 1)
	fn.MergeAccumulators(accum1, accum2)

	got, err := fn.ExtractOutput(accum1)
	if err != nil {
		t.Fatalf("Couldn't extract output: %v", err)
	}
	want := testutils.Int64Ptr(3)
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("unexpected output (-want +got):\n%s", diff)
	}
}

func TestBoundedSumInt64FnExtractOutputReturnsNilForSmallPartitions(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		inputSize int
	}{
		// It's a special case for partition selection in which the algorithm should always eliminate the partition.
		{"Empty input", 0},
		// The probability of keeping a partition with 1 privacy unit is equal to δ=1e-23 which results in a flakiness of 10⁻²³.
		{"Input with 1 privacy unit", 1},
	} {
		spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 0.5, PartitionSelectionEpsilon: 0.5, PartitionSelectionDelta: 1e-23})
		fn, err := newBoundedSumInt64Fn(
			*spec,
			SumParams{AggregationEpsilon: 0.5, PartitionSelectionParams: PartitionSelectionParams{Epsilon: 0.5, Delta: 1e-23}, MaxPartitionsContributed: 1, MinValue: 0, MaxValue: 2},
			noise.LaplaceNoise,
			false,
		)
		if err != nil {
			t.Fatalf("Couldn't get boundedSumInt64Fn: %v", err)
		}
		fn.Setup()
		accum, err := fn.CreateAccumulator()
		if err != nil {
			t.Fatalf("Couldn't create accum: %v", err)
		}
		for i := 0; i < tc.inputSize; i++ {
			fn.AddInput(accum, 1)
		}

		got, err := fn.ExtractOutput(accum)
		if err != nil {
			t.Fatalf("Couldn't extract output: %v", err)
		}

		// Should return nil output for small partitions.
		if got != nil {
			t.Errorf("ExtractOutput: for %s got: %d, want nil", tc.desc, *got)
		}
	}
}

func TestBoundedSumInt64FnExtractOutputWithPublicPartitionsDoesNotThreshold(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		inputSize int
	}{
		{"Empty input", 0},
		{"Input with 1 user", 1},
		{"Input with 10 users", 10},
		{"Input with 100 users", 100},
	} {
		spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1.0})
		fn, err := newBoundedSumFloat64Fn(
			*spec,
			SumParams{AggregationEpsilon: 1.0, MaxPartitionsContributed: 1, MinValue: 0, MaxValue: 2},
			noise.LaplaceNoise,
			true,
		)
		if err != nil {
			t.Fatalf("Couldn't get boundedSumInt64Fn: %v", err)
		}
		fn.Setup()
		accum, err := fn.CreateAccumulator()
		if err != nil {
			t.Fatalf("Couldn't create accum: %v", err)
		}
		for i := 0; i < tc.inputSize; i++ {
			fn.AddInput(accum, 1)
		}

		got, err := fn.ExtractOutput(accum)
		if err != nil {
			t.Fatalf("Couldn't extract output: %v", err)
		}

		if got == nil {
			t.Errorf("ExtractOutput for %s thresholded with public partitions when it shouldn't", tc.desc)
		}
	}
}

func TestBoundedSumFloat64FnAddInput(t *testing.T) {
	// Since δ=0.5 and 2 entries are added, PreAggPartitionSelection always emits.
	// Since AggregationEpsilon=1e50, added noise is negligible.
	spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1e50, PartitionSelectionEpsilon: 1e50, PartitionSelectionDelta: 0.5})
	fn, err := newBoundedSumFloat64Fn(
		*spec,
		SumParams{AggregationEpsilon: 1e50, PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1e50, Delta: 0.5}, MaxPartitionsContributed: 1, MinValue: 0, MaxValue: 2},
		noise.LaplaceNoise,
		false,
	)
	if err != nil {
		t.Fatalf("Couldn't get boundedSumFloat64Fn: %v", err)
	}
	fn.Setup()

	accum, err := fn.CreateAccumulator()
	if err != nil {
		t.Fatalf("Couldn't create accum: %v", err)
	}
	fn.AddInput(accum, 2)
	fn.AddInput(accum, 2)

	got, err := fn.ExtractOutput(accum)
	if err != nil {
		t.Fatalf("Couldn't extract output: %v", err)
	}
	want := testutils.Float64Ptr(4)
	if diff := cmp.Diff(want, got, cmpopts.EquateApprox(0, testutils.LaplaceTolerance(23, 2, 1e50))); diff != "" {
		t.Errorf("unexpected output (-want +got):\n%s", diff)
	}
}

func TestBoundedSumFloat64FnMergeAccumulators(t *testing.T) {
	// We use δ=0.5 so that partition selection is non-deterministic with 1 input
	// and deterministic with 2 inputs. This is used to verify that merging
	// accumulators is also effecting our partition selection outcome.
	//
	// Since AggregationEpsilon=1e50, added noise is negligible.
	spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1e50, PartitionSelectionEpsilon: 1e50, PartitionSelectionDelta: 0.5})
	fn, err := newBoundedSumFloat64Fn(
		*spec,
		SumParams{AggregationEpsilon: 1e50, PartitionSelectionParams: PartitionSelectionParams{Epsilon: 1e50, Delta: 0.5}, MaxPartitionsContributed: 1, MinValue: 0, MaxValue: 2},
		noise.LaplaceNoise,
		false,
	)
	if err != nil {
		t.Fatalf("Couldn't get boundedSumFloat64Fn: %v", err)
	}
	fn.Setup()

	accum1, err := fn.CreateAccumulator()
	if err != nil {
		t.Fatalf("Couldn't create accum1: %v", err)
	}
	fn.AddInput(accum1, 2)
	accum2, err := fn.CreateAccumulator()
	if err != nil {
		t.Fatalf("Couldn't create accum2: %v", err)
	}
	fn.AddInput(accum2, 1)
	fn.MergeAccumulators(accum1, accum2)

	got, err := fn.ExtractOutput(accum1)
	if err != nil {
		t.Fatalf("Couldn't extract output: %v", err)
	}
	want := testutils.Float64Ptr(3)
	if diff := cmp.Diff(want, got, cmpopts.EquateApprox(0, testutils.LaplaceTolerance(23, 2, 1e50))); diff != "" {
		t.Errorf("unexpected output (-want +got):\n%s", diff)
	}
}

func TestBoundedSumFloat64FnExtractOutputReturnsNilForSmallPartitions(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		inputSize int
	}{
		// It's a special case for partition selection in which the algorithm should always eliminate the partition.
		{"Empty input", 0},
		// The probability of keeping a partition with 1 privacy unit is equal to δ=1e-23 which results in a flakiness of 10⁻²³.
		{"Input with 1 privacy unit", 1}} {

		spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 0.5, PartitionSelectionEpsilon: 0.5, PartitionSelectionDelta: 1e-23})
		fn, err := newBoundedSumFloat64Fn(
			*spec,
			SumParams{AggregationEpsilon: 0.5, PartitionSelectionParams: PartitionSelectionParams{Epsilon: 0.5, Delta: 1e-23}, MaxPartitionsContributed: 1, MinValue: 0, MaxValue: 2},
			noise.LaplaceNoise,
			false,
		)
		if err != nil {
			t.Fatalf("Couldn't get boundedSumFloat64Fn: %v", err)
		}
		fn.Setup()
		accum, err := fn.CreateAccumulator()
		if err != nil {
			t.Fatalf("Couldn't create accum: %v", err)
		}
		for i := 0; i < tc.inputSize; i++ {
			fn.AddInput(accum, 1)
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

func TestBoundedSumFloat64FnExtractOutputWithPublicPartitionsDoesNotThreshold(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		inputSize int
	}{
		{"Empty input", 0},
		{"Input with 1 user", 1},
		{"Input with 10 users", 10},
		{"Input with 100 users", 100}} {
		spec := privacySpec(t, PrivacySpecParams{AggregationEpsilon: 1})
		fn, err := newBoundedSumFloat64Fn(
			*spec,
			SumParams{AggregationEpsilon: 1, MaxPartitionsContributed: 1, MinValue: 0, MaxValue: 2},
			noise.LaplaceNoise,
			true,
		)
		if err != nil {
			t.Fatalf("Couldn't get boundedSumFloat64Fn: %v", err)
		}
		fn.Setup()
		accum, err := fn.CreateAccumulator()
		if err != nil {
			t.Fatalf("Couldn't create accum: %v", err)
		}
		for i := 0; i < tc.inputSize; i++ {
			fn.AddInput(accum, 1)
		}

		got, err := fn.ExtractOutput(accum)
		if err != nil {
			t.Fatalf("Couldn't extract output: %v", err)
		}
		if got == nil {
			t.Errorf("ExtractOutput for %s thresholded with public partitions when it shouldn't", tc.desc)
		}
	}
}

func TestCheckNumericType(t *testing.T) {
	for _, tc := range []struct {
		desc     string
		fullType typex.FullType
		wantErr  bool
	}{
		{"int", typex.New(reflect.TypeOf(int(0))), false},
		{"int8", typex.New(reflect.TypeOf(int8(0))), false},
		{"int16", typex.New(reflect.TypeOf(int16(0))), false},
		{"int32", typex.New(reflect.TypeOf(int32(0))), false},
		{"int64", typex.New(reflect.TypeOf(int64(0))), false},
		{"uint", typex.New(reflect.TypeOf(uint(0))), false},
		{"uint8", typex.New(reflect.TypeOf(uint8(0))), false},
		{"uint16", typex.New(reflect.TypeOf(uint16(0))), false},
		{"uint32", typex.New(reflect.TypeOf(uint32(0))), false},
		{"uint64", typex.New(reflect.TypeOf(uint64(0))), false},
		{"float32", typex.New(reflect.TypeOf(float32(0))), false},
		{"float64", typex.New(reflect.TypeOf(float64(0))), false},
		{"string", typex.New(reflect.TypeOf("")), true},
	} {
		err := checkNumericType(tc.fullType)
		if (err != nil) != tc.wantErr {
			t.Errorf("checkNumericType: when %s for err got got %v, want %t", tc.desc, err, tc.wantErr)
		}
	}
}
