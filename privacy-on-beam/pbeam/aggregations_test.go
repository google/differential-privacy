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

	"github.com/google/differential-privacy/go/noise"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestNewBoundedSumFn(t *testing.T) {
	opts := []cmp.Option{
		cmpopts.EquateApprox(0, 1e-10),
		cmpopts.IgnoreUnexported(boundedSumFloat64Fn{}, boundedSumInt64Fn{}),
	}
	for _, tc := range []struct {
		desc      string
		noiseKind noise.Kind
		vKind     reflect.Kind
		want      interface{}
	}{
		{"Laplace Float64", noise.LaplaceNoise, reflect.Float64,
			&boundedSumFloat64Fn{
				NoiseEpsilon:              0.5,
				PartitionSelectionEpsilon: 0.5,
				NoiseDelta:                0,
				PartitionSelectionDelta:   1e-5,
				MaxPartitionsContributed:  17,
				Lower:                     0,
				Upper:                     10,
				NoiseKind:                 noise.LaplaceNoise,
			}},
		{"Gaussian Float64", noise.GaussianNoise, reflect.Float64,
			&boundedSumFloat64Fn{
				NoiseEpsilon:              0.5,
				PartitionSelectionEpsilon: 0.5,
				NoiseDelta:                5e-6,
				PartitionSelectionDelta:   5e-6,
				MaxPartitionsContributed:  17,
				Lower:                     0,
				Upper:                     10,
				NoiseKind:                 noise.GaussianNoise,
			}},
		{"Laplace Int64", noise.LaplaceNoise, reflect.Int64,
			&boundedSumInt64Fn{
				NoiseEpsilon:              0.5,
				PartitionSelectionEpsilon: 0.5,
				NoiseDelta:                0,
				PartitionSelectionDelta:   1e-5,
				MaxPartitionsContributed:  17,
				Lower:                     0,
				Upper:                     10,
				NoiseKind:                 noise.LaplaceNoise,
			}},
		{"Gaussian Int64", noise.GaussianNoise, reflect.Int64,
			&boundedSumInt64Fn{
				NoiseEpsilon:              0.5,
				PartitionSelectionEpsilon: 0.5,
				NoiseDelta:                5e-6,
				PartitionSelectionDelta:   5e-6,
				MaxPartitionsContributed:  17,
				Lower:                     0,
				Upper:                     10,
				NoiseKind:                 noise.GaussianNoise,
			}},
	} {
		got := newBoundedSumFn(1, 1e-5, 17, 0, 10, tc.noiseKind, tc.vKind)
		if diff := cmp.Diff(tc.want, got, opts...); diff != "" {
			t.Errorf("newBoundedSumFn mismatch for '%s' (-want +got):\n%s", tc.desc, diff)
		}
	}
}

func TestBoundedSumFloat64FnSetup(t *testing.T) {
	for _, tc := range []struct {
		desc      string
		noiseKind noise.Kind
		wantNoise interface{}
	}{
		{"Laplace noise kind", noise.LaplaceNoise, noise.Laplace()},
		{"Gaussian noise kind", noise.GaussianNoise, noise.Gaussian()}} {
		got := newBoundedSumFloat64Fn(1, 1e-5, 17, 0, 10, tc.noiseKind)
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
		wantNoise interface{}
	}{
		{"Laplace noise kind", noise.LaplaceNoise, noise.Laplace()},
		{"Gaussian noise kind", noise.GaussianNoise, noise.Gaussian()}} {
		got := newBoundedSumInt64Fn(1, 1e-5, 17, 0, 10, tc.noiseKind)
		got.Setup()
		if !cmp.Equal(tc.wantNoise, got.noise) {
			t.Errorf("Setup: for %s got %v, want %v", tc.desc, got.noise, tc.wantNoise)
		}
	}
}

func TestBoundedSumInt64FnAddInput(t *testing.T) {
	// Since δ=0.5 and 2 entries are added, PreAggPartitionSelection always emits.
	// Since ε=1e100, the noise is added with probability in the order of exp(-1e100),
	// which means we don't have to worry about tolerance/flakiness calculations.
	fn := newBoundedSumInt64Fn(1e100, 0.5, 1, 0, 2, noise.LaplaceNoise)
	fn.Setup()

	accum := fn.CreateAccumulator()
	fn.AddInput(accum, 2)
	fn.AddInput(accum, 2)

	got := fn.ExtractOutput(accum)
	want := int64Ptr(4)
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("unexpected output (-want +got):\n%s", diff)
	}
}

func TestBoundedSumInt64FnMergeAccumulators(t *testing.T) {
	// We use δ=0.5 so that partition selection is non-deterministic with 1 input
	// and deterministic with 2 inputs. This is used to verify that merging
	// accumulators is also affecting our partition selection outcome.
	//
	// Since ε=1e100, the noise is added with probability in the order of exp(-1e100),
	// which means we don't have to worry about tolerance/flakiness calculations.
	fn := newBoundedSumInt64Fn(1e100, 0.5, 1, 0, 2, noise.LaplaceNoise)
	fn.Setup()

	accum1 := fn.CreateAccumulator()
	fn.AddInput(accum1, 2)
	accum2 := fn.CreateAccumulator()
	fn.AddInput(accum2, 1)
	fn.MergeAccumulators(accum1, accum2)

	got := fn.ExtractOutput(accum1)
	want := int64Ptr(3)
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
		{"Input with 1 privacy unit", 1}} {

		fn := newBoundedSumInt64Fn(1, 1e-23, 1, 0, 2, noise.LaplaceNoise)
		fn.Setup()
		accum := fn.CreateAccumulator()
		for i := 0; i < tc.inputSize; i++ {
			fn.AddInput(accum, 1)
		}

		got := fn.ExtractOutput(accum)

		// Should return nil output for small partitions.
		if got != nil {
			t.Errorf("ExtractOutput: for %s got: %d, want nil", tc.desc, *got)
		}
	}
}

func TestBoundedSumFloat64FnAddInput(t *testing.T) {
	// Since δ=0.5 and 2 entries are added, PreAggPartitionSelection always emits.
	// Since ε=1e100, added noise is negligible.
	fn := newBoundedSumFloat64Fn(1e100, 0.5, 1, 0, 2, noise.LaplaceNoise)
	fn.Setup()

	accum := fn.CreateAccumulator()
	fn.AddInput(accum, 2)
	fn.AddInput(accum, 2)

	got := fn.ExtractOutput(accum)
	want := float64Ptr(4)
	if diff := cmp.Diff(want, got, cmpopts.EquateApprox(0, laplaceTolerance(23, 2, 1e100))); diff != "" {
		t.Errorf("unexpected output (-want +got):\n%s", diff)
	}
}

func TestBoundedSumFloat64FnMergeAccumulators(t *testing.T) {
	// We use δ=0.5 so that partition selection is non-deterministic with 1 input
	// and deterministic with 2 inputs. This is used to verify that merging
	// accumulators is also effecting our partition selection outcome.
	//
	// Since ε=1e100, added noise is negligible.
	fn := newBoundedSumFloat64Fn(1e100, 0.5, 1, 0, 2, noise.LaplaceNoise)
	fn.Setup()

	accum1 := fn.CreateAccumulator()
	fn.AddInput(accum1, 2)
	accum2 := fn.CreateAccumulator()
	fn.AddInput(accum2, 1)
	fn.MergeAccumulators(accum1, accum2)

	got := fn.ExtractOutput(accum1)
	want := float64Ptr(3)
	if diff := cmp.Diff(want, got, cmpopts.EquateApprox(0, laplaceTolerance(23, 2, 1e100))); diff != "" {
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

		fn := newBoundedSumFloat64Fn(1, 1e-23, 1, 0, 2, noise.LaplaceNoise)
		fn.Setup()
		accum := fn.CreateAccumulator()
		for i := 0; i < tc.inputSize; i++ {
			fn.AddInput(accum, 1)
		}

		got := fn.ExtractOutput(accum)

		// Should return nil output for small partitions.
		if got != nil {
			t.Errorf("ExtractOutput: for %s got: %f, want nil", tc.desc, *got)
		}
	}
}
