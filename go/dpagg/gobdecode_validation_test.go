//
// Copyright 2026 Google LLC
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

package dpagg

import (
	"testing"

	"github.com/google/differential-privacy/go/v4/noise"
)

func TestCountGobDecodeRejectsInvalidNoiseKind(t *testing.T) {
	enc := encodableCount{
		Epsilon:         1.0,
		Delta:           0,
		L0Sensitivity:   1,
		LInfSensitivity: 1,
		NoiseKind:       99, // invalid
		Count:           0,
	}
	data, err := encode(&enc)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}
	var c Count
	if err := c.GobDecode(data); err == nil {
		t.Error("GobDecode: expected error for invalid NoiseKind, got nil")
	}
}

func TestCountGobDecodeRejectsInvalidL0Sensitivity(t *testing.T) {
	enc := encodableCount{
		Epsilon:         1.0,
		Delta:           0,
		L0Sensitivity:   -1, // invalid
		LInfSensitivity: 1,
		NoiseKind:       noise.LaplaceNoise,
		Count:           0,
	}
	data, err := encode(&enc)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}
	var c Count
	if err := c.GobDecode(data); err == nil {
		t.Error("GobDecode: expected error for negative l0Sensitivity, got nil")
	}
}

func TestBoundedSumFloat64GobDecodeRejectsInvalidNoiseKind(t *testing.T) {
	enc := encodableBoundedSumFloat64{
		Epsilon:         1.0,
		Delta:           0,
		L0Sensitivity:   1,
		LInfSensitivity: 1.0,
		Lower:           0,
		Upper:           10,
		NoiseKind:       99, // invalid
		Sum:             0,
	}
	data, err := encode(&enc)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}
	var bs BoundedSumFloat64
	if err := bs.GobDecode(data); err == nil {
		t.Error("GobDecode: expected error for invalid NoiseKind, got nil")
	}
}

func TestBoundedSumFloat64GobDecodeRejectsInvalidLInfSensitivity(t *testing.T) {
	enc := encodableBoundedSumFloat64{
		Epsilon:         1.0,
		Delta:           0,
		L0Sensitivity:   1,
		LInfSensitivity: -1.0, // invalid: must be positive
		Lower:           0,
		Upper:           10,
		NoiseKind:       noise.LaplaceNoise,
		Sum:             0,
	}
	data, err := encode(&enc)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}
	var bs BoundedSumFloat64
	if err := bs.GobDecode(data); err == nil {
		t.Error("GobDecode: expected error for negative lInfSensitivity, got nil")
	}
}

func TestBoundedQuantilesGobDecodeRejectsZeroBranchingFactor(t *testing.T) {
	enc := encodableBoundedQuantiles{
		Epsilon:           1.0,
		Delta:             1e-5,
		L0Sensitivity:     1,
		LInfSensitivity:   1.0,
		TreeHeight:        4,
		BranchingFactor:   0, // invalid: causes div-by-zero
		Lower:             0,
		Upper:             10,
		NumLeaves:         0,
		LeftmostLeafIndex: 0,
		NoiseKind:         noise.GaussianNoise,
		QuantileTree:      make(map[int]int64),
	}
	data, err := encode(&enc)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}
	var bq BoundedQuantiles
	if err := bq.GobDecode(data); err == nil {
		t.Error("GobDecode: expected error for zero branchingFactor, got nil")
	}
}

func TestBoundedQuantilesGobDecodeRejectsInvalidNoiseKind(t *testing.T) {
	enc := encodableBoundedQuantiles{
		Epsilon:           1.0,
		Delta:             1e-5,
		L0Sensitivity:     1,
		LInfSensitivity:   1.0,
		TreeHeight:        4,
		BranchingFactor:   16,
		Lower:             0,
		Upper:             10,
		NumLeaves:         0,
		LeftmostLeafIndex: 0,
		NoiseKind:         99, // invalid
		QuantileTree:      make(map[int]int64),
	}
	data, err := encode(&enc)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}
	var bq BoundedQuantiles
	if err := bq.GobDecode(data); err == nil {
		t.Error("GobDecode: expected error for invalid NoiseKind, got nil")
	}
}
