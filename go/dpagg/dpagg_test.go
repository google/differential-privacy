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

package dpagg

import (
	"math"

	"github.com/google/differential-privacy/go/noise"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

// This file contains structs, functions, and values used to test DP aggregations.

var (
	ln3        = math.Log(3)
	tenten     = math.Pow10(-10)
	tenfive    = math.Pow10(-5)
	alphaLevel = 0.05
)

// noNoise is a Noise instance that doesn't add noise to the data, and has a
// threshold of 5.
type noNoise struct {
	noise.Noise
}

func (noNoise) AddNoiseInt64(x, _, _ int64, _, _ float64) int64 {
	return x
}

func (noNoise) AddNoiseFloat64(x float64, _ int64, _, _, _ float64) float64 {
	return x
}

func ApproxEqual(x, y float64) bool {
	return cmp.Equal(x, y, cmpopts.EquateApprox(0, tenten))
}

func (noNoise) Threshold(_ int64, _, _, _, _ float64) float64 {
	return 5
}

func (nN noNoise) ComputeConfidenceIntervalInt64(noisedX, l0, lInf int64, eps, del, alpha float64) (noise.ConfidenceInterval, error) {
	confInt, err := nN.Noise.ComputeConfidenceIntervalInt64(noisedX, l0, lInf, eps, del, alpha)
	return confInt, err
}

func (nN noNoise) ComputeConfidenceIntervalFloat64(noisedX float64, l0 int64, lInf, eps, del, alpha float64) (noise.ConfidenceInterval, error) {
	confInt, err := nN.Noise.ComputeConfidenceIntervalFloat64(noisedX, l0, lInf, eps, del, alpha)
	return confInt, err
}

func getNoiselessConfInt(noise noise.Noise) noise.Noise {
	return noNoise{noise}
}

// mockConfInt is a Noise instance that returns a pre-set confidence interval.
// Useful for testing post-processing in confidence intervals.
type mockConfInt struct {
	noNoise
	confInt noise.ConfidenceInterval
}

func (mCI mockConfInt) ComputeConfidenceIntervalInt64(_, _, _ int64, _, _, _ float64) (noise.ConfidenceInterval, error) {
	return mCI.confInt, nil
}

func (mCI mockConfInt) ComputeConfidenceIntervalFloat64(_ float64, _ int64, _, _, _, _ float64) (noise.ConfidenceInterval, error) {
	return mCI.confInt, nil
}

func getMockConfInt(confInt noise.ConfidenceInterval) noise.Noise {
	return mockConfInt{confInt: confInt}
}
