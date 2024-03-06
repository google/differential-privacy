//
// Copyright 2021 Google LLC
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

import "github.com/google/differential-privacy/go/v3/noise"

// TestMode is an enum representing different test modes for test pipelines available in Privacy on Beam.
type TestMode int

const (
	// TestModeDisabled indicates that test mode is disabled. Default.
	TestModeDisabled TestMode = iota
	// TestModeWithContributionBounding is the test mode where no noise is added, but contribution bounding is done.
	TestModeWithContributionBounding
	// TestModeWithoutContributionBounding is the test mode where no noise is added and no contribution bounding is done.
	TestModeWithoutContributionBounding
)

func (tm TestMode) isEnabled() bool {
	return tm != TestModeDisabled
}

// noNoise is a Noise instance that doesn't add noise to the data, and has a
// threshold of 0. Used as the noise type only when testMode is enabled in PrivacySpec.
type noNoise struct{}

func (noNoise) AddNoiseInt64(x, _, _ int64, _, _ float64) (int64, error) {
	return x, nil
}

func (noNoise) AddNoiseFloat64(x float64, _ int64, _, _, _ float64) (float64, error) {
	return x, nil
}

func (noNoise) Threshold(_ int64, _, _, _, _ float64) (float64, error) {
	return 0.0, nil
}

func (noNoise) ComputeConfidenceIntervalInt64(noisedX, _, _ int64, _, _, _ float64) (noise.ConfidenceInterval, error) {
	return noise.ConfidenceInterval{LowerBound: float64(noisedX), UpperBound: float64(noisedX)}, nil
}

func (noNoise) ComputeConfidenceIntervalFloat64(noisedX float64, _ int64, _, _, _, _ float64) (noise.ConfidenceInterval, error) {
	return noise.ConfidenceInterval{LowerBound: noisedX, UpperBound: noisedX}, nil
}
