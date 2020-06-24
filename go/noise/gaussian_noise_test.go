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

package noise

import (
	"math"
	"testing"

	"github.com/grd/stat"
)

func TestGaussianStatistics(t *testing.T) {
	const numberOfSamples = 125000
	for _, tc := range []struct {
		l0Sensitivity                                   int64
		lInfSensitivity, epsilon, delta, mean, variance float64
	}{
		{
			l0Sensitivity:   1,
			lInfSensitivity: 1.0,
			epsilon:         ln3,
			delta:           1e-10,
			mean:            0.0,
			variance:        28.76478576660,
		},
		{
			l0Sensitivity:   1,
			lInfSensitivity: 1.0,
			epsilon:         ln3,
			delta:           1e-10,
			mean:            45941223.02107,
			variance:        28.76478576660,
		},
		{
			l0Sensitivity:   1,
			lInfSensitivity: 1.0,
			epsilon:         ln3,
			delta:           1e-10,
			mean:            0.0,
			variance:        28.76478576660,
		},
		{
			l0Sensitivity:   1,
			lInfSensitivity: 2.0,
			epsilon:         2.0 * ln3,
			delta:           1e-10,
			mean:            0.0,
			variance:        30.637955,
		},
		{
			l0Sensitivity:   2,
			lInfSensitivity: 1.0,
			epsilon:         2.0 * ln3,
			delta:           1e-10,
			mean:            0.0,
			variance:        15.318977,
		},
		{
			l0Sensitivity:   1,
			lInfSensitivity: 1.0,
			epsilon:         2 * ln3,
			delta:           1e-10,
			mean:            0.0,
			variance:        7.65948867798,
		},
		{
			l0Sensitivity:   1,
			lInfSensitivity: 1.0,
			epsilon:         ln3,
			delta:           1e-5,
			mean:            0.0,
			variance:        11.73597717285,
		},
	} {
		noisedSamples := make(stat.Float64Slice, numberOfSamples)
		for i := 0; i < numberOfSamples; i++ {
			noisedSamples[i] = gauss.AddNoiseFloat64(tc.mean, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.delta)
		}
		sampleMean, sampleVariance := stat.Mean(noisedSamples), stat.Variance(noisedSamples)
		// Assuming that the Gaussian samples have a mean of 0 and the specified variance of tc.variance,
		// sampleMeanFloat64 and sampleMeanInt64 are approximately Gaussian distributed with a mean of 0
		// and standard deviation of sqrt(tc.variance⁻ / numberOfSamples).
		//
		// The meanErrorTolerance is set to the 99.9995% quantile of the anticipated distribution. Thus,
		// the test falsely rejects with a probability of 10⁻⁵.
		meanErrorTolerance := 4.41717 * math.Sqrt(tc.variance/float64(numberOfSamples))
		// Assuming that the Gaussian samples have the specified variance of tc.variance, sampleVarianceFloat64
		// and sampleVarianceInt64 are approximately Gaussian distributed with a mean of tc.variance and a
		// standard deviation of sqrt(2) * tc.variance / sqrt(numberOfSamples).
		//
		// The varianceErrorTolerance is set to the 99.9995% quantile of the anticipated distribution. Thus,
		// the test falsely rejects with a probability of 10⁻⁵.
		varianceErrorTolerance := 4.41717 * math.Sqrt2 * tc.variance / math.Sqrt(float64(numberOfSamples))

		if !nearEqual(sampleMean, tc.mean, meanErrorTolerance) {
			t.Errorf("float64 got mean = %f, want %f (parameters %+v)", sampleMean, tc.mean, tc)
		}
		if !nearEqual(sampleVariance, tc.variance, varianceErrorTolerance) {
			t.Errorf("float64 got variance = %f, want %f (parameters %+v)", sampleVariance, tc.variance, tc)
			sigma := sigmaForGaussian(tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.delta)
			t.Errorf("btw, true sigma is %f, squares to %f", sigma, sigma*sigma)
		}
	}
}

func TestSymmetricBinomialStatisitcs(t *testing.T) {
	const numberOfSamples = 125000
	for _, tc := range []struct {
		sqrtN  float64
		mean   float64
		stdDev float64
	}{
		{
			sqrtN:  1000.0,
			mean:   0.0,
			stdDev: 500.0,
		},
		{
			sqrtN:  1000000.0,
			mean:   0.0,
			stdDev: 500000.0,
		},
		{
			sqrtN:  1000000000.0,
			mean:   0.0,
			stdDev: 500000000.0,
		},
	} {
		binomialSamples := make(stat.IntSlice, numberOfSamples)
		for i := 0; i < numberOfSamples; i++ {
			binomialSamples[i] = symmetricBinomial(tc.sqrtN)
		}
		sampleMean, sampleVariance := stat.Mean(binomialSamples), stat.Variance(binomialSamples)
		// Assuming that the binomial samples have a mean of 0 and the specified standard deviation
		// of tc.stdDev, sampleMean is approximately Gaussian-distributed with a mean of 0
		// and standard deviation of tc.stdDev / sqrt(numberOfSamples).
		//
		// The meanErrorTolerance is set to the 99.9995% quantile of the anticipated distribution
		// of sampleMean. Thus, the test falsely rejects with a probability of 10⁻⁵.
		meanErrorTolerance := 4.41717 * tc.stdDev / math.Sqrt(float64(numberOfSamples))
		// Assuming that the binomial samples have the specified standard deviation of tc.stdDev,
		// sampleVariance is approximately Gaussian-distributed with a mean of tc.stdDev²
		// and a standard deviation of sqrt(2) * tc.stdDev² / sqrt(numberOfSamples).
		//
		// The varianceErrorTolerance is set to the 99.9995% quantile of the anticipated distribution
		// of sampleVariance. Thus, the test falsely rejects with a probability of 10⁻⁵.
		varianceErrorTolerance := 4.41717 * math.Sqrt2 * math.Pow(tc.stdDev, 2.0) / math.Sqrt(float64(numberOfSamples))

		if !nearEqual(sampleMean, tc.mean, meanErrorTolerance) {
			t.Errorf("got mean = %f, want %f (parameters %+v)", sampleMean, tc.mean, tc)
		}
		if !nearEqual(sampleVariance, math.Pow(tc.stdDev, 2.0), varianceErrorTolerance) {
			t.Errorf("got variance = %f, want %f (parameters %+v)", sampleVariance, math.Pow(tc.stdDev, 2.0), tc)
		}
	}
}

func TestDeltaForGaussian(t *testing.T) {
	for _, tc := range []struct {
		desc            string
		sigma           float64
		epsilon         float64
		l0Sensitivity   int64
		lInfSensitivity float64
		wantDelta       float64
		allowError      float64
	}{
		{
			desc:            "No noise added case",
			sigma:           0,
			epsilon:         1,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			// Attacker can deterministically verify all outputs of the Gaussian
			// mechanism when no noise is added.
			wantDelta: 1,
		},
		{
			desc:            "Overflow handling from large epsilon",
			sigma:           1,
			epsilon:         math.Inf(+1),
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			// The full privacy leak is captured in the ε term.
			wantDelta: 0,
		},
		{
			desc:            "Overflow handling from large sensitivity",
			sigma:           1,
			epsilon:         1,
			l0Sensitivity:   1,
			lInfSensitivity: math.Inf(+1),
			// Infinite sensitivity cannot be hidden by finite noise.
			// No privacy guarantees.
			wantDelta: 1,
		},
		{
			desc:            "Underflow handling from low sensitivity",
			sigma:           1,
			epsilon:         1,
			l0Sensitivity:   1,
			lInfSensitivity: math.Nextafter(0, math.Inf(+1)),
			wantDelta:       0,
		},
		{
			desc:            "Correct value calculated",
			sigma:           10,
			epsilon:         0.1,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			wantDelta:       0.008751768145810,
			allowError:      1e-10,
		},
		{
			desc:            "Correct value calculated non-trivial lInfSensitivity",
			sigma:           20,
			l0Sensitivity:   1,
			lInfSensitivity: 2,
			epsilon:         0.1,
			wantDelta:       0.008751768145810,
			allowError:      1e-10,
		},
		{
			desc:            "Correct value calculated non-trivial l0Sensitivity",
			sigma:           20,
			l0Sensitivity:   4,
			lInfSensitivity: 1,
			epsilon:         0.1,
			wantDelta:       0.008751768145810,
			allowError:      1e-10,
		},
		{
			desc:            "Correct value calculated using typical epsilon",
			sigma:           10,
			l0Sensitivity:   1,
			lInfSensitivity: 5,
			epsilon:         math.Log(3),
			wantDelta:       0.004159742234000802,
			allowError:      1e-10,
		},
		{
			desc:            "Correct value calculated with epsilon = 0",
			sigma:           0.5,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0,
			wantDelta:       0.6826894921370859,
			allowError:      1e-10,
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			got := deltaForGaussian(tc.sigma, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon)
			if math.Abs(got-tc.wantDelta) > tc.allowError {
				t.Errorf("Got delta: %1.11f, want delta: %1.11f", got, tc.wantDelta)
			}
		})
	}
}

func TestSigmaForGaussianInvertsDeltaForGaussian(t *testing.T) {
	// For these tests, we specify the value of sigma that we want to compute and
	// use DeltaForGaussian to determine the corresponding delta. We then verify
	// whether (given said delta) we can reconstruct sigma within the desired
	// tolerance. This validates that the function
	//   delta ↦ SigmaForGaussian(l2Sensitivity, epsilon, delta)
	// is an approximate inverse function of
	//   sigma ↦ DeltaForGuassian(sigma, l2Sensitivity, epsilon).

	for _, tc := range []struct {
		desc            string
		sigma           float64
		l0Sensitivity   int64
		lInfSensitivity float64
		epsilon         float64
	}{
		{
			desc:            "sigma smaller than l2Sensitivity",
			sigma:           0.3,
			l0Sensitivity:   1,
			lInfSensitivity: 0.5,
			epsilon:         0.5,
		},
		{
			desc:            "sigma larger than l2Sensitivity",
			sigma:           15,
			l0Sensitivity:   1,
			lInfSensitivity: 10,
			epsilon:         0.5,
		},
		{
			desc:            "sigma smaller non-trivial l0Sensitivity",
			sigma:           0.3,
			l0Sensitivity:   5,
			lInfSensitivity: 0.5,
			epsilon:         0.5,
		},
		{
			desc: "small delta",
			// Results in delta = 3.129776773173962e-141
			sigma:           500,
			l0Sensitivity:   1,
			lInfSensitivity: 10,
			epsilon:         0.5,
		},
		{
			desc:            "high lInfSensitivity",
			sigma:           1e102,
			l0Sensitivity:   1,
			lInfSensitivity: 1e100,
			epsilon:         0.1,
		},
		{
			desc:            "epsilon = 0",
			sigma:           0.5,
			l0Sensitivity:   1,
			lInfSensitivity: 1.0,
			epsilon:         0,
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			deltaTight := deltaForGaussian(tc.sigma, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon)
			gotSigma := sigmaForGaussian(tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, deltaTight)
			if !(tc.sigma <= gotSigma && gotSigma <= (1+gaussianSigmaAccuracy)*tc.sigma) {
				t.Errorf("Got sigma: %f, want sigma in [%f, %f]", gotSigma, tc.sigma, (1+gaussianSigmaAccuracy)*tc.sigma)

			}
		})
	}
}

// This tests any logic that we need to special case for computing sigma (e.g.,
// precondition checking and boundary conditions).
func TestSigmaForGaussianWithDeltaOf1(t *testing.T) {
	got := sigmaForGaussian(1 /* l0 */, 1 /* lInf */, 0 /* ε */, 1 /* δ */)
	if got != 0 {
		t.Errorf("Got sigma: %f, want sigma: 0,", got)
	}
}

var thresholdGaussianTestCases = []struct {
	desc            string
	l0Sensitivity   int64
	lInfSensitivity float64
	epsilon         float64
	deltaNoise      float64
	deltaThreshold  float64
	threshold       float64
}{
	{
		desc:            "simple values",
		l0Sensitivity:   1,
		lInfSensitivity: 1,
		epsilon:         ln3,
		// deltaNoise is chosen to get a sigma of 1.
		deltaNoise: 0.10985556344445052,
		// 0.022750131948 is the 1-sided tail probability of landing more than 2
		// standard deviations from the mean of the Gaussian distribution.
		deltaThreshold: 0.022750131948,
		threshold:      3,
	},
	{
		desc:            "scale lInfSensitivity",
		l0Sensitivity:   1,
		lInfSensitivity: 0.5,
		epsilon:         ln3,
		// deltaNoise is chosen to get a sigma of 1.
		deltaNoise:     0.0041597422340007885,
		deltaThreshold: 0.000232629079,
		threshold:      4,
	},
	{
		desc:            "scale lInfSensitivity and sigma",
		l0Sensitivity:   1,
		lInfSensitivity: 2,
		epsilon:         ln3,
		// deltaNoise is chosen to get a sigma of 2.
		deltaNoise:     0.10985556344445052,
		deltaThreshold: 0.022750131948,
		threshold:      6,
	},
	{
		desc:            "scale l0Sensitivity",
		l0Sensitivity:   2,
		lInfSensitivity: 1,
		epsilon:         ln3,
		// deltaNoise is chosen to get a sigma of 1.
		deltaNoise:     0.26546844106038714,
		deltaThreshold: 0.022828893856,
		threshold:      3.275415487306,
	},
	{
		desc:            "small deltaThreshold",
		l0Sensitivity:   1,
		lInfSensitivity: 1,
		epsilon:         ln3,
		// deltaNoise is chosen to get a sigma of 1.
		deltaNoise: 0.10985556344445052,
		// 3e-5 is an approximate 1-sided tail probability of landing 4 standard
		// deviations from the mean of a Gaussian distribution.
		deltaThreshold: 3e-5,
		threshold:      5.012810811118,
	},
}

func TestThresholdGaussian(t *testing.T) {
	for _, tc := range thresholdGaussianTestCases {
		t.Run(tc.desc, func(t *testing.T) {
			gotThreshold := gauss.Threshold(tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.deltaNoise, tc.deltaThreshold)
			if math.Abs(gotThreshold-tc.threshold) > 1e-10 {
				t.Errorf("Got threshold: %0.12f, want threshold: %0.12f", gotThreshold, tc.threshold)
			}
		})
	}
}

func TestDeltaForThresholdGaussian(t *testing.T) {
	for _, tc := range thresholdGaussianTestCases {
		t.Run(tc.desc, func(t *testing.T) {
			gotDelta := gauss.(gaussian).DeltaForThreshold(tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.deltaNoise, tc.threshold)
			if math.Abs(gotDelta-tc.deltaThreshold) > 1e-10 {
				t.Errorf("Got delta: %0.12f, want delta: %0.12f", gotDelta, tc.deltaThreshold)
			}
		})
	}
}
