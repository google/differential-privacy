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
			sigma := SigmaForGaussian(tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.delta)
			t.Errorf("btw, true sigma is %f, squares to %f", sigma, sigma*sigma)
		}
	}
}

func TestSymmetricBinomialStatistics(t *testing.T) {
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
	//   sigma ↦ DeltaForGaussian(sigma, l2Sensitivity, epsilon).

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
			gotSigma := SigmaForGaussian(tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, deltaTight)
			if !(tc.sigma <= gotSigma && gotSigma <= (1+gaussianSigmaAccuracy)*tc.sigma) {
				t.Errorf("Got sigma: %f, want sigma in [%f, %f]", gotSigma, tc.sigma, (1+gaussianSigmaAccuracy)*tc.sigma)

			}
		})
	}
}

// This tests any logic that we need to special case for computing sigma (e.g.,
// precondition checking and boundary conditions).
func TestSigmaForGaussianWithDeltaOf1(t *testing.T) {
	got := SigmaForGaussian(1 /* l0 */, 1 /* lInf */, 0 /* ε */, 1 /* δ */)
	if got != 0 {
		t.Errorf("Got sigma: %f, want sigma: 0,", got)
	}
}

var thresholdGaussianTestCases = []struct {
	desc            string
	l0Sensitivity   int64
	lInfSensitivity float64
	epsilon         float64
	noiseDelta      float64
	thresholdDelta  float64
	threshold       float64
}{
	{
		desc:            "simple values",
		l0Sensitivity:   1,
		lInfSensitivity: 1,
		epsilon:         ln3,
		// noiseDelta is chosen to get a sigma of 1.
		noiseDelta: 0.10985556344445052,
		// 0.022750131948 is the 1-sided tail probability of landing more than 2
		// standard deviations from the mean of the Gaussian distribution.
		thresholdDelta: 0.022750131948,
		threshold:      3,
	},
	{
		desc:            "scale lInfSensitivity",
		l0Sensitivity:   1,
		lInfSensitivity: 0.5,
		epsilon:         ln3,
		// noiseDelta is chosen to get a sigma of 1.
		noiseDelta:     0.0041597422340007885,
		thresholdDelta: 0.000232629079,
		threshold:      4,
	},
	{
		desc:            "scale lInfSensitivity and sigma",
		l0Sensitivity:   1,
		lInfSensitivity: 2,
		epsilon:         ln3,
		// noiseDelta is chosen to get a sigma of 2.
		noiseDelta:     0.10985556344445052,
		thresholdDelta: 0.022750131948,
		threshold:      6,
	},
	{
		desc:            "scale l0Sensitivity",
		l0Sensitivity:   2,
		lInfSensitivity: 1,
		epsilon:         ln3,
		// noiseDelta is chosen to get a sigma of 1.
		noiseDelta:     0.26546844106038714,
		thresholdDelta: 0.022828893856,
		threshold:      3.275415487306,
	},
	{
		desc:            "small thresholdDelta",
		l0Sensitivity:   1,
		lInfSensitivity: 1,
		epsilon:         ln3,
		// noiseDelta is chosen to get a sigma of 1.
		noiseDelta: 0.10985556344445052,
		// 3e-5 is an approximate 1-sided tail probability of landing 4 standard
		// deviations from the mean of a Gaussian distribution.
		thresholdDelta: 3e-5,
		threshold:      5.012810811118,
	},
}

func TestThresholdGaussian(t *testing.T) {
	for _, tc := range thresholdGaussianTestCases {
		t.Run(tc.desc, func(t *testing.T) {
			gotThreshold := gauss.Threshold(tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.noiseDelta, tc.thresholdDelta)
			if math.Abs(gotThreshold-tc.threshold) > 1e-10 {
				t.Errorf("Got threshold: %0.12f, want threshold: %0.12f", gotThreshold, tc.threshold)
			}
		})
	}
}

func TestDeltaForThresholdGaussian(t *testing.T) {
	for _, tc := range thresholdGaussianTestCases {
		t.Run(tc.desc, func(t *testing.T) {
			gotDelta := gauss.(gaussian).DeltaForThreshold(tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.noiseDelta, tc.threshold)
			if math.Abs(gotDelta-tc.thresholdDelta) > 1e-10 {
				t.Errorf("Got delta: %0.12f, want delta: %0.12f", gotDelta, tc.thresholdDelta)
			}
		})
	}
}

func TestInverseCDFGaussian(t *testing.T) {
	for _, tc := range []struct {
		desc           string
		sigma, p, want float64 // Where p is equal to alpha/2.
	}{

		{
			desc:  "Arbitrary input test",
			sigma: 1.0,
			p:     0.05,
			want:  -1.6448536,
		},
		{
			desc:  "Arbitrary input test",
			sigma: 2.342354,
			p:     0.0240299,
			want:  -4.6304574,
		},
		{
			desc:  "Arbitrary input test",
			sigma: 0.3,
			p:     0.07534589,
			want:  -0.43112767,
		},
		{
			desc:  "Edge case test with low alpha",
			sigma: 0.356,
			p:     10e-10,
			want:  -2.1352193,
		},
		{
			desc:  "Edge case test with high alpha",
			sigma: 0.84,
			p:     1 - 10e-10,
			want:  5.0381579,
		},
		// For p = 0.5, the result should be the mean regardless of sigma.
		{
			desc:  "Test with p = 0.5",
			sigma: 0.3,
			p:     0.5,
			want:  0.0,
		},
		{
			desc:  "Test with p = 0.5",
			sigma: 0.8235243,
			p:     0.5,
			want:  0.0,
		},
	} {
		Zc := inverseCDFGaussian(tc.sigma, tc.p)
		if !approxEqual(Zc, tc.want) {
			t.Errorf("inverseCDFGaussian(%f, %f) = %0.10f, want %0.10f, desc: %s", tc.sigma, tc.p, Zc, tc.want, tc.desc)
		}
	}
}

func TestComputeConfidenceIntervalGaussian(t *testing.T) {
	// Tests for ComputeConfidenceIntervalGaussian function.
	for _, tc := range []struct {
		desc    string
		noisedX float64
		alpha   float64
		sigma   float64
		want    ConfidenceInterval
	}{
		{
			desc:    "Arbitrary input test",
			noisedX: 21.0,
			sigma:   1.0,
			alpha:   0.05,
			want:    ConfidenceInterval{19.040036, 22.959964},
		},
		{
			desc:    "Arbitrary input test",
			noisedX: 40.003,
			sigma:   0.333,
			alpha:   1 - 0.888,
			want:    ConfidenceInterval{39.473774, 40.532226},
		},
		{
			desc:    "Arbitrary input test",
			noisedX: 0.1,
			sigma:   0.292929,
			alpha:   1 - 0.888,
			want:    ConfidenceInterval{-0.36554256, 0.56554256},
		},
		{
			desc:    "Arbitrary input test",
			noisedX: 99.98989898,
			sigma:   15423235,
			alpha:   1 - 0.111,
			want:    ConfidenceInterval{-2.1525159e+06, 2.1527159e+06},
		},
		{
			desc:    "Low confidence level",
			noisedX: 100.0,
			sigma:   10.0,
			alpha:   1 - 1e-10,
			want:    ConfidenceInterval{99.999999, 100.00000},
		},
		{
			desc:    "High confidence level",
			noisedX: 100.0,
			sigma:   10.0,
			alpha:   1e-10,
			want:    ConfidenceInterval{35.330489, 164.66951},
		},
		// Testing that bounds are accurate for abs(bound) < 2^53
		{
			desc:    "Large positive noisedX",
			noisedX: 38475693.0,
			sigma:   2.8469244,
			alpha:   0.1,
			want:    ConfidenceInterval{38475688.3, 38475697.7},
		},
		{
			desc:    "Large negative noisedX",
			noisedX: -38475693.0,
			sigma:   2.8469244,
			alpha:   0.1,
			want:    ConfidenceInterval{-38475697.7, -38475688.3},
		},
	} {
		result := computeConfidenceIntervalGaussian(tc.noisedX, tc.sigma, tc.alpha)
		if !approxEqual(result.LowerBound, tc.want.LowerBound) {
			t.Errorf("computeConfidenceIntervalGaussian(%f, %f, %f)=%0.10f, want %0.10f, desc %s, LowerBounds are not equal",
				tc.noisedX, tc.alpha, tc.sigma, result.LowerBound, tc.want.LowerBound, tc.desc)
		}
		if !approxEqual(result.UpperBound, tc.want.UpperBound) {
			t.Errorf("computeConfidenceIntervalGaussian(%f, %f, %f)=%0.10f, want %0.10f, desc %s, UpperBounds are not equal",
				tc.noisedX, tc.alpha, tc.sigma, result.UpperBound, tc.want.UpperBound, tc.desc)
		}
	}
}

func TestComputeConfidenceIntervalInt64Gaussian(t *testing.T) {
	for _, tc := range []struct {
		desc                                    string
		noisedX, l0Sensitivity, lInfSensitivity int64
		epsilon, delta, alpha                   float64
		want                                    ConfidenceInterval
	}{
		{
			desc:            "Arbitrary test",
			noisedX:         70,
			l0Sensitivity:   6,
			lInfSensitivity: 10,
			epsilon:         0.3,
			delta:           0.1,
			alpha:           0.2,
			want:            ConfidenceInterval{8, 132},
		},
		{
			desc:            "Arbitrary test",
			noisedX:         1,
			l0Sensitivity:   1,
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			alpha:           0.152145599,
			want:            ConfidenceInterval{-5, 7},
		},
		{
			desc:            "Arbitrary test with high alpha",
			noisedX:         70,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			alpha:           1 - 7.856382354e-10,
			want:            ConfidenceInterval{70, 70},
		},
		{
			desc:            "Arbitrary test with low alpha",
			noisedX:         70,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			alpha:           7.856382354e-10,
			want:            ConfidenceInterval{-97, 237},
		},
		// Tests for nextSmallerFloat64 and nextLargerFloat64.
		{
			desc: "Large positive noisedX",
			// Distance to neighbouring float64 values is greater than half the size of the confidence interval.
			noisedX:         (1 << 58),
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			delta:           0.1,
			alpha:           0.1,
			want:            ConfidenceInterval{math.Nextafter(1<<58, math.Inf(-1)), math.Nextafter(1<<58, math.Inf(1))},
		},
		{
			desc: "Large negative noisedX",
			// Distance to neighbouring float64 values is greater than half the size of the confidence interval.
			noisedX:         -(1 << 58),
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			delta:           0.1,
			alpha:           0.1,
			want:            ConfidenceInterval{math.Nextafter(-1<<58, math.Inf(-1)), math.Nextafter(-1<<58, math.Inf(1))},
		},
	} {
		got, err := gauss.ComputeConfidenceIntervalInt64(tc.noisedX, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.delta, tc.alpha)
		if err != nil {
			t.Errorf("ComputeConfidenceIntervalInt64: when %s got err %v", tc.desc, err)
		}
		if got.LowerBound != tc.want.LowerBound {
			t.Errorf("ComputeConfidenceIntervalInt64(%d, %d, %d, %f, %f, %f)=%f, want %f, desc %s, LowerBounds are not equal",
				tc.noisedX, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.delta, tc.alpha, got.LowerBound, tc.want.LowerBound, tc.desc)
		}
		if got.UpperBound != tc.want.UpperBound {
			t.Errorf("ComputeConfidenceIntervalInt64(%d, %d, %d, %f, %f, %f)=%f, want %f, desc %s, UpperBounds are not equal",
				tc.noisedX, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.delta, tc.alpha, got.UpperBound, tc.want.UpperBound, tc.desc)
		}
	}
}

func TestComputeConfidenceIntervalInt64GaussianArgumentCheck(t *testing.T) {
	for _, tc := range []struct {
		desc                                    string
		noisedX, l0Sensitivity, lInfSensitivity int64
		epsilon, delta, alpha                   float64
	}{
		{
			desc:            "Testing alpha bigger than 1",
			noisedX:         1,
			l0Sensitivity:   1,
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			alpha:           1.2, // alpha should not be larger than 1.
		},
		{
			desc:            "Testing negative alpha",
			noisedX:         1,
			l0Sensitivity:   1,
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			alpha:           -5, // alpha should not be smaller than 0.
		},
		{
			desc:            "Testing negative l0Sensitivity",
			noisedX:         1,
			l0Sensitivity:   -1, // l0Sensitivity should be strictly positive.
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing zero l0Sensitivity",
			noisedX:         1,
			l0Sensitivity:   0, // l0Sensitivity should be strictly positive.
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing negative lInfSensitivity",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: -4, // lInfSensitivity should be strictly positive.
			epsilon:         0.5,
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing zero lInfSensitivity",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 0, // lInfSensitivity should be strictly positive.
			epsilon:         0.5,
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing negative epsilon",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         -0.05, // epsilon should be strictly positive.
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing infinite epsilon",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         math.Inf(1), // epsilon cannot be infinite.
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing NaN epsilon",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         math.Inf(1), // epsilon cannot be NaN.
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing negative delta",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         0.05,
			delta:           -0.9, // delta should be strictly positive and smaller than 1.
			alpha:           0.2,
		},
		{
			desc:            "Testing bigger than 1 delta",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         0.05,
			delta:           10, // delta should be strictly positive and smaller than 1.
			alpha:           0.2,
		},
		{
			desc:            "Testing zero delta",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         0.05,
			delta:           10, // delta should be strictly positive and smaller than 1.
			alpha:           0.2,
		},
		{
			desc:          "Arbitrary test with 0 alpha",
			noisedX:       70,
			l0Sensitivity: 5,
			epsilon:       0.8,
			delta:         0.8,
			alpha:         0,
		},
		{
			desc:            "Arbitrary test with 1 alpha",
			noisedX:         70,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			alpha:           1,
		},
		{
			desc:            "Arbitrary test with negative alpha",
			noisedX:         70,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			alpha:           -1,
		},
		{
			desc:            "Arbitrary test with greater than 1 alpha",
			noisedX:         70,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			alpha:           10,
		},
		{
			desc:            "Arbitrary test with NaN alpha",
			noisedX:         70,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			alpha:           math.NaN(),
		},
	} {
		_, err := gauss.ComputeConfidenceIntervalInt64(tc.noisedX, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.delta, tc.alpha)
		if err == nil {
			t.Errorf("ComputeConfidenceIntervalInt64: when %s no error was returned, expected error", tc.desc)
		}
	}
}

func TestComputeConfidenceIntervalFloat64GaussianArgumentCheck(t *testing.T) {
	for _, tc := range []struct {
		desc                                   string
		noisedX                                float64
		l0Sensitivity                          int64
		lInfSensitivity, epsilon, delta, alpha float64
		want                                   ConfidenceInterval
	}{
		{
			desc:            "Arbitrary test with 0 alpha",
			noisedX:         598.21548,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			alpha:           0,
		},
		{
			desc:            "Arbitrary test with 1 alpha",
			noisedX:         70.0,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			alpha:           1,
		},
		{
			desc:            "Testing alpha bigger than 1",
			noisedX:         1,
			l0Sensitivity:   1,
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			alpha:           1.2, // alpha should not be smaller than 0.
		},
		{
			desc:            "Testing negative alpha",
			noisedX:         1,
			l0Sensitivity:   1,
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			alpha:           -5, // alpha should not be smaller than 0.
		},
		{
			desc:            "Testing negative l0Sensitivity",
			noisedX:         1,
			l0Sensitivity:   -1, // l0Sensitivity should be strictly positive.
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing zero l0Sensitivity",
			noisedX:         1,
			l0Sensitivity:   0, // l0Sensitivity should be strictly positive.
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing negative lInfSensitivity",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: -4, // lInfSensitivity should be strictly positive.
			epsilon:         0.5,
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing zero lInfSensitivity",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 0, // lInfSensitivity should be strictly positive.
			epsilon:         0.5,
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing positive infinity lInfSensitivity",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: math.Inf(1), // lInfSensitivity should not be infinite.
			epsilon:         0.5,
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing NaN lInfSensitivity",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: math.NaN(), // lInfSensitivity cannot be NaN.
			epsilon:         0.5,
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing negative epsilon",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         -0.05, // epsilon should be strictly positive.
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing infinite epsilon",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         math.Inf(1), // epsilon should not be infinite.
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing NaN epsilon",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         math.NaN(), // epsilon cannot be NaN.
			delta:           0.9,
			alpha:           0.2,
		},
		{
			desc:            "Testing negative delta",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         0.05,
			delta:           -0.9, // delta should be strictly positive and smaller than 1.
			alpha:           0.2,
		},
		{
			desc:            "Testing bigger than 1 delta",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         0.05,
			delta:           10, // delta should be strictly positive and smaller than 1.
			alpha:           0.2,
		},
		{
			desc:            "Testing zero delta",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         0.05,
			delta:           0, // delta should be strictly positive and smaller than 1.
			alpha:           0.2,
		},
		{
			desc:            "Testing infinite delta",
			noisedX:         1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         0.05,
			delta:           math.Inf(1), // delta should be strictly positive and smaller than 1.
			alpha:           0.2,
		},
	} {
		_, err := gauss.ComputeConfidenceIntervalFloat64(tc.noisedX, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.delta, tc.alpha)
		if err == nil {
			t.Errorf("ComputeConfidenceIntervalFloat64: when %s no error was returned, expected error", tc.desc)
		}
	}
}
