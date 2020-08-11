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

func TestInverseCDFGaussian(t *testing.T) {
	for _, tc := range []struct {
		desc                         string
		sigma, confidenceLevel, want float64
	}{

		{
			desc:            "Abitrary input test",
			sigma:           1,
			confidenceLevel: 0.95,
			want:            1.64485362695,
		},
		{
			desc:            "Abitrary input test",
			sigma:           2.342354,
			confidenceLevel: 0.8734521154362147425,
			want:            2.67698807013,
		},
		{
			desc:            "Abitrary input test",
			sigma:           0.3,
			confidenceLevel: 0.75345892435835346586,
			want:            0.205624466704,
		},
		{
			desc:            "Edge case test with probability = 0",
			sigma:           0.3,
			confidenceLevel: 0,
			want:            math.Inf(-1),
		},
		{
			desc:            "Edge case test with probability = 1",
			sigma:           0.8,
			confidenceLevel: 1,
			want:            math.Inf(1),
		},
		{
			desc:            "Edge case test with low probability",
			sigma:           0.356,
			confidenceLevel: 0.05,
			want:            -0.585567891195,
		},
		{
			desc:            "Edge case test with high probability",
			sigma:           0.84,
			confidenceLevel: 0.99,
			want:            1.95413221419,
		},
		{
			desc:            "Test with probability of 1",
			sigma:           0.356,
			confidenceLevel: 1,
			want:            math.Inf(1),
		},
		{
			desc:            "Test with probability of 0",
			sigma:           0.84,
			confidenceLevel: 0,
			want:            math.Inf(-1),
		},
		// For a probablity of 0.5 the result should be the mean regardless of lambda
		{
			desc:            "Test with probability = 0.5",
			sigma:           0.3,
			confidenceLevel: 0.5,
			want:            0,
		},
		{
			desc:            "Test with probability = 0.5",
			sigma:           0.8235243,
			confidenceLevel: 0.5,
			want:            0,
		},
	} {

		Zc := inverseCDFGaussian(tc.sigma, tc.confidenceLevel)
		if !(approxEqual(Zc, tc.want)) {
			t.Errorf(" TestInverseCDFGaussian(%f, %f) = %0.12f, want %0.12f, desc: %s", tc.sigma, tc.confidenceLevel, Zc, tc.want, tc.desc)

		}
	}
}

func TestConfidenceIntervalGaussian(t *testing.T) {
	// Tests for getConfidenceIntervalGaussian function.
	for _, tc := range []struct {
		desc            string
		noisedValue     float64
		confidenceLevel float64
		sigma           float64
		want            ConfidenceInterval
	}{
		{
			desc:            "getConfidenceIntervalGaussian arbitrary input test",
			noisedValue:     21,
			sigma:           1,
			confidenceLevel: 0.95,
			want:            ConfidenceInterval{19.0400360155, 22.9599639845},
		},
		{
			desc:            "getConfidenceIntervalGaussian arbitrary input test",
			noisedValue:     40.003,
			sigma:           0.333,
			confidenceLevel: 0.888,
			want:            ConfidenceInterval{39.473773903501886, 40.532226096498114},
		},
		{
			desc:            "getConfidenceIntervalGaussian arbitrary input test",
			noisedValue:     0.1,
			sigma:           0.292929,
			confidenceLevel: 0.888,
			want:            ConfidenceInterval{-0.36554255621950726, 0.5655425562195072},
		},
		{
			desc:            "getConfidenceIntervalGaussian arbitrary input test",
			noisedValue:     99.98989898,
			sigma:           15423235,
			confidenceLevel: 0.111,
			want:            ConfidenceInterval{-2.1525159435946424e+06, 2.1527159233926027e+06},
		},
		{
			desc:            "Low confidence level",
			noisedValue:     100,
			sigma:           10,
			confidenceLevel: 10e-10,
			want:            ConfidenceInterval{99.99999998746686, 100.00000001253314},
		},
		{
			desc:            "High confidence level",
			noisedValue:     100,
			sigma:           10,
			confidenceLevel: 1 - 10e-10,
			want:            ConfidenceInterval{38.90589790616554, 161.09410209383446},
		},
	} {
		result := getConfidenceIntervalGaussian(tc.noisedValue, tc.sigma, tc.confidenceLevel)
		if !approxEqual(result.LowerBound, tc.want.LowerBound) {
			t.Errorf("TestConfidenceIntervalGaussian(%f, %f, %f)=%0.10f, want %0.10f, desc %s, LowerBound is not equal",
				tc.noisedValue, tc.confidenceLevel, tc.sigma,
				result.LowerBound, tc.want.LowerBound, tc.desc)
		}
		if !approxEqual(result.UpperBound, tc.want.UpperBound) {
			t.Errorf("TestConfidenceIntervalLaplace(%f, %f, %f)=%0.10f, want %0.10f, desc %s, UpperBound is not equal",
				tc.noisedValue, tc.confidenceLevel, tc.sigma,
				result.UpperBound, tc.want.UpperBound, tc.desc)
		}
	}

}

func TestConfidenceIntervalInt64(t *testing.T) {
	for _, tc := range []struct {
		desc                                        string
		noisedValue, l0Sensitivity, lInfSensitivity int64
		epsilon, delta, confidenceLevel             float64
		want                                        ConfidenceInterval
		wantErr                                     bool
	}{
		{
			desc:            "Arbitrary test",
			noisedValue:     70,
			l0Sensitivity:   6,
			lInfSensitivity: 10,
			epsilon:         0.3,
			delta:           0.1,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{110.0, 30.0},
			wantErr:         false,
		},
		{
			desc:            "Arbitrary test",
			noisedValue:     1,
			l0Sensitivity:   1,
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: 0.9,
			want:            ConfidenceInterval{-4.0, 6.0},
			wantErr:         false,
		},
		// Testing checkArgsConfidenceIntervalGaussian.
		{
			desc:            "Testing confidence level bigger than 1",
			noisedValue:     1,
			l0Sensitivity:   1,
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: 1.2, // The confidence level should not be bigger than 1.
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing negative confidence level",
			noisedValue:     1,
			l0Sensitivity:   1,
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: -5, // The confidence level should not be smaller than 0.
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing negative l0Sensitivity",
			noisedValue:     1,
			l0Sensitivity:   -1, // l0Sensitivity should be strictly positive.
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing zero l0Sensitivity",
			noisedValue:     1,
			l0Sensitivity:   0, // l0Sensitivity should be strictly positive.
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing negative lInfSensitivity",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: -4, // lInfSensitivity should be strictly positive.
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing zero lInfSensitivity",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 0, // lInfSensitivity should be strictly positive.
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing negative epsilon",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         -0.05, // epsilon should be strictly positive.
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing infinite epsilon",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         math.Inf(1), // epsilon cannot be infinite.
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing NaN epsilon",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         math.Inf(1), // epsilon cannot be NaN.
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing negative dela",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         0.05,
			delta:           -0.9, // delta should be strictly positive and smaller than 1.
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing bigger than 1 delta",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         0.05,
			delta:           10, // delta should be strictly positive and smaller than 1.
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing zero delta",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         0.05,
			delta:           10, // delta should be strictly positive and smaller than 1.
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Arbitrary test with 0 probability",
			noisedValue:     70,
			l0Sensitivity:   5,
			epsilon:         0.8,
			delta:           0.8,
			confidenceLevel: 0,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Arbitrary test with 1 probability",
			noisedValue:     70,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			confidenceLevel: 1,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Arbitrary test with negative probability",
			noisedValue:     70,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			confidenceLevel: -1,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Arbitrary test with greater than 1 probability",
			noisedValue:     70,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			confidenceLevel: 10,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Arbitrary test with NaN probability",
			noisedValue:     70,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			confidenceLevel: math.NaN(),
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
	} {
		got, err := gauss.ConfidenceIntervalInt64(tc.noisedValue, tc.l0Sensitivity, tc.lInfSensitivity,
			tc.epsilon, tc.delta, tc.confidenceLevel)
		if (err != nil) != tc.wantErr {
			t.Errorf("ConfidenceIntervalInt64: when %s for err got %v", tc.desc, err)
			if got.LowerBound != tc.want.LowerBound {
				t.Errorf("TestConfidenceIntervalInt64(%d, %d, %d, %f, %f, %f)=%f, want %f, desc %s, LowerBound is not equal",
					tc.noisedValue, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.delta, tc.confidenceLevel,
					got.LowerBound, tc.want.LowerBound, tc.desc)
			}
			if got.UpperBound != tc.want.UpperBound {
				t.Errorf("TestConfidenceIntervalInt64(%d, %d, %d, %f, %f, %f)=%f, want %f, desc %s, UpperBound is not equal",
					tc.noisedValue, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.delta, tc.confidenceLevel,
					got.UpperBound, tc.want.UpperBound, tc.desc)
			}
		}
	}
}

func TestConfidenceIntervalFloat64(t *testing.T) {
	for _, tc := range []struct {
		desc                                             string
		noisedValue                                      float64
		l0Sensitivity                                    int64
		lInfSensitivity, epsilon, delta, confidenceLevel float64
		want                                             ConfidenceInterval
		wantErr                                          bool
	}{
		{
			desc:            "Arbitrary test",
			noisedValue:     70,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{92.80911868743263, 47.19088131256736},
			wantErr:         false,
		},
		{
			desc:            "Arbitrary test with 0 probability",
			noisedValue:     70,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			confidenceLevel: 0,
			want:            ConfidenceInterval{math.Inf(1), math.Inf(-1)},
			wantErr:         true,
		},
		{
			desc:            "Arbitrary test with 1 probability",
			noisedValue:     70,
			l0Sensitivity:   5,
			lInfSensitivity: 36,
			epsilon:         0.8,
			delta:           0.8,
			confidenceLevel: 1,
			want:            ConfidenceInterval{math.Inf(-1), math.Inf(1)},
			wantErr:         true,
		},
		{
			desc:            "Arbitrary test",
			noisedValue:     60,
			l0Sensitivity:   1,
			lInfSensitivity: 5,
			epsilon:         0.333,
			delta:           0.9,
			confidenceLevel: 0.7,
			want:            ConfidenceInterval{59.23887669725359, 60.76112330274641},
			wantErr:         false,
		},
		// Testing checkArgsConfidenceIntervalGaussian
		{
			desc:            "Testing confidence level bigger than 1",
			noisedValue:     1,
			l0Sensitivity:   1,
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: 1.2, // The confidence level should not be smaller than 0.
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing negative confidence level",
			noisedValue:     1,
			l0Sensitivity:   1,
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: -5, // The confidence level should not be smaller than 0.
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing negative l0Sensitivity",
			noisedValue:     1,
			l0Sensitivity:   -1, // l0Sensitivity should be strictly positive.
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing zero l0Sensitivity",
			noisedValue:     1,
			l0Sensitivity:   0, // l0Sensitivity should be strictly positive.
			lInfSensitivity: 15,
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing negative lInfSensitivity",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: -4, // lInfSensitivity should be strictly positive.
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing zero lInfSensitivity",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 0, // lInfSensitivity should be strictly positive.
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing positive infinity lInfSensitivity",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: math.Inf(1), // lInfSensitivity should not be infinite.
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing NaN lInfSensitivity",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: math.NaN(), // lInfSensitivity cannot be NaN.
			epsilon:         0.5,
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing negative epsilon",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         -0.05, // epsilon should be strictly positive.
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing infinite epsilon",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         math.Inf(1), // epsilon should not be infinite.
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing NaN epsilon",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         math.NaN(), // epsilon cannot be NaN.
			delta:           0.9,
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing negative dela",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         0.05,
			delta:           -0.9, // delta should be strictly positive and smaller than 1.
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing bigger than 1 delta",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         0.05,
			delta:           10, // delta should be strictly positive and smaller than 1.
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing zero delta",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         0.05,
			delta:           0, // delta should be strictly positive and smaller than 1.
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
		{
			desc:            "Testing infinite delta",
			noisedValue:     1,
			l0Sensitivity:   4,
			lInfSensitivity: 5,
			epsilon:         0.05,
			delta:           math.Inf(1), // delta should be strictly positive and smaller than 1.
			confidenceLevel: 0.2,
			want:            ConfidenceInterval{},
			wantErr:         true,
		},
	} {
		got, err := gauss.ConfidenceIntervalFloat64(tc.noisedValue, tc.l0Sensitivity, tc.lInfSensitivity,
			tc.epsilon, tc.delta, tc.confidenceLevel)
		if (err != nil) != tc.wantErr {
			t.Errorf("ConfidenceIntervalFloat64: when %s for err got %v", tc.desc, err)

			if !approxEqual(got.LowerBound, tc.want.LowerBound) {
				t.Errorf("TestConfidenceIntervalFloat64(%f, %d, %f, %f, %f)=%0.10f, want %0.10f, desc %s, LowerBound is not equal",
					tc.noisedValue, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.confidenceLevel,
					got.UpperBound, tc.want.UpperBound, tc.desc)
			}
			if !approxEqual(got.UpperBound, tc.want.UpperBound) {
				t.Errorf("TestConfidenceIntervalFloat64(%f, %d, %f, %f, %f)=%0.10f, want %0.10f, desc %s, UpperBound is not equal",
					tc.noisedValue, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.confidenceLevel,
					got.LowerBound, tc.want.LowerBound, tc.desc)
			}
		}
	}
}
