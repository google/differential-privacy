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

func TestLaplaceStatistics(t *testing.T) {
	const numberOfSamples = 125000
	for _, tc := range []struct {
		l0Sensitivity                            int64
		lInfSensitivity, epsilon, mean, variance float64
	}{
		{
			l0Sensitivity:   1,
			lInfSensitivity: 1.0,
			epsilon:         1.0,
			mean:            0.0,
			variance:        2.0,
		},
		{
			l0Sensitivity:   1,
			lInfSensitivity: 1.0,
			epsilon:         ln3,
			mean:            0.0,
			variance:        2.0 / (ln3 * ln3),
		},
		{
			l0Sensitivity:   1,
			lInfSensitivity: 1.0,
			epsilon:         ln3,
			mean:            45941223.02107,
			variance:        2.0 / (ln3 * ln3),
		},
		{
			l0Sensitivity:   1,
			lInfSensitivity: 1.0,
			epsilon:         2.0 * ln3,
			mean:            0.0,
			variance:        2.0 / (2.0 * ln3 * 2.0 * ln3),
		},
		{
			l0Sensitivity:   1,
			lInfSensitivity: 2.0,
			epsilon:         2.0 * ln3,
			mean:            0.0,
			variance:        2.0 / (ln3 * ln3),
		},
		{
			l0Sensitivity:   2,
			lInfSensitivity: 1.0,
			epsilon:         2.0 * ln3,
			mean:            0.0,
			variance:        2.0 / (ln3 * ln3),
		},
	} {
		noisedSamples := make(stat.Float64Slice, numberOfSamples)
		for i := 0; i < numberOfSamples; i++ {
			noisedSamples[i] = lap.AddNoiseFloat64(tc.mean, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, 0)
		}
		sampleMean, sampleVariance := stat.Mean(noisedSamples), stat.Variance(noisedSamples)
		// Assuming that the Laplace samples have a mean of 0 and the specified variance of tc.variance,
		// sampleMeanFloat64 and sampleMeanInt64 are approximately Gaussian distributed with a mean of 0
		// and standard deviation of sqrt(tc.variance⁻ / numberOfSamples).
		//
		// The meanErrorTolerance is set to the 99.9995% quantile of the anticipated distribution. Thus,
		// the test falsely rejects with a probability of 10⁻⁵.
		meanErrorTolerance := 4.41717 * math.Sqrt(tc.variance/float64(numberOfSamples))
		// Assuming that the Laplace samples have the specified variance of tc.variance, sampleVarianceFloat64
		// and sampleVarianceInt64 are approximately Gaussian distributed with a mean of tc.variance and a
		// standard deviation of sqrt(5) * tc.variance / sqrt(numberOfSamples).
		//
		// The varianceErrorTolerance is set to the 99.9995% quantile of the anticipated distribution. Thus,
		// the test falsely rejects with a probability of 10⁻⁵.
		varianceErrorTolerance := 4.41717 * math.Sqrt(5.0) * tc.variance / math.Sqrt(float64(numberOfSamples))

		if !nearEqual(sampleMean, tc.mean, meanErrorTolerance) {
			t.Errorf("float64 got mean = %f, want %f (parameters %+v)", sampleMean, tc.mean, tc)
		}
		if !nearEqual(sampleVariance, tc.variance, varianceErrorTolerance) {
			t.Errorf("float64 got variance = %f, want %f (parameters %+v)", sampleVariance, tc.variance, tc)
		}
	}
}

func TestThresholdLaplace(t *testing.T) {
	// For the l0Sensitivity=1 cases, we make certain that we have implemented
	// both tails of the Laplace distribution. To do so, we write tests in pairs by
	// reflecting the value of delta around the axis 0.5 and the threshold "want"
	// value around the axis lInfSensitivity.
	//
	// This symmetry in the CDF is exploited by implicitly reflecting
	// partitionDelta (the per-partition delta) around the 0.5 axis. When
	// l0Sensitivity is 1, this can be easily expressed in tests since delta ==
	// partitionDelta. When l0Sensitivity != 1, it is no longer easy to express.
	for _, tc := range []struct {
		l0Sensitivity                         int64
		lInfSensitivity, epsilon, delta, want float64
	}{
		// Base test case
		{1, 1, ln3, 1e-10, 21.33},
		{1, 1, ln3, 1 - 1e-10, -19.33},
		// Scale lambda
		{1, 1, 2 * ln3, 1e-10, 11.16},
		{1, 1, 2 * ln3, 1 - 1e-10, -9.16},
		// Scale lInfSensitivity and lambda
		{1, 10, ln3, 1e-10, 213.28},
		{1, 10, ln3, 1 - 1e-10, -193.28},
		// Scale l0Sensitivity
		{10, 10, 10 * ln3, 1e-9, 213.28},
		{10, 10, 10 * ln3, 1 - 1e-9, -2.55}, // l0Sensitivity != 1, not expecting symmetry in "want" threshold around lInfSensitivity.
		// High precision delta case
		{1, 1, ln3, 1e-200, 419.55},
	} {
		got := lap.Threshold(tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, 0, tc.delta)
		if !nearEqual(got, tc.want, 0.01) {
			t.Errorf("ThresholdForLaplace(%d,%f,%f,%e)=%f, want %f", tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.delta, got, tc.want)
		}
	}
}

func TestDeltaForThresholdLaplace(t *testing.T) {
	// For the l0Sensitivity=1 cases, we make certain that we have implemented
	// both tails of the Laplace distribution. To do so, we write tests in pairs by
	// reflecting the "want" value of delta around the axis 0.5 and the threshold
	// value k around the axis lInfSensitivity.
	//
	// This symmetry in the CDF is exploited by implicitly reflecting
	// partitionDelta (the per-partition delta) around the 0.5 axis. When
	// l0Sensitivity is 1, this can be easily expressed in tests since delta ==
	// partitionDelta. When l0Sensitivity != 1, it is no longer easy to express.
	for _, tc := range []struct {
		l0Sensitivity                     int64
		lInfSensitivity, epsilon, k, want float64
	}{
		// Base test case
		{1, 1, ln3, 20, 4.3e-10},
		{1, 1, ln3, -18, 1 - 4.3e-10},
		// Scale lInfSensitivity, lambda, and k
		{1, 10, ln3, 200, 4.3e-10},
		{1, 10, ln3, -180, 1 - 4.3e-10},
		// Scale lambda and k
		{1, 1, 2 * ln3, 10, 1.29e-9},
		{1, 1, 2 * ln3, -8, 1 - 1.29e-9},
		// Scale l0Sensitivity
		{10, 1, 10 * ln3, 20, 4.3e-9},
		{10, 1, 10 * ln3, -18, 1}, // l0Sensitivity != 1, not expecting symmetry in "want" delta around 0.5.
		// High precision delta case
		{1, 1, ln3, 419.55, 1e-200},
	} {
		got := lap.(laplace).DeltaForThreshold(tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, 0, tc.k)
		if !nearEqual(got, tc.want, 1e-2*tc.want) {
			t.Errorf("ThresholdForLaplace(%d,%f,%f,%f)=%e, want %e", tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.k, got, tc.want)
		}
	}
}

func TestInverseCDFLaplace(t *testing.T) {
	for _, tc := range []struct {
		desc               string
		lambda, prob, want float64
	}{
		{
			desc:   "Arbitrary test",
			lambda: 4.0,
			prob:   0.78754042,
			want:   3.4234254,
		},
		{
			desc:   "Arbitrary test",
			lambda: 2.0,
			prob:   0.14796856,
			want:   -2.4352165,
		},
		// For a probability of 0.5, the result should be the zero regardless of lambda.
		{
			desc:   "0.5 Probability, output is zero",
			lambda: 5.0,
			prob:   0.5,
			want:   0,
		},
		{
			desc:   "0.5 Probability, output is zero",
			lambda: 10.0,
			prob:   0.5,
			want:   0,
		},
		// Tests for convergence to infinities with low and high probabilities.
		{
			desc:   "Low probability",
			lambda: 3.0,
			prob:   1.2375736e-15,
			want:   -100.89743,
		},
		{
			desc:   "High probability",
			lambda: 3.0,
			prob:   1 - 1.237574E-10,
			want:   66.358653,
		},
	} {
		got := inverseCDFLaplace(tc.lambda, tc.prob)
		if !approxEqual(got, tc.want) {
			t.Errorf("inverseCDFLaplace(%f,%f)=%0.12f, want %0.12f, desc: %s", tc.lambda,
				tc.prob, got, tc.want, tc.desc)
		}
	}
}

func TestComputeConfidenceIntervalLaplace(t *testing.T) {
	for _, tc := range []struct {
		desc          string
		noisedX       float64
		lambda, alpha float64
		want          ConfidenceInterval
	}{
		{
			desc:    "Arbitrary test",
			noisedX: 13.0,
			lambda:  27.333333,
			alpha:   0.05,
			want:    ConfidenceInterval{-68.883349, 94.883349},
		},
		{
			desc:    "Arbitrary test",
			noisedX: 83.1235,
			lambda:  60.0,
			alpha:   0.24,
			want:    ConfidenceInterval{-2.5034813, 168.75048},
		},
		{
			desc:    "Arbitrary test",
			noisedX: 5.0,
			lambda:  6.6666667,
			alpha:   0.6,
			want:    ConfidenceInterval{1.5944958, 8.40550416},
		},
		{
			desc:    "Arbitrary test",
			noisedX: 65.4621,
			lambda:  700.0,
			alpha:   0.8,
			want:    ConfidenceInterval{-90.738386, 221.66259},
		},
		{
			desc:    "Extremely low confidence level",
			noisedX: 0,
			lambda:  10.0,
			alpha:   1 - 3.5489574e-10,
			want:    ConfidenceInterval{-3.5489574e-9, 3.5489574e-9},
		},
		{
			desc:    "Extremely high confidence level",
			noisedX: 50.0,
			lambda:  10.0,
			alpha:   7.8563824e-10,
			want:    ConfidenceInterval{-159.64525, 259.64525},
		},
		// Testing that bounds are accurate for abs(bound) < 2^53
		{
			desc:    "Large positive noisedX",
			noisedX: 38475693.0,
			lambda:  10.0,
			alpha:   0.1,
			want:    ConfidenceInterval{38475670.0, 38475716.0},
		},
		{
			desc:    "Large negative noisedX",
			noisedX: -38475693.0,
			lambda:  10.0,
			alpha:   0.1,
			want:    ConfidenceInterval{-38475716.0, -38475670.0},
		},
	} {
		got := computeConfidenceIntervalLaplace(tc.noisedX, tc.lambda, tc.alpha)
		if !approxEqual(got.LowerBound, tc.want.LowerBound) {
			t.Errorf("computeConfidenceIntervalLaplace(%f, %f, %f)=%0.10f, want %0.10f, desc %s, LowerBounds are not equal",
				tc.noisedX, tc.lambda, tc.alpha, got.LowerBound, tc.want.LowerBound, tc.desc)
		}
		if !approxEqual(got.UpperBound, tc.want.UpperBound) {
			t.Errorf("computeConfidenceIntervalLaplace(%f, %f, %f)=%0.10f, want %0.10f, desc %s, UpperBounds are not equal",
				tc.noisedX, tc.lambda, tc.alpha, got.UpperBound, tc.want.UpperBound, tc.desc)
		}
	}
}

func TestComputeConfidenceIntervalFloat64LaplaceArgumentCheck(t *testing.T) {
	for _, tc := range []struct {
		desc                                   string
		noisedX                                float64
		l0Sensitivity                          int64
		lInfSensitivity, epsilon, alpha, delta float64
	}{
		{
			desc:            "Zero l0Sensitivity",
			noisedX:         0,
			l0Sensitivity:   0,
			lInfSensitivity: 1,
			epsilon:         0.1,
			alpha:           0.5,
		},
		{
			desc:            "Negative l0Sensitivity",
			noisedX:         0,
			l0Sensitivity:   -1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			alpha:           0.5,
		},
		{
			desc:            "Zero lInfSensitivity",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 0,
			epsilon:         0.1,
			alpha:           0.5,
		},
		{
			desc:            "Negative lInfSensitivity",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: -1,
			epsilon:         0.1,
			alpha:           0.5,
		},
		{
			desc:            "Infinite lInfSensitivity",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: math.Inf(1),
			epsilon:         0.1,
			alpha:           0.5,
		},
		{
			desc:            "NaN lInfSensitivity",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: math.NaN(),
			epsilon:         0.1,
			alpha:           0.5,
		},
		{
			desc:            "Epsilon less than 2^-50",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         1.0 / (1 << 51),
			alpha:           0.5,
		},
		{
			desc:            "Negative epsilon",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         -1,
			alpha:           0.5,
		},
		{
			desc:            "Infinite epsilon",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         math.Inf(1),
			alpha:           0.5,
		},
		{
			desc:            "NaN epsilon",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         math.NaN(),
			alpha:           0.5,
		},
		{
			desc:            "Non-zero delta",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			delta:           1,
			alpha:           0.5,
		},
		{
			desc:            "Zero alpha",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			alpha:           0,
		},
		{
			desc:            "Negative alpha",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			alpha:           -1,
		},
		{
			desc:            "1 alpha",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			alpha:           1,
		},
		{
			desc:            "Greater than 1 alpha",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			alpha:           2,
		},
		{
			desc:            "NaN alpha",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			alpha:           math.NaN(),
		},
	} {
		_, err := lap.ComputeConfidenceIntervalFloat64(tc.noisedX, tc.l0Sensitivity, tc.lInfSensitivity,
			tc.epsilon, tc.delta, tc.alpha)

		if err == nil {
			t.Errorf("ComputeConfidenceIntervalFloat64: when %s no error was returned, expected error", tc.desc)
		}
	}
}

func TestComputeConfidenceIntervalInt64Laplace(t *testing.T) {
	for _, tc := range []struct {
		desc                                    string
		noisedX, l0Sensitivity, lInfSensitivity int64
		epsilon, delta, alpha                   float64
		want                                    ConfidenceInterval
	}{
		{
			desc:            "Arbitrary test",
			noisedX:         -12,
			l0Sensitivity:   5,
			lInfSensitivity: 6,
			epsilon:         0.1,
			alpha:           0.8,
			want:            ConfidenceInterval{-79, 55},
		},
		// Tests for nextSmallerFloat64 and nextLargerFloat64.
		{
			desc: "Large positive noisedX",
			// Distance to neighbouring float64 values is greater than half the size of the confidence interval.
			noisedX:         (1 << 58),
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
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
			alpha:           0.1,
			want:            ConfidenceInterval{math.Nextafter(-1<<58, math.Inf(-1)), math.Nextafter(-1<<58, math.Inf(1))},
		},
	} {
		got, err := lap.ComputeConfidenceIntervalInt64(tc.noisedX, tc.l0Sensitivity, tc.lInfSensitivity,
			tc.epsilon, tc.delta, tc.alpha)
		if err != nil {
			t.Errorf("ComputeConfidenceIntervalInt64: when %v for err got %v", tc.desc, err)
		}
		if got.LowerBound != tc.want.LowerBound {
			t.Errorf("ComputeConfidenceIntervalInt64(%d, %d, %d, %f, %f)=%0.10f, want %0.10f, desc %s, LowerBounds are not equal",
				tc.noisedX, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.alpha, got.LowerBound, tc.want.LowerBound, tc.desc)
		}
		if got.UpperBound != tc.want.UpperBound {
			t.Errorf("ComputeConfidenceIntervalInt64(%d, %d, %d, %f, %f)=%0.10f, want %0.10f, desc %s, UpperBounds are not equal",
				tc.noisedX, tc.l0Sensitivity, tc.lInfSensitivity, tc.epsilon, tc.alpha, got.UpperBound, tc.want.UpperBound, tc.desc)
		}
	}
}

func TestComputeConfidenceIntervalInt64LaplaceArgumentCheck(t *testing.T) {
	for _, tc := range []struct {
		desc                                    string
		noisedX, l0Sensitivity, lInfSensitivity int64
		epsilon, alpha, delta                   float64
	}{
		{
			desc:            "Zero l0Sensitivity",
			noisedX:         0,
			l0Sensitivity:   0,
			lInfSensitivity: 1,
			epsilon:         0.1,
			alpha:           0.5,
		},
		{
			desc:            "Negative l0Sensitivity",
			noisedX:         0,
			l0Sensitivity:   -1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			alpha:           0.5,
		},
		{
			desc:            "Zero lInfSensitivity",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 0,
			epsilon:         0.1,
			alpha:           0.5,
		},
		{
			desc:            "Negative lInfSensitivity",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: -1,
			epsilon:         0.1,
			alpha:           0.5,
		},
		{
			desc:            "Epsilon less than 2^-50",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         1.0 / (1 << 51),
			alpha:           0.5,
		},
		{
			desc:            "Negative epsilon",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         -1,
			alpha:           0.5,
		},
		{
			desc:            "Infinite epsilon",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         math.Inf(1),
			alpha:           0.5,
		},
		{
			desc:            "NaN epsilon",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         math.NaN(),
			alpha:           0.5,
		},
		{
			desc:            "Non-zero delta",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			delta:           1,
			alpha:           0.5,
		},
		{
			desc:            "Zero alpha",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			alpha:           0,
		},
		{
			desc:            "Negative alpha",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			alpha:           -1,
		},
		{
			desc:            "1 alpha",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			alpha:           1,
		},
		{
			desc:            "Greater than 1 alpha",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			alpha:           2,
		},
		{
			desc:            "NaN alpha",
			noisedX:         0,
			l0Sensitivity:   1,
			lInfSensitivity: 1,
			epsilon:         0.1,
			alpha:           math.NaN(),
		},
	} {
		_, err := lap.ComputeConfidenceIntervalInt64(tc.noisedX, tc.l0Sensitivity, tc.lInfSensitivity,
			tc.epsilon, tc.delta, tc.alpha)

		if err == nil {
			t.Errorf("ComputeConfidenceIntervalInt64: when %s no error was returned, expected error", tc.desc)
		}
	}
}

func TestGeometricStatistics(t *testing.T) {
	const numberOfSamples = 125000
	for _, tc := range []struct {
		lambda float64
		mean   float64
		stdDev float64
	}{
		{
			lambda: 0.1,
			mean:   10.50833,
			stdDev: 9.99583,
		},
		{
			lambda: 0.0001,
			mean:   10000.50001,
			stdDev: 9999.99999,
		},
		{
			lambda: 0.0000001,
			mean:   10000000.5,
			stdDev: 9999999.99999,
		},
	} {
		geometricSamples := make(stat.IntSlice, numberOfSamples)
		for i := 0; i < numberOfSamples; i++ {
			geometricSamples[i] = geometric(tc.lambda)
		}
		sampleMean := stat.Mean(geometricSamples)
		// Assuming that the geometric samples are distributed according to the specified lambda, the
		// sampleMean is approximately Gaussian distributed with a mean of tc.mean and standard deviation
		// of tc.stdDev / sqrt(numberOfSamples).
		//
		// The meanErrorTolerance is set to the 99.9995% quantile of the anticipated distribution
		// of sampleMean. Thus, the test falsely rejects with a probability of 10⁻⁵.
		meanErrorTolerance := 4.41717 * tc.stdDev / math.Sqrt(float64(numberOfSamples))

		if !nearEqual(sampleMean, tc.mean, meanErrorTolerance) {
			t.Errorf("got mean = %f, want %f (parameters %+v)", sampleMean, tc.mean, tc)
		}
	}
}
