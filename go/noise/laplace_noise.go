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

	"github.com/google/differential-privacy/go/v3/checks"
	"github.com/google/differential-privacy/go/v3/rand"
)

var (
	// granularityParam determines the resolution of the numerical noise that is
	// being generated relative to the L_inf sensitivity and privacy parameter epsilon.
	// More precisely, the granularity parameter corresponds to the value 2ᵏ described in
	// https://github.com/google/differential-privacy/blob/main/common_docs/Secure_Noise_Generation.pdf.
	// Larger values result in more fine grained noise, but increase the chance of
	// sampling inaccuracies due to overflows. The probability of an overflow is less
	// than 2⁻¹⁰⁰⁰, if the granularity parameter is set to a value of 2⁴⁰ or less and
	// the epsilon passed to addNoise is at least 2⁻⁵⁰.
	//
	// This parameter should be a power of 2.
	granularityParam = math.Exp2(40)
	// deltaLowPrecisionThreshold ensures that addition and subtraction operations
	// involving delta and numbers resulting in values within [0, 1] maintain at
	// least 6 significant digits of precision (in base 10) from delta.
	deltaLowPrecisionThreshold = (1 - math.Nextafter(1.0, math.Inf(-1))) * 1e6
)

type laplace struct{}

// Laplace returns a Noise instance that adds Laplace noise to its input.
// Its AddNoise* functions will fail if called with a non-zero delta.
//
// The Laplace noise is based on a geometric sampling mechanism that is robust against
// unintentional privacy leaks due to artifacts of floating point arithmetic. See
// https://github.com/google/differential-privacy/blob/main/common_docs/Secure_Noise_Generation.pdf
// for more information.
func Laplace() Noise {
	return laplace{}
}

// AddNoiseFloat64 adds Laplace noise to the specified float64 x so that the
// output is ε-differentially private given the L_0 and L_∞ sensitivities of the
// database.
func (laplace) AddNoiseFloat64(x float64, l0Sensitivity int64, lInfSensitivity, epsilon, delta float64) (float64, error) {
	if err := checkArgsLaplace(l0Sensitivity, lInfSensitivity, epsilon, delta); err != nil {
		return 0, err
	}
	return addLaplaceFloat64(x, epsilon, lInfSensitivity*float64(l0Sensitivity) /* l1Sensitivity */), nil
}

// AddNoiseInt64 adds Laplace noise to the specified int64 x so that the
// output is ε-differentially private given the L_0 and L_∞ sensitivities of the
// database.
func (laplace) AddNoiseInt64(x, l0Sensitivity, lInfSensitivity int64, epsilon, delta float64) (int64, error) {
	if err := checkArgsLaplace(l0Sensitivity, float64(lInfSensitivity), epsilon, delta); err != nil {
		return 0, err
	}
	return addLaplaceInt64(x, epsilon, lInfSensitivity*l0Sensitivity /* l1Sensitivity */), nil
}

// Threshold returns the smallest threshold k to use in a differentially private
// histogram with added Laplace noise. Like other functions for Laplace noise,
// it fails if noiseDelta is non-zero.
func (laplace) Threshold(l0Sensitivity int64, lInfSensitivity, epsilon, noiseDelta, thresholdDelta float64) (float64, error) {
	if err := checkArgsLaplace(l0Sensitivity, lInfSensitivity, epsilon, noiseDelta); err != nil {
		return 0, err
	}
	if err := checks.CheckThresholdDelta(thresholdDelta, noiseDelta); err != nil {
		return 0, err
	}
	// λ is the scale of the Laplace noise that needs to be added to each sum
	// to get pure ε-differential privacy if all keys are the same.
	lambda := laplaceLambda(l0Sensitivity, lInfSensitivity, epsilon)

	// For the special case where a key is present in one dataset and not the
	// other, the worst case happens when the value of this key is exactly the
	// lInfSensitivity. Let F denote the CDF of the Laplace distribution with mean
	// lInfSensitivity and scale λ. The key will be kept with probability 1-F(k),
	// breaking ε-differential privacy. With probability F(k), it will be
	// thresholded, maintaining differential privacy in that partition.
	//
	// To achieve (ε, δ)-differential privacy, we must gaurantee that with
	// probability at least 1-δ, all coordinates on which two adjacent datasets
	// differ are dropped. Adjacent datasets can differ only by l0Sensitivity
	// partitions. Using independence, these coordinates must be dropped with
	// probability (1-δ)^{1/l0Sensitivity}. Conceptually, δ_p = 1 -
	// (1-δ)^{1/l0Sensitivity} is the per-partition probability of leaking the
	// individual parition. It suffices to choose k such that F(k) ≥ 1-δ_p
	//
	// The formula for F(k) is:
	//          {  1 - 1/2 * exp(-(k-lInfSensitivity)/λ)  if k ≥ lInfSensitivity
	//   F(k) = {  1/2 * exp((k-lInfSensitivity)/λ)       otherwise
	//
	// Solving for k in F(k) ≥ 1-δ_p yields:
	//
	//       { lInfSensitivity - λlog(2δ_p)        if δ_p ≤ 0.5
	//   k ≥ { lInfSensitivity + λlog[2(1-δ_p)]    otherwise
	//
	// To see the condition `if δ_p ≤ 0.5`, note that k ≥ lInfSensitivity
	// corresponds to the case where F(k) ≥ 0.5. We are solving for k in the
	// inequality F(k) ≥ 1-δ_p, which puts us in the F(k) ≥ 0.5 case when
	// 1-δ/l0Sensitivity ≥ 0.5 (and hence δ_p ≤ 0.5).
	partitionDelta := 1 - math.Pow(1-thresholdDelta, 1/float64(l0Sensitivity))
	if thresholdDelta < deltaLowPrecisionThreshold {
		// The above calculation of partitionDelta can lose precision in the 1-delta
		// computation if delta is too small. So, we fall back on the lower bound of
		// partitionDelta that does not make the independence assumption. This lower
		// bound will be more accurate for sufficiently small delta.
		partitionDelta = thresholdDelta / float64(l0Sensitivity)
	}
	if partitionDelta <= 0.5 {
		return lInfSensitivity - lambda*math.Log(2*partitionDelta), nil
	}
	return lInfSensitivity + lambda*math.Log(2*(1-partitionDelta)), nil
}

// DeltaForThreshold is the inverse operation of Threshold: given the parameters
// passed to AddNoise and a threshold, it returns the delta induced by
// thresholding. Just like other functions for Laplace noise, it fails if
// delta is non-zero.
func (laplace) DeltaForThreshold(l0Sensitivity int64, lInfSensitivity, epsilon, delta, threshold float64) (float64, error) {
	if err := checkArgsLaplace(l0Sensitivity, lInfSensitivity, epsilon, delta); err != nil {
		return 0, err
	}
	lambda := laplaceLambda(l0Sensitivity, lInfSensitivity, epsilon)
	var partitionDelta float64
	if threshold >= lInfSensitivity {
		partitionDelta = 0.5 * math.Exp(-(threshold-lInfSensitivity)/lambda)
	} else {
		partitionDelta = (1 - 0.5*math.Exp((threshold-lInfSensitivity)/lambda))
	}
	if partitionDelta < deltaLowPrecisionThreshold {
		// This is an upper bound on the induced delta that does not use
		// independence between coordinates. It has the advantage over the
		// calculation below that uses independence between coordinates that it does
		// not floating point lose precision as easily as the step 1-partitionDelta.
		return math.Min(partitionDelta*float64(l0Sensitivity), 1), nil
	}
	return 1 - math.Pow(1-partitionDelta, float64(l0Sensitivity)), nil
}

// ComputeConfidenceIntervalInt64 computes a confidence interval that contains the raw integer value x from which int64 noisedX
// is computed with a probability greater or equal to 1 - alpha based on the specified laplace noise parameters.
//
// See https://github.com/google/differential-privacy/tree/main/common_docs/confidence_intervals.md.
func (laplace) ComputeConfidenceIntervalInt64(noisedX, l0Sensitivity, lInfSensitivity int64, epsilon, delta, alpha float64) (ConfidenceInterval, error) {
	err := checkArgsConfidenceIntervalLaplace(l0Sensitivity, float64(lInfSensitivity), epsilon, delta, alpha)
	if err != nil {
		return ConfidenceInterval{}, err
	}
	lambda := laplaceLambda(l0Sensitivity, float64(lInfSensitivity), epsilon)
	// Computing the confidence interval around zero rather than nosiedX helps represent the
	// interval bounds more accurately. The reason is that the resolution of float64 values is most
	// fine grained around zero.
	confIntAroundZero := computeConfidenceIntervalLaplace(0, lambda, alpha).roundToInt64()
	// Adding noisedX after converting the interval bounds to int64 ensures that no precision is lost
	// due to the coarse resolution of float64 values for large instances of noisedX.
	lowerBound := nextSmallerFloat64(int64(confIntAroundZero.LowerBound) + noisedX)
	upperBound := nextLargerFloat64(int64(confIntAroundZero.UpperBound) + noisedX)
	return ConfidenceInterval{LowerBound: lowerBound, UpperBound: upperBound}, nil
}

// ComputeConfidenceIntervalFloat64 computes a confidence interval that contains the raw value x from which float64
// noisedX is computed with a probability equal to 1 - alpha based on the specified laplace noise parameters.
//
// See https://github.com/google/differential-privacy/tree/main/common_docs/confidence_intervals.md.
func (laplace) ComputeConfidenceIntervalFloat64(noisedX float64, l0Sensitivity int64, lInfSensitivity, epsilon, delta, alpha float64) (ConfidenceInterval, error) {
	err := checkArgsConfidenceIntervalLaplace(l0Sensitivity, lInfSensitivity, epsilon, delta, alpha)
	if err != nil {
		return ConfidenceInterval{}, err
	}
	lambda := laplaceLambda(l0Sensitivity, lInfSensitivity, epsilon)
	return computeConfidenceIntervalLaplace(noisedX, lambda, alpha), nil
}

func (laplace) String() string {
	return "Laplace Noise"
}

func checkArgsLaplace(l0Sensitivity int64, lInfSensitivity, epsilon, delta float64) error {
	if err := checks.CheckL0Sensitivity(l0Sensitivity); err != nil {
		return err
	}
	if err := checks.CheckLInfSensitivity(lInfSensitivity); err != nil {
		return err
	}
	if err := checks.CheckEpsilonVeryStrict(epsilon); err != nil {
		return err
	}
	return checks.CheckNoDelta(delta)
}

func checkArgsConfidenceIntervalLaplace(l0Sensitivity int64, lInfSensitivity, epsilon, delta, alpha float64) error {
	if err := checks.CheckAlpha(alpha); err != nil {
		return err
	}
	return checkArgsLaplace(l0Sensitivity, lInfSensitivity, epsilon, delta)
}

// addLaplaceFloat64 adds Laplace noise scaled to the given epsilon and l1Sensitivity to the
// specified float64
func addLaplaceFloat64(x, epsilon, l1Sensitivity float64) float64 {
	granularity := ceilPowerOfTwo((l1Sensitivity / epsilon) / granularityParam)
	sample := twoSidedGeometric(granularity * epsilon / (l1Sensitivity + granularity))
	return roundToMultipleOfPowerOfTwo(x, granularity) + float64(sample)*granularity
}

// addLaplaceInt64 adds Laplace noise scaled to the given epsilon and l1Sensitivity to the
// specified int64
func addLaplaceInt64(x int64, epsilon float64, l1Sensitivity int64) int64 {
	granularity := ceilPowerOfTwo((float64(l1Sensitivity) / epsilon) / granularityParam)
	sample := twoSidedGeometric(granularity * epsilon / (float64(l1Sensitivity) + granularity))
	if granularity < 1 {
		return x + int64(math.Round(float64(sample)*granularity))
	}
	return roundToMultiple(x, int64(granularity)) + sample*int64(granularity)
}

// laplaceLambda computes the scale parameter λ for the Laplace noise
// distribution required by the Laplace mechanism for achieving ε-differential
// privacy on databases with the given L_0 and L_∞ sensitivities.
func laplaceLambda(l0Sensitivity int64, lInfSensitivity, epsilon float64) float64 {
	l1Sensitivity := lInfSensitivity * float64(l0Sensitivity)
	return l1Sensitivity / epsilon
}

// computeConfidenceIntervalLaplace computes a confidence interval that contains the raw value x from which
// float64 noisedX is computed with a probability equal to 1 - alpha with the given lambda.
func computeConfidenceIntervalLaplace(noisedX float64, lambda, alpha float64) ConfidenceInterval {
	z := inverseCDFLaplace(lambda, alpha/2)
	// Because of the symmetry of the Laplace distribution,
	// -z corresponds to the (1 - alpha/2)-quantile of the distribution,
	// meaning that the interval [z, -z] contains 1-alpha of the probability mass.
	// Deriving the (1 - alpha/2)-quantile from the (alpha/2)-quantile and not vice versa is a
	// deliberate choice. The reason is that alpha tends to be very small.
	// Consequently, alpha/2 is more accurately representable as a float64 than 1 - alpha/2,
	// facilitating numerical computations.
	return ConfidenceInterval{LowerBound: noisedX + z, UpperBound: noisedX - z}
}

// inverseCDFLaplace computes the quantile z satisfying Pr[Y <= z] = p for a random variable Y
// that is Laplace distributed with the specified lambda where mean is zero.
func inverseCDFLaplace(lambda, p float64) float64 {
	if p < 0.5 {
		return lambda * math.Log(2*p)
	}
	return -lambda * math.Log(2*(1-p))
}

// geometric draws a sample drawn from a geometric distribution with parameter
//   p = 1 - e^-λ.
// More precisely, it returns the number of Bernoulli trials until the first success
// where the success probability is p = 1 - e^-λ. The returned sample is truncated
// to the max int64 value.
//
// Note that to ensure that a truncation happens with probability less than 10⁻⁶,
// λ must be greater than 2⁻⁵⁹.
func geometric(lambda float64) int64 {
	// Return truncated sample in the case that the sample exceeds the max int64.
	if rand.Uniform() > -1.0*math.Expm1(-1.0*lambda*math.MaxInt64) {
		return math.MaxInt64
	}

	// Perform a binary search for the sample in the interval from 1 to max int64.
	// Each iteration splits the interval in two and randomly keeps either the
	// left or the right subinterval depending on the respective probability of
	// the sample being contained in them. The search ends once the interval only
	// contains a single sample.
	var left int64 = 0              // exclusive bound
	var right int64 = math.MaxInt64 // inclusive bound

	for left+1 < right {
		// Compute a midpoint that divides the probability mass of the current interval
		// approximately evenly between the left and right subinterval. The resulting
		// midpoint will be less or equal to the arithmetic mean of the interval. This
		// reduces the expected number of iterations of the binary search compared to a
		// search that uses the arithmetic mean as a midpoint. The speed up is more
		// pronounced the higher the success probability p is.
		mid := left - int64(math.Floor((math.Log(0.5)+math.Log1p(math.Exp(lambda*float64(left-right))))/lambda))
		// Ensure that mid is contained in the search interval. This is a safeguard to
		// account for potential mathematical inaccuracies due to finite precision arithmetic.
		if mid <= left {
			mid = left + 1
		} else if mid >= right {
			mid = right - 1
		}

		// Probability that the sample is at most mid, i.e.,
		//   q = Pr[X ≤ mid | left < X ≤ right]
		// where X denotes the sample. The value of q should be approximately one half.
		q := math.Expm1(lambda*float64(left-mid)) / math.Expm1(lambda*float64(left-right))
		if rand.Uniform() <= q {
			right = mid
		} else {
			left = mid
		}
	}
	return right
}

// twoSidedGeometric draws a sample from a geometric distribution that is
// mirrored at 0. The non-negative part of the distribution's PDF matches
// the PDF of a geometric distribution of parameter p = 1 - e^-λ that is
// shifted to the left by 1 and scaled accordingly.
func twoSidedGeometric(lambda float64) int64 {
	var sample int64 = 0
	var sign int64 = -1
	// Keep a sample of 0 only if the sign is positive. Otherwise, the
	// probability of 0 would be twice as high as it should be.
	for sample == 0 && sign == -1 {
		sample = geometric(lambda) - 1
		sign = int64(rand.Sign())
	}
	return sample * sign
}
