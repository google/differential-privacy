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
	"gonum.org/v1/gonum/stat/distuv"
)

var (
	// The square root of the maximum number n of Bernoulli trials from which a binomial
	// sample is drawn. Larger values result in more fine-grained noise, but increase the
	// chance of sampling inaccuracies due to overflows. The probability of such an event
	// will be roughly 2⁻⁴⁵ or less, if the square root is set to 2⁵⁷.
	binomialBound float64 = math.Exp2(57.0)
	// The absolute bound of the two-sided geometric samples k that are used for creating
	// a binomial sample is m + n / 2. For performance reasons, m is not composed of n
	// Bernoulli trials. Instead, m is obtained via a rejection sampling technique, which sets
	//   m = (k + l) * (sqrt(2 * n) + 1),
	// where l is a uniform random sample between 0 and 1. Bounding k is therefore necessary
	// to prevent m from overflowing.
	//
	// The probability of a single sample k being bounded is 2⁻⁴⁵.
	geometricBound int64 = (math.MaxInt64 / int64(math.Round(math.Sqrt2*binomialBound+1.0))) - 1
	// gaussianSigmaAccuracy approximates the accuracy up to which the smallest sigma that
	// satisfies the given DP parameters.
	gaussianSigmaAccuracy = 1e-3
)

type gaussian struct{}

// Gaussian returns a Noise instance that adds Gaussian noise to its input.
//
// The Gaussian noise is based on a binomial sampling mechanism that is robust against
// unintentional privacy leaks due to artifacts of floating-point arithmetic. See
// https://github.com/google/differential-privacy/blob/main/common_docs/Secure_Noise_Generation.pdf
// for more information.
func Gaussian() Noise {
	return gaussian{}
}

// AddNoiseFloat64 adds Gaussian noise to the specified float64, so that its
// output is (ε,δ)-differentially private.
func (gaussian) AddNoiseFloat64(x float64, l0Sensitivity int64, lInfSensitivity, epsilon, delta float64) (float64, error) {
	if err := checkArgsGaussian(l0Sensitivity, lInfSensitivity, epsilon, delta); err != nil {
		return 0, err
	}

	sigma := SigmaForGaussian(l0Sensitivity, lInfSensitivity, epsilon, delta)
	return addGaussianFloat64(x, sigma), nil
}

// AddNoiseInt64 adds Gaussian noise to the specified int64, so that the
// output is (ε,δ)-differentially private.
func (gaussian) AddNoiseInt64(x, l0Sensitivity, lInfSensitivity int64, epsilon, delta float64) (int64, error) {
	if err := checkArgsGaussian(l0Sensitivity, float64(lInfSensitivity), epsilon, delta); err != nil {
		return 0, err
	}

	sigma := SigmaForGaussian(l0Sensitivity, float64(lInfSensitivity), epsilon, delta)
	return addGaussianInt64(x, sigma), nil
}

// Threshold returns the smallest threshold k to use in a differentially private
// histogram with added Gaussian noise.
//
// See https://github.com/google/differential-privacy/blob/main/common_docs/Delta_For_Thresholding.pdf for details on the math underlying this.
func (gaussian) Threshold(l0Sensitivity int64, lInfSensitivity, epsilon, noiseDelta, thresholdDelta float64) (float64, error) {
	if err := checkArgsGaussian(l0Sensitivity, lInfSensitivity, epsilon, noiseDelta); err != nil {
		return 0, err
	}
	if err := checks.CheckThresholdDelta(thresholdDelta, noiseDelta); err != nil {
		return 0, err
	}

	sigma := SigmaForGaussian(l0Sensitivity, lInfSensitivity, epsilon, noiseDelta)
	noiseDist := distuv.Normal{Mu: 0, Sigma: sigma}
	return lInfSensitivity + noiseDist.Quantile(math.Pow(1-thresholdDelta, 1.0/float64(l0Sensitivity))), nil
}

// DeltaForThreshold is the inverse operation of Threshold. Specifically, given
// the parameters and a threshold, it returns the delta induced by thresholding.
//
// See https://github.com/google/differential-privacy/blob/main/common_docs/Delta_For_Thresholding.pdf for details on the math underlying this.
func (gaussian) DeltaForThreshold(l0Sensitivity int64, lInfSensitivity, epsilon, delta, threshold float64) (float64, error) {
	if err := checkArgsGaussian(l0Sensitivity, lInfSensitivity, epsilon, delta); err != nil {
		return 0, err
	}
	sigma := SigmaForGaussian(l0Sensitivity, lInfSensitivity, epsilon, delta)
	noiseDist := distuv.Normal{Mu: 0, Sigma: sigma}
	return 1 - math.Pow(noiseDist.CDF(threshold-lInfSensitivity), float64(l0Sensitivity)), nil
}

// ComputeConfidenceIntervalInt64 computes a confidence interval that contains the raw integer value x from which int64 noisedX
// is computed with a probability greater or equal to 1 - alpha based on the specified gaussian noise parameters.
//
// See https://github.com/google/differential-privacy/tree/main/common_docs/confidence_intervals.md.
func (gaussian) ComputeConfidenceIntervalInt64(noisedX, l0Sensitivity, lInfSensitivity int64, epsilon, delta, alpha float64) (ConfidenceInterval, error) {
	err := checkArgsConfidenceIntervalGaussian(l0Sensitivity, float64(lInfSensitivity), epsilon, delta, alpha)
	if err != nil {
		return ConfidenceInterval{}, err
	}
	sigma := SigmaForGaussian(l0Sensitivity, float64(lInfSensitivity), epsilon, delta)
	// Computing the confidence interval around zero rather than nosiedX helps represent the
	// interval bounds more accurately. The reason is that the resolution of float64 values is most
	// fine grained around zero.
	confIntAroundZero := computeConfidenceIntervalGaussian(0, sigma, alpha).roundToInt64()
	// Adding noisedX after converting the interval bounds to int64 ensures that no precision is lost
	// due to the coarse resolution of float64 values for large instances of noisedX.
	lowerBound := nextSmallerFloat64(int64(confIntAroundZero.LowerBound) + noisedX)
	upperBound := nextLargerFloat64(int64(confIntAroundZero.UpperBound) + noisedX)
	return ConfidenceInterval{LowerBound: lowerBound, UpperBound: upperBound}, nil
}

// ComputeConfidenceIntervalFloat64 computes a confidence interval that contains the raw value x from which float64
// noisedX is computed with a probability equal to 1 - alpha based on the specified gaussian noise parameters.
//
// See https://github.com/google/differential-privacy/tree/main/common_docs/confidence_intervals.md.
func (gaussian) ComputeConfidenceIntervalFloat64(noisedX float64, l0Sensitivity int64, lInfSensitivity, epsilon, delta, alpha float64) (ConfidenceInterval, error) {
	err := checkArgsConfidenceIntervalGaussian(l0Sensitivity, lInfSensitivity, epsilon, delta, alpha)
	if err != nil {
		return ConfidenceInterval{}, err
	}
	sigma := SigmaForGaussian(l0Sensitivity, lInfSensitivity, epsilon, delta)
	return computeConfidenceIntervalGaussian(noisedX, sigma, alpha), nil
}

func checkArgsGaussian(l0Sensitivity int64, lInfSensitivity, epsilon, delta float64) error {
	if err := checks.CheckL0Sensitivity(l0Sensitivity); err != nil {
		return err
	}
	if err := checks.CheckLInfSensitivity(lInfSensitivity); err != nil {
		return err
	}
	if err := checks.CheckEpsilonVeryStrict(epsilon); err != nil {
		return err
	}
	return checks.CheckDeltaStrict(delta)
}

func (gaussian) String() string {
	return "Gaussian Noise"
}

// checkArgsConfidenceIntervalGaussian checks the parameters for gaussian confidence interval, as well as the provided confidence level.
func checkArgsConfidenceIntervalGaussian(l0Sensitivity int64, lInfSensitivity, epsilon, delta, alpha float64) error {
	if err := checks.CheckAlpha(alpha); err != nil {
		return err
	}
	return checkArgsGaussian(l0Sensitivity, lInfSensitivity, epsilon, delta)
}

// addGaussianFloat64 adds Gaussian noise of scale σ to the specified float64.
func addGaussianFloat64(x, sigma float64) float64 {
	granularity := ceilPowerOfTwo(2.0 * sigma / binomialBound)

	// sqrtN is chosen in a way that places it in the interval between binomialBound
	// and binomialBound / 2. This ensures that the respective binomial distribution
	// consists of enough Bernoulli samples to closely approximate a Gaussian distribution.
	sqrtN := 2.0 * sigma / granularity
	sample := symmetricBinomial(sqrtN)
	return roundToMultipleOfPowerOfTwo(x, granularity) + float64(sample)*granularity
}

// addGaussianInt64 adds Gaussian noise of scale σ to the specified int64.
func addGaussianInt64(x int64, sigma float64) int64 {
	granularity := ceilPowerOfTwo(2.0 * sigma / binomialBound)

	// sqrtN is chosen in a way that places it in the interval between binomialBound
	// and binomialBound / 2. This ensures that the respective binomial distribution
	// consists of enough Bernoulli samples to closely approximate a Gaussian distribution.
	sqrtN := 2.0 * sigma / granularity
	sample := symmetricBinomial(sqrtN)
	if granularity < 1 {
		return x + int64(math.Round(float64(sample)*granularity))
	}
	return roundToMultiple(x, int64(granularity)) + sample*int64(granularity)
}

// computeConfidenceIntervalGaussian computes a confidence interval that contains the raw value x from which
// float64 noisedX is computed with a probability equal to 1 - alpha with the given sigma.
func computeConfidenceIntervalGaussian(noisedX, sigma, alpha float64) ConfidenceInterval {
	z := inverseCDFGaussian(sigma, alpha/2)
	// Because of the symmetry of the Gaussian distribution,
	// -z corresponds to the (1 - alpha/2)-quantile of the distribution,
	// meaning that the interval [z, -z] contains 1-alpha of the probability mass.
	// Deriving the (1 - alpha/2)-quantile from the (alpha/2)-quantile and not vice versa is a
	// deliberate choice. The reason is that alpha tends to be very small.
	// Consequently, alpha/2 is more accurately representable as a float64 than 1 - alpha/2,
	// facilitating numerical computations.
	return ConfidenceInterval{LowerBound: noisedX + z, UpperBound: noisedX - z}
}

// inverseCDFGaussian computes the quantile z satisfying Pr[Y <= z] = p for a random variable Y that is Gaussian
// distributed with the specified sigma and mean 0.
func inverseCDFGaussian(sigma, p float64) float64 {
	return -sigma * math.Sqrt(2) * math.Erfcinv(2*p)
}

// symmetricBinomial returns a random sample m where the term m + n / 2 is drawn from
// a binomial distribution of n Bernoulli trials that have a success probability of
// 0.5 each. The sampling technique is based on Bringmann et al.'s rejection sampling
// approach proposed in "Internal DLA: Efficient Simulation of a Physical Growth Model"
// (https://people.mpi-inf.mpg.de/~kbringma/paper/2014ICALP.pdf).
func symmetricBinomial(sqrtN float64) int64 {
	stepSize := int64(math.Round(math.Sqrt2*sqrtN + 1.0))
	var result int64
	i := 0
	for true {
		// 1 is subtracted from the geometric sample to count the number of Bernoulli fails
		// rather than the number of trials until the first success.
		boundedGeometricSample := int64(math.Min(rand.Geometric()-1.0, float64(geometricBound)))
		twoSidedGeometricSample := boundedGeometricSample
		if rand.Boolean() {
			twoSidedGeometricSample = -twoSidedGeometricSample - 1
		}

		result = stepSize*twoSidedGeometricSample + rand.I63n(stepSize)
		resultProbability := binomialProbability(sqrtN, result)
		rejectProbability := rand.Uniform()
		if resultProbability > 0.0 &&
			rejectProbability < resultProbability*float64(stepSize)*math.Pow(2.0, float64(boundedGeometricSample))/4.0 {
			break
		}
		i++
	}
	return result
}

// Approximates the probability of a random sample m + n / 2 drawn from a binomial
// distribution of n Bernoulli trials that have a success probability of 1 / 2 each.
// The approximation is based on Lemma 7 of
// https://github.com/google/differential-privacy/blob/main/common_docs/Secure_Noise_Generation.pdf
func binomialProbability(sqrtN float64, m int64) float64 {
	if math.Abs(float64(m)) > sqrtN*math.Sqrt(math.Log(sqrtN)/2.0) {
		return 0.0
	}
	return (math.Sqrt(2.0/math.Pi) / sqrtN) *
		math.Exp((-2.0*float64(m)*float64(m))/(sqrtN*sqrtN)) *
		(1 - 0.4*math.Pow(2.0, 1.5)*math.Pow(math.Log(sqrtN), 1.5)/sqrtN)
}

// deltaForGaussian computes the smallest δ such that the Gaussian mechanism
// with fixed standard deviation σ is (ε,δ)-differentially private. The
// calculation is based on Theorem 8 of Balle and Wang's "Improving the Gaussian
// Mechanism for Differential Privacy: Analytical Calibration and Optimal
// Denoising" (https://arxiv.org/abs/1805.06530v2).
func deltaForGaussian(sigma float64, l0Sensitivity int64, lInfSensitivity, epsilon float64) float64 {
	l2Sensitivity := lInfSensitivity * math.Sqrt(float64(l0Sensitivity))
	// Defining
	//   Φ – Standard Gaussian distribution (mean: 0, variance: 1) CDF function
	//   s – L2 sensitivity
	//   δ(σ,s,ε) – The level of (ε,δ)-approximate differential privacy achieved
	//              by the Gaussian mechanism applied with standard deviation σ
	//              to data with L2 sensitivity s with fixed ε.
	// The tight choice of δ (see https://arxiv.org/abs/1805.06530v2, Theorem 8) is:
	//   δ(σ,s,ε) := Φ(s/(2σ) - εσ/s) - exp(ε)Φ(-s/(2σ) - εσ/s)
	// To simplify the calculation of this formula and to simplify reasoning about
	// overflow and underflow, we pull out terms a := s/(2σ), b := εσ/s, c := exp(ε)
	// so that δ(σ,s,ε) = Φ(a - b) - cΦ(-a - b)
	a := l2Sensitivity / (2 * sigma)
	b := epsilon * sigma / l2Sensitivity
	c := math.Exp(epsilon)

	if math.IsInf(c, +1) {
		// δ(σ,s,ε) –> 0 as ε –> ∞, so return 0.
		return 0
	}
	if math.IsInf(b, +1) {
		// δ(σ,s,ε) –> 0 as the L2 sensitivity –> 0, so return 0.
		return 0
	}

	return distuv.UnitNormal.CDF(a-b) - c*distuv.UnitNormal.CDF(-a-b)
}

// SigmaForGaussian calculates the standard deviation σ of Gaussian noise
// needed to achieve (ε,δ)-approximate differential privacy.
//
// SigmaForGaussian uses binary search. The result will deviate from the exact value
// σ_tight by at most gaussianSigmaAccuracy*σ_tight.
//
// Runtime: O(log(max(σ_tight/l2Sensitivity, l2Sensitivity/σ_tight)) + log(gaussianSigmaAccuracy)),
// where l2Sensitivity := lInfSensitivity * math.Sqrt(l0Sensitivity)
//
// The calculation is based on Balle and Wang's ["Improving the Gaussian Mechanism
// for Differential Privacy: Analytical Calibration and Optimal Denoising"].
// The paper states that the lower bound on sigma from the original
// analysis of the Gaussian mechanism (sigma ≥ sqrt(2 * l2_sensitivity^2 * log(1.25/𝛿) / 𝜖^2))
// is far from tight and binary search can give us a better lower bound.
//
// TODO: Memorize the results of this function to avoid recomputing it
//
// ["Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising"]: https://arxiv.org/abs/1805.06530v2
func SigmaForGaussian(l0Sensitivity int64, lInfSensitivity, epsilon, delta float64) float64 {
	if delta >= 1 {
		return 0
	}

	// We use l2Sensitivity as a starting guess for the upper bound since the
	// required noise grows linearly with sensitivity.
	l2Sensitivity := lInfSensitivity * math.Sqrt(float64(l0Sensitivity))
	upperBound := l2Sensitivity
	var lowerBound float64

	// Increase upperBound until it is actually an upper bound of σ_tight.
	//
	// DeltaForGaussian(sigma, l2Sensitivity, epsilon) is a decreasing function with
	// respect to sigma. This loop terminates in
	//   O(log(σ_tight/l2Sensitivity)) if σ_tight > l2Sensitivity
	//   O(1)                          otherwise.
	// In the case where σ_tight > l2Sensitivity, when the loop exits, the
	// following things are true:
	//   (1) upperBound - lowerBound <= σ_tight
	//   (2) lowerBound >= 0.5*σ_tight.
	for deltaForGaussian(upperBound, l0Sensitivity, lInfSensitivity, epsilon) > delta {
		lowerBound = upperBound
		upperBound = upperBound * 2
	}

	// Loop runtime:
	//   O(log(1/σ_tight) + log(1/gaussianSigmaAccuracy)) if σ_tight < l2Sensitivity
	//   O(log(1/gaussianSigmaAccuracy))                  otherwise.
	//
	// Proof. First, suppose σ_tight > l2Sensitivity. The prior for-loop guarantees that
	//   (1) upperBound - lowerBound <= σ_tight
	//   (2) lowerBound >= 0.5*σ_tight
	// at the start of this loop.  Using (1), binary search takes
	// O(log(1/gaussianSigmaAccuracy)) iterations to bound the solution within an interval
	// of width 0.5*σ_tight*gaussianSigmaAccuracy. Since (2) holds over all iterations of
	// this loop, that is sufficient iterations to meet the loop's exit criterion.
	//
	// Now suppose σ_tight <= l2Sensitivity. It takes
	// O(log(l2Sensitivity/σ_tight)) iterations to calculate a middle that is less
	// than σ_tight. At that iteration, lowerBound is updated to be at least
	// 0.5*σ_tight. After this first update to lowerBound, we use the argument of
	// the preceding paragraph (noting that (1) and (2) now both hold) to see that
	// it takes an additional O(log(l2Sensitivity/gaussianSigmaAccuracy)) iterations to
	// find a sufficiently accurate estimate of σ_tight to exit the loop.
	for upperBound-lowerBound > gaussianSigmaAccuracy*lowerBound {
		middle := lowerBound*0.5 + upperBound*0.5
		if deltaForGaussian(middle, l0Sensitivity, lInfSensitivity, epsilon) > delta {
			lowerBound = middle
		} else {
			upperBound = middle
		}
	}

	return upperBound
}
