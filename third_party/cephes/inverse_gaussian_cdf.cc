// Distributed under 3-clause BSD license with permission from the author,
//   see https://lists.debian.org/debian-legal/2004/12/msg00295.html
//
//   Cephes Math Library Release 2.8:  June, 2000
//
//   Copyright 1984, 1995, 2000 by Stephen L. Moshier
//
//   This software is derived from the Cephes Math Library and is
//   incorporated herein by permission of the author.
//
//   All rights reserved.
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions are met:
//
//       * Redistributions of source code must retain the above copyright
//         notice, this list of conditions and the following disclaimer.
//       * Redistributions in binary form must reproduce the above copyright
//         notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//       * Neither the name of the <organization> nor the
//         names of its contributors may be used to endorse or promote products
//         derived from this software without specific prior written permission.
//
//   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//   ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
//   DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
//   OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//   DAMAGE.
//
// Note: This file is a rewrite of the implementation of the ndtri function in
// the cephes library. It is not a copy; modification was permitted by the
// author under the above license. The original file can be found here:
// https://netlib.org/cephes/doubldoc.html#ndtri

#include "third_party/cephes/inverse_gaussian_cdf.h"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace differential_privacy::third_party::cephes {
namespace {
// Square-root of 2*Pi
constexpr double kSqrtTwoPi = 2.50662827463100050242e0;

// Precomputed coefficients for the rational approximation of the inverse
// of the Gaussian cdf around 0. The values are taken form
// https://github.com/scipy/scipy/blob/main/scipy/special/cephes/ndtri.c
constexpr std::array kAroundZeroApproximationNumeratorCoefficients = {
    /*x^5*/ -5.99633501014107895267e1,
    /*x^4*/ 9.80010754185999661536e1,
    /*x^3*/ -5.66762857469070293439e1,
    /*x^2*/ 1.39312609387279679503e1,
    /*x^1*/ -1.23916583867381258016e0,
    /*x^0*/ 0.00000000000000000000e0};
constexpr std::array kAroundZeroApproximationDenominatorCoefficients = {
    /*x^8*/ 1.00000000000000000000e0,
    /*x^7*/ 1.95448858338141759834e0,
    /*x^6*/ 4.67627912898881538453e0,
    /*x^5*/ 8.63602421390890590575e1,
    /*x^4*/ -2.25462687854119370527e2,
    /*x^3*/ 2.00260212380060660359e2,
    /*x^2*/ -8.20372256168333339912e1,
    /*x^1*/ 1.59056225126211695515e1,
    /*x^0*/ -1.18331621121330003142e0};
// Precomputed coefficients for the rational approximation of the inverse
// of the Gaussian cdf in the tail region. The values are taken form
// https://github.com/scipy/scipy/blob/main/scipy/special/cephes/ndtri.c
constexpr std::array kExtremeTailApproximationNumeratorCoefficients = {
    /*x^8*/ 3.23774891776946035970e0,
    /*x^7*/ 6.91522889068984211695e0,
    /*x^6*/ 3.93881025292474443415e0,
    /*x^5*/ 1.33303460815807542389e0,
    /*x^4*/ 2.01485389549179081538e-1,
    /*x^3*/ 1.23716634817820021358e-2,
    /*x^2*/ 3.01581553508235416007e-4,
    /*x^1*/ 2.65806974686737550832e-6,
    /*x^0*/ 6.23974539184983293730e-9};
constexpr std::array kExtremeTailApproximationDenominatorCoefficients = {
    /*x^8*/ 1.00000000000000000000e0,
    /*x^7*/ 6.02427039364742014255e0,
    /*x^6*/ 3.67983563856160859403e0,
    /*x^5*/ 1.37702099489081330271e0,
    /*x^4*/ 2.16236993594496635890e-1,
    /*x^3*/ 1.34204006088543189037e-2,
    /*x^2*/ 3.28014464682127739104e-4,
    /*x^1*/ 2.89247864745380683936e-6,
    /*x^0*/ 6.79019408009981274425e-9};
constexpr std::array kOrdinaryTailApproximationNumeratorCoefficients = {
    /*x^8*/ 4.05544892305962419923e0,
    /*x^7*/ 3.15251094599893866154e1,
    /*x^6*/ 5.71628192246421288162e1,
    /*x^5*/ 4.40805073893200834700e1,
    /*x^4*/ 1.46849561928858024014e1,
    /*x^3*/ 2.18663306850790267539e0,
    /*x^2*/ -1.40256079171354495875e-1,
    /*x^1*/ -3.50424626827848203418e-2,
    /*x^0*/ -8.57456785154685413611e-4};
constexpr std::array kOrdinaryTailApproximationDenominatorCoefficients = {
    /*x^8*/ 1.00000000000000000000e0,
    /*x^7*/ 1.57799883256466749731e1,
    /*x^6*/ 4.53907635128879210584e1,
    /*x^5*/ 4.13172038254672030440e1,
    /*x^4*/ 1.50425385692907503408e1,
    /*x^3*/ 2.50464946208309415979e0,
    /*x^2*/ -1.42182922854787788574e-1,
    /*x^1*/ -3.80806407691578277194e-2,
    /*x^0*/ -9.33259480895457427372e-4};

// Evaluates a polynomial with given coefficients. For n := coefficients.size(),
// this is sum[i = 0...n-1] coefficients[i] * x^i.
template <std::size_t N>
double EvaluatePolynomial(double x, const std::array<double, N>& coefficients) {
  double result = 0.0;
  for (double coefficient : coefficients) {
    result = result * x + coefficient;
  }
  return result;
}

// Computes the inverse of a modified error function for values of p in
// [exp(-2), 1 - exp(-2)]
double EvaluateApproximationAroundCenter(double p) {
  // The equation that needs to be solved for x is:
  // (2pi)^(-1/2) integral[-inf, x] exp(-t^2/2) dt = p
  // This is equivalent to solving
  // (2pi)^(-1/2) integral[0, x] exp(-t^2/2) dt = (p-0.5) for x.
  // Thus the problem reduces to inverting a scaled version of the error
  // function around 0. For this problem, an ordinary rational approximation is
  // accurate enough. The computational rule is:
  // inverseScaledErr(p) = p * sqrt(2pi) * (1 + P(p^2)/Q(p^2)),
  // where P(p) and Q(p) are polynomials of degree 5 and 8, respectively.
  p -= 0.5;
  double numerator =
      EvaluatePolynomial(p * p, kAroundZeroApproximationNumeratorCoefficients);
  double denominator = EvaluatePolynomial(
      p * p, kAroundZeroApproximationDenominatorCoefficients);
  return p * kSqrtTwoPi * (1 + numerator / denominator);
}

// Helper function that is used to evaluate the formula devised for the inverse
// af the Gaussian cdf, associated with the tail of the distribution.
template <std::size_t N, std::size_t M>
double EvaluateTailApproximation(
    double p, const std::array<double, N>& numerator_coeffs,
    const std::array<double, M>& denominator_coeffs) {
  // The tail-approximation is given by the formula
  // log(z) / z - z + P(1/z) / Q(1/z), where z = sqrt(-2log(p)) and
  // P, Q are polynomials with coefficients given as parameters.
  double z = std::sqrt(-2.0 * std::log(p));
  double numerator = EvaluatePolynomial(1 / z, numerator_coeffs);
  double denominator = EvaluatePolynomial(1 / z, denominator_coeffs);
  return std::log(z) / z - z + (numerator / denominator) / z;
}

}  // namespace

// Calculates the inverse of the cumulative distribution function of the
// standard Gaussian distribution. That is, for given p in [0, 1] it determines
// x such that 1/sqrt(2pi) * integral[-inf, x] exp(-x^2/2) dx = p.
double InverseCdfStandardGaussian(double p) {
  if (p < 0.0 || p > 1.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (p == 0.0) {
    return -std::numeric_limits<double>::infinity();
  }
  if (p == 1.0) {
    return std::numeric_limits<double>::infinity();
  }

  // Not tail case, i.e. when p in ]exp(-2), 1-exp(-2)[.
  if (p > std::exp(-2) && p < (1 - std::exp(-2))) {
    return EvaluateApproximationAroundCenter(p);
  }

  // Reduce the calculation to the case p <= 0.5 by using
  // inverseCdf(p) = - inverseCdf(1-p). This yields higher accuracy.
  // Thus, for p > 0.5, the calculation needs to be performed for p = 1-p
  // and the sign has to be corrected in the end.
  bool p_was_close_to_one = false;
  if (p > (1.0 - std::exp(-2))) {
    p = 1.0 - p;
    p_was_close_to_one = true;
  }

  // Starting here, p <= exp(-2) holds.
  // Further distinguish between the extreme tail of the distribution
  // (i.e. p <= exp(-32) = 1.2664165549e-14) and the "moderate" tail
  // exp(-32) < y <= exp(-2). Both cases are covered by the same approximation
  // rule, but with different rational approximation coefficients.
  double result = 0.0;
  if (p > std::exp(-32)) {
    result = EvaluateTailApproximation(
        p, kOrdinaryTailApproximationNumeratorCoefficients,
        kOrdinaryTailApproximationDenominatorCoefficients);
  } else {
    result = EvaluateTailApproximation(
        p, kExtremeTailApproximationNumeratorCoefficients,
        kExtremeTailApproximationDenominatorCoefficients);
  }

  // If we used inverseCdf(1 - p) = - inverseCdf(p) above, we have to correct
  // the sign.
  return p_was_close_to_one ? -result : result;
}
}  // namespace differential_privacy::third_party::cephes
