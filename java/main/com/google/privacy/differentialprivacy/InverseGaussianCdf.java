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
//         documentation and/or other materials provided with the distribution.
//       * Neither the name of the <organization> nor the
//         names of its contributors may be used to endorse or promote products
//         derived from this software without specific prior written permission.
//
//   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//   ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
//   DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Note: This file is a rewrite of the implementation of the ndtri function in
// the cephes library. It is not a copy; modification was permitted by the author
// under the above license. The original file can be found here:
// https://netlib.org/cephes/doubldoc.html#ndtri

package com.google.privacy.differentialprivacy;

import com.google.common.primitives.ImmutableDoubleArray;

/**
 * Computes the inverse cumulative distribution function (quantile function) of the standard normal
 * distribution using the Cephes {@code ndtri} rational approximation, preserving precision deep in
 * the distribution tails.
 */
final class InverseGaussianCdf {

  // Square-root of 2*Pi
  private static final double SQRT_TWO_PI = 2.5066282746310007;
  private static final double EXP_MINUS_2 = Math.exp(-2.0);
  private static final double EXP_MINUS_32 = Math.exp(-32.0);

  // Precomputed coefficients for the rational approximation of the inverse
  // of the Gaussian cdf around 0. The values are taken from
  // https://github.com/scipy/scipy/blob/main/scipy/special/cephes/ndtri.c
  private static final ImmutableDoubleArray AROUND_ZERO_APPROXIMATION_NUMERATOR_COEFFICIENTS =
      ImmutableDoubleArray.of(
          /*x^5*/ -59.96335010141079,
          /*x^4*/ 98.00107541859997,
          /*x^3*/ -56.67628574690703,
          /*x^2*/ 13.931260938727968,
          /*x^1*/ -1.2391658386738125,
          /*x^0*/ 0.00000000000000000000e0);
  private static final ImmutableDoubleArray AROUND_ZERO_APPROXIMATION_DENOMINATOR_COEFFICIENTS =
      ImmutableDoubleArray.of(
          /*x^8*/ 1.00000000000000000000e0,
          /*x^7*/ 1.9544885833814176,
          /*x^6*/ 4.676279128988815,
          /*x^5*/ 86.36024213908905,
          /*x^4*/ -225.46268785411937,
          /*x^3*/ 200.26021238006066,
          /*x^2*/ -82.03722561683334,
          /*x^1*/ 15.90562251262117,
          /*x^0*/ -1.1833162112133);

  // Precomputed coefficients for the rational approximation of the inverse
  // of the Gaussian cdf in the tail region. The values are taken from
  // https://github.com/scipy/scipy/blob/main/scipy/special/cephes/ndtri.c
  private static final ImmutableDoubleArray EXTREME_TAIL_APPROXIMATION_NUMERATOR_COEFFICIENTS =
      ImmutableDoubleArray.of(
          /*x^8*/ 3.2377489177694603,
          /*x^7*/ 6.915228890689842,
          /*x^6*/ 3.9388102529247444,
          /*x^5*/ 1.3330346081580755,
          /*x^4*/ 0.20148538954917908,
          /*x^3*/ 0.012371663481782003,
          /*x^2*/ 0.00030158155350823543,
          /*x^1*/ 0.0000026580697468673755,
          /*x^0*/ 6.239745391849833E-9);
  private static final ImmutableDoubleArray EXTREME_TAIL_APPROXIMATION_DENOMINATOR_COEFFICIENTS =
      ImmutableDoubleArray.of(
          /*x^8*/ 1.00000000000000000000e0,
          /*x^7*/ 6.02427039364742,
          /*x^6*/ 3.6798356385616087,
          /*x^5*/ 1.3770209948908132,
          /*x^4*/ 0.21623699359449663,
          /*x^3*/ 0.013420400608854318,
          /*x^2*/ 0.00032801446468212774,
          /*x^1*/ 0.0000028924786474538068,
          /*x^0*/ 6.790194080099813E-9);
  private static final ImmutableDoubleArray ORDINARY_TAIL_APPROXIMATION_NUMERATOR_COEFFICIENTS =
      ImmutableDoubleArray.of(
          /*x^8*/ 4.0554489230596245,
          /*x^7*/ 31.525109459989388,
          /*x^6*/ 57.16281922464213,
          /*x^5*/ 44.08050738932008,
          /*x^4*/ 14.684956192885803,
          /*x^3*/ 2.1866330685079025,
          /*x^2*/ -0.1402560791713545,
          /*x^1*/ -0.03504246268278482,
          /*x^0*/ -0.0008574567851546854);
  private static final ImmutableDoubleArray ORDINARY_TAIL_APPROXIMATION_DENOMINATOR_COEFFICIENTS =
      ImmutableDoubleArray.of(
          /*x^8*/ 1.00000000000000000000e0,
          /*x^7*/ 15.779988325646675,
          /*x^6*/ 45.39076351288792,
          /*x^5*/ 41.3172038254672,
          /*x^4*/ 15.04253856929075,
          /*x^3*/ 2.504649462083094,
          /*x^2*/ -0.14218292285478779,
          /*x^1*/ -0.03808064076915783,
          /*x^0*/ -0.0009332594808954574);

  /**
   * Calculates the inverse of the cumulative distribution function of the standard Gaussian
   * distribution. That is, for given {@code p} in {@code [0, 1]} it determines {@code x} such that
   * {@code 1/sqrt(2pi) * integral[-inf, x] exp(-t^2/2) dt = p}.
   *
   * <p>Returns {@code Double.NaN} if {@code p} is {@code NaN} or outside {@code [0, 1]}, {@code
   * Double.NEGATIVE_INFINITY} if {@code p == 0.0}, and {@code Double.POSITIVE_INFINITY} if {@code p
   * == 1.0}.
   */
  static double compute(double p) {
    if (Double.isNaN(p) || p < 0.0 || p > 1.0) {
      return Double.NaN;
    }
    if (p == 0.0) {
      return Double.NEGATIVE_INFINITY;
    }
    if (p == 1.0) {
      return Double.POSITIVE_INFINITY;
    }

    // Not tail case, i.e. when p in (exp(-2), 1-exp(-2)).
    if (p > EXP_MINUS_2 && p < (1.0 - EXP_MINUS_2)) {
      return evaluateApproximationAroundCenter(p);
    }

    // Reduce the calculation to the case p <= 0.5 by using
    // inverseCdf(p) = -inverseCdf(1-p). This yields higher accuracy.
    // Thus, for p > 0.5, the calculation needs to be performed for p = 1-p
    // and the sign has to be corrected in the end.
    boolean pWasCloseToOne = false;
    if (p >= (1.0 - EXP_MINUS_2)) {
      p = 1.0 - p;
      pWasCloseToOne = true;
    }

    // Starting here, p <= exp(-2) holds.
    // Further distinguish between the extreme tail of the distribution
    // (i.e. p <= exp(-32) = 1.2664165549e-14) and the "moderate" tail
    // exp(-32) < p <= exp(-2). Both cases are covered by the same approximation
    // rule, but with different rational approximation coefficients.
    double result;
    if (p > EXP_MINUS_32) {
      result =
          evaluateTailApproximation(
              p,
              ORDINARY_TAIL_APPROXIMATION_NUMERATOR_COEFFICIENTS,
              ORDINARY_TAIL_APPROXIMATION_DENOMINATOR_COEFFICIENTS);
    } else {
      result =
          evaluateTailApproximation(
              p,
              EXTREME_TAIL_APPROXIMATION_NUMERATOR_COEFFICIENTS,
              EXTREME_TAIL_APPROXIMATION_DENOMINATOR_COEFFICIENTS);
    }

    // If we used inverseCdf(1 - p) = -inverseCdf(p) above, we have to correct the sign.
    return pWasCloseToOne ? -result : result;
  }

  /**
   * Computes the inverse of a modified error function for values of {@code p} in {@code (exp(-2), 1
   * - exp(-2))}.
   */
  private static double evaluateApproximationAroundCenter(double p) {
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
    double numerator = evaluatePolynomial(p * p, AROUND_ZERO_APPROXIMATION_NUMERATOR_COEFFICIENTS);
    double denominator =
        evaluatePolynomial(p * p, AROUND_ZERO_APPROXIMATION_DENOMINATOR_COEFFICIENTS);
    return p * SQRT_TWO_PI * (1.0 + numerator / denominator);
  }

  /**
   * Evaluates the rational approximation for the inverse standard Gaussian CDF in the distribution
   * tails.
   */
  private static double evaluateTailApproximation(
      double p, ImmutableDoubleArray numeratorCoeffs, ImmutableDoubleArray denominatorCoeffs) {
    // The tail-approximation is given by the formula
    // log(z) / z - z + P(1/z) / Q(1/z), where z = sqrt(-2*log(p)) and
    // P, Q are polynomials with coefficients given as parameters.
    double z = Math.sqrt(-2.0 * Math.log(p));
    double numerator = evaluatePolynomial(1.0 / z, numeratorCoeffs);
    double denominator = evaluatePolynomial(1.0 / z, denominatorCoeffs);
    return Math.log(z) / z - z + (numerator / denominator) / z;
  }

  /**
   * Evaluates a polynomial with given coefficients in Horner's form: {@code result = (((c[0] * x +
   * c[1]) * x + ... ) * x + c[n - 1])}.
   */
  private static double evaluatePolynomial(double x, ImmutableDoubleArray coefficients) {
    double result = 0.0;
    for (int i = 0; i < coefficients.length(); i++) {
      result = result * x + coefficients.get(i);
    }
    return result;
  }

  private InverseGaussianCdf() {}
}
