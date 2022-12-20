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
// Note: The file inverse_gaussian_cdf.cc is a rewrite of the implementation
// of the ndtri function in the cephes library. It is not a copy; modification
// was permitted by the author under the above license. The original file can be
// found here: https://netlib.org/cephes/doubldoc.html#ndtri

#ifndef DIFFERENTIAL_PRIVACY_THIRD_PARTY_CEPHES_INVERSE_GAUSSIAN_CDF_H_
#define DIFFERENTIAL_PRIVACY_THIRD_PARTY_CEPHES_INVERSE_GAUSSIAN_CDF_H_

namespace differential_privacy::third_party::cephes {
// Calculates the inverse of the cumulative distribution function of the
// standard Gaussian distribution. That is, for given p in [0, 1] it determines
// x such that 1/sqrt(2pi) * integral[-inf, x] exp(-x^2/2) dx = p.
// The behaviour of this function is undefined if p < 0 or p > 1.
double InverseCdfStandardGaussian(double p);

}  // namespace differential_privacy::third_party::cephes

#endif  // DIFFERENTIAL_PRIVACY_THIRD_PARTY_CEPHES_INVERSE_GAUSSIAN_CDF_H_
