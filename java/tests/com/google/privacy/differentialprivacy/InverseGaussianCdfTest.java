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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class InverseGaussianCdfTest {

  @Test
  public void compute_inputZero_returnsNegativeInfinity() {
    assertThat(InverseGaussianCdf.compute(0.0)).isNegativeInfinity();
  }

  @Test
  public void compute_inputOne_returnsPositiveInfinity() {
    assertThat(InverseGaussianCdf.compute(1.0)).isPositiveInfinity();
  }

  @Test
  public void compute_invalidInputs_returnsNaN(
      @TestParameter({"-0.1", "1.1", "-Infinity", "Infinity", "NaN"}) double p) {
    assertThat(InverseGaussianCdf.compute(p)).isNaN();
  }

  @Test
  public void compute_justOutsideUnitInterval_returnsNaN() {
    assertThat(InverseGaussianCdf.compute(Math.nextDown(0.0))).isNaN();
    assertThat(InverseGaussianCdf.compute(Math.nextUp(1.0))).isNaN();
  }

  @Test
  public void compute_median_returnsZero() {
    assertThat(InverseGaussianCdf.compute(0.5)).isWithin(1e-15).of(0.0);
  }

  @Test
  public void compute_negativeZero_returnsNegativeInfinity() {
    assertThat(InverseGaussianCdf.compute(-0.0)).isNegativeInfinity();
  }

  @Test
  public void compute_subnormalPositiveInput_returnsFiniteValue() {
    // Use Math.scalb(1.0, -1074) (= Double.MIN_VALUE) to satisfy MinPositive lint without
    // requiring Guava >= 33.8.0 (OSS MODULE.bazel pins guava:33.5.0-jre).
    double result = InverseGaussianCdf.compute(Math.scalb(1.0, -1074));
    assertThat(result).isFinite();
    assertThat(result).isWithin(1e-6).of(-38.46740562);
  }

  @Test
  public void compute_regionBoundaries_returnAccurateResult(@TestParameter BoundaryCase testCase) {
    double output = InverseGaussianCdf.compute(testCase.p);
    assertWithMessage("p = %s", testCase.p)
        .that(output)
        .isWithin(1e-15 * Math.abs(testCase.expected))
        .of(testCase.expected);
  }

  @Test
  @TestParameters({
    // Values generated using scipy.stats.norm.ppf (matching third_party/cephes test cases).
    // p close to 1, such that 1-p is of order 10^(-i).
    "{p: 9.590875421333870943e-01, expectedOutput: 1.740194245526870631}",
    "{p: 9.204728130878864212e-01, expectedOutput: 1.408259059429154192}",
    "{p: 9.987822916403579221e-01, expectedOutput: 3.031252275870897073}",
    "{p: 9.925605599510384236e-01, expectedOutput: 2.435313800480054969}",
    "{p: 9.990447702229799942e-01, expectedOutput: 3.103810031034545780}",
    "{p: 9.996763629885648816e-01, expectedOutput: 3.410990265746212557}",
    "{p: 9.999886033679458164e-01, expectedOutput: 4.235609454156513465}",
    "{p: 9.999760495761819135e-01, expectedOutput: 4.065639908388418711}",
    "{p: 9.999918285209185020e-01, expectedOutput: 4.309763019113548310}",
    "{p: 9.999949297010878313e-01, expectedOutput: 4.414153527077065320}",
    "{p: 9.999995741498048929e-01, expectedOutput: 4.923131819830989464}",
    "{p: 9.999997043019083209e-01, expectedOutput: 4.994005806628707411}",
    "{p: 9.999999930150607064e-01, expectedOutput: 5.673757581075118850}",
    "{p: 9.999999627780953304e-01, expectedOutput: 5.380101608071190533}",
    "{p: 9.999999977551378150e-01, expectedOutput: 5.865059216395640540}",
    "{p: 9.999999987822485448e-01, expectedOutput: 5.965724924667977547}",
    "{p: 9.999999996778887734e-01, expectedOutput: 6.179212230118906746}",
    "{p: 9.999999995927016938e-01, expectedOutput: 6.142056907043477842}",
    "{p: 9.999999999050942501e-01, expectedOutput: 6.369365733005997399}",
    "{p: 9.999999999612441126e-01, expectedOutput: 6.505351784059330456}",
    "{p: 9.999999999965828446e-01, expectedOutput: 6.861077421111833274}",
    "{p: 9.999999999911421966e-01, expectedOutput: 6.723709001398219698}",
    "{p: 9.999999999990811794e-01, expectedOutput: 7.046280136673924055}",
    "{p: 9.999999999995440314e-01, expectedOutput: 7.143183551114592689}",
    "{p: 9.999999999999484857e-01, expectedOutput: 7.436960043545139065}",
    "{p: 9.999999999999442668e-01, expectedOutput: 7.426550452501929911}",
    // Center and lower-tail values, down to p ~ 10^(-301).
    "{p: 7.078202134622706421e-01, expectedOutput: 0.547027887498922949}",
    "{p: 3.389657500740557161e-01, expectedOutput: -0.415287432523317290}",
    "{p: 5.957715702260371615e-02, expectedOutput: -1.558333067730558463}",
    "{p: 7.814884269185563836e-02, expectedOutput: -1.417633872953655949}",
    "{p: 7.031239497917323583e-03, expectedOutput: -2.455663440450241986}",
    "{p: 1.539105641403513232e-03, expectedOutput: -2.959817704299271046}",
    "{p: 4.308036802518321198e-05, expectedOutput: -3.926584208482659655}",
    "{p: 4.792417534760087735e-05, expectedOutput: -3.900868137755500342}",
    "{p: 4.463560376039031862e-09, expectedOutput: -5.749947420065786297}",
    "{p: 4.134037552039715277e-09, expectedOutput: -5.762900076062255295}",
    "{p: 4.483536530421512750e-16, expectedOutput: -8.040229004994209561}",
    "{p: 3.339983254697393151e-16, expectedOutput: -8.076230307018560595}",
    "{p: 6.215336569124182823e-17, expectedOutput: -8.278911061075142186}",
    "{p: 9.601065568608536918e-17, expectedOutput: -8.226962019547366722}",
    "{p: 6.991559520245878977e-18, expectedOutput: -8.535267778973684827}",
    "{p: 7.798272601053520958e-18, expectedOutput: -8.522633695927112285}",
    "{p: 9.753826112572825312e-33, expectedOutput: -11.858220714899207593}",
    "{p: 7.239945717749332110e-33, expectedOutput: -11.883153923605899394}",
    "{p: 3.081878658827924595e-65, expectedOutput: -17.016810944246216764}",
    "{p: 5.854299805089005594e-65, expectedOutput: -16.979192464465175050}",
    "{p: 8.217571054262086351e-129, expectedOutput: -24.117363860470103276}",
    "{p: 5.940660698416777152e-129, expectedOutput: -24.130790266655189669}",
    "{p: 7.104594771285924014e-257, expectedOutput: -34.215527373144254852}",
    "{p: 5.222691495369919695e-257, expectedOutput: -34.224512368673167373}",
    "{p: 9.255861729918001973e-301, expectedOutput: -37.049182013454569073}",
    "{p: 5.918562407887116857e-301, expectedOutput: -37.061240736056291212}",
  })
  public void compute_relativeAccuracy(double p, double expectedOutput) {
    double output = InverseGaussianCdf.compute(p);
    assertWithMessage("p = %s", p)
        .that(output)
        .isWithin(1e-15 * Math.abs(expectedOutput))
        .of(expectedOutput);
  }

  private enum BoundaryCase {
    EXP_MINUS_2(Math.exp(-2.0), -1.1015196284987503),
    ONE_MINUS_EXP_MINUS_2(1.0 - Math.exp(-2.0), 1.1015196284987503),
    EXP_MINUS_32(Math.exp(-32.0), -7.620199825255225),
    BELOW_EXP_MINUS_32(Math.nextDown(Math.exp(-32.0)), -7.620199825255225),
    BELOW_ONE(Math.nextDown(1.0), 8.209536151601387),
    ABOVE_HALF(0.5 + 1e-12, 2.5065728237018603e-12),
    BELOW_HALF(0.5 - 1e-12, -2.5065728237018603e-12);

    final double p;
    final double expected;

    BoundaryCase(double p, double expected) {
      this.p = p;
      this.expected = expected;
    }
  }
}
