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

package com.google.privacy.differentialprivacy;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class SecureNoiseMathTest {

  @Test
  public void ceilPowerOfTwo_nonPositiveInput_throwsException() {
    assertThrows(IllegalArgumentException.class, () -> SecureNoiseMath.ceilPowerOfTwo(0.0));
    assertThrows(IllegalArgumentException.class, () -> SecureNoiseMath.ceilPowerOfTwo(-1.0));
  }

  @Test
  public void ceilPowerOfTwo_infiniteInput_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () -> SecureNoiseMath.ceilPowerOfTwo(Double.POSITIVE_INFINITY));
    assertThrows(
        IllegalArgumentException.class,
        () -> SecureNoiseMath.ceilPowerOfTwo(Double.NEGATIVE_INFINITY));
  }

  @Test
  public void ceilPowerOfTwo_inputIsNaN_throwsException() {
    assertThrows(IllegalArgumentException.class, () -> SecureNoiseMath.ceilPowerOfTwo(Double.NaN));
  }

  @Test
  public void ceilPowerOfTwo_inputIsGreaterThan2ToThePowerOf1023_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () -> SecureNoiseMath.ceilPowerOfTwo(Math.pow(2.001, 1023)));
    assertThrows(
        IllegalArgumentException.class, () -> SecureNoiseMath.ceilPowerOfTwo(Double.MAX_VALUE));
  }

  @Test
  public void ceilPowerOfTwo_returnsExactPowerOfTwo() {
    // Since x is a finite positive number, x is a power of 2 if and only if it has a mantissa of 0.
    final long mantissaMask = 0x000fffffffffffffL;
    assertThat(Double.doubleToLongBits(SecureNoiseMath.ceilPowerOfTwo(0.123456789)) & mantissaMask)
        .isEqualTo(0x0000000000000000L);
    assertThat(Double.doubleToLongBits(SecureNoiseMath.ceilPowerOfTwo(987654321.0)) & mantissaMask)
        .isEqualTo(0x0000000000000000L);
  }

  @Test
  public void ceilPowerOfTwo_inputIsAPowerOfTwo_returnsInput() {
    for (double exponent = -1022.0; exponent <= 1023.0; exponent++) {
      assertThat(SecureNoiseMath.ceilPowerOfTwo(Math.pow(2.0, exponent)))
          .isEqualTo(Math.pow(2.0, exponent));
    }
  }

  @Test
  public void ceilPowerOfTwo_inputIsNotAPowerOfTwo_hardCodedExamplesReturnCorrectResult() {
    assertThat(SecureNoiseMath.ceilPowerOfTwo(0.2078795763)).isEqualTo(0.25);
    assertThat(SecureNoiseMath.ceilPowerOfTwo(0.6931471805)).isEqualTo(1.0);
    assertThat(SecureNoiseMath.ceilPowerOfTwo(3.1415926535)).isEqualTo(4.0);
    assertThat(SecureNoiseMath.ceilPowerOfTwo(36.4621596072)).isEqualTo(64.0);
  }

  @Test
  public void ceilPowerOfTwo_inputIsNotAPowerOfTwo_exhaustiveExponentCoverReturnsCorrectResult() {
    for (double exponent = -1022.0; exponent <= -1.0; exponent++) {
      assertThat(SecureNoiseMath.ceilPowerOfTwo(Math.pow(2.001, exponent)))
          .isEqualTo(Math.pow(2.0, exponent));
    }
    assertThat(SecureNoiseMath.ceilPowerOfTwo(0.99)).isEqualTo(1.0);
    for (double exponent = 1.0; exponent <= 1022.0; exponent++) {
      assertThat(SecureNoiseMath.ceilPowerOfTwo(Math.pow(2.001, exponent)))
          .isEqualTo(Math.pow(2.0, exponent + 1.0));
    }
  }

  @Test
  public void roundToMultipleOfPowerOfTwo_granularityIsNotAPowerOfTwo_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () -> SecureNoiseMath.roundToMultipleOfPowerOfTwo(2.0, 0.0));
    assertThrows(
        IllegalArgumentException.class,
        () -> SecureNoiseMath.roundToMultipleOfPowerOfTwo(2.0, 0.3));
    assertThrows(
        IllegalArgumentException.class,
        () -> SecureNoiseMath.roundToMultipleOfPowerOfTwo(2.0, -3.0));
    assertThrows(
        IllegalArgumentException.class,
        () -> SecureNoiseMath.roundToMultipleOfPowerOfTwo(2.0, -1.0));
    assertThrows(
        IllegalArgumentException.class,
        () -> SecureNoiseMath.roundToMultipleOfPowerOfTwo(2.0, Double.NEGATIVE_INFINITY));
    assertThrows(
        IllegalArgumentException.class,
        () -> SecureNoiseMath.roundToMultipleOfPowerOfTwo(2.0, Double.POSITIVE_INFINITY));
  }

  @Test
  public void roundToMultipleOfPowerOfTwo_inputIsAMultipleOfTheGranularity_returnsInput() {
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(0.0, 0.5)).isEqualTo(0.0);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(0.125, 0.125)).isEqualTo(0.125);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(-0.125, 0.125)).isEqualTo(-0.125);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(16512.0, 1.0)).isEqualTo(16512.0);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(-16512.0, 1.0)).isEqualTo(-16512.0);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(3936.0, 32.0)).isEqualTo(3936.0);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(-3936.0, 32.0)).isEqualTo(-3936.0);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(7.9990234375, 0.0009765625))
        .isEqualTo(7.9990234375);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(-7.9990234375, 0.0009765625))
        .isEqualTo(-7.9990234375);
  }

  @Test
  public void
      roundToMultipleOfPowerOfTwo_inputIsNotAMultipleOfTheGranularity_returnsCorrectResult() {
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(0.124, 0.125)).isEqualTo(0.125);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(-0.124, 0.125)).isEqualTo(-0.125);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(0.126, 0.125)).isEqualTo(0.125);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(-0.126, 0.125)).isEqualTo(-0.125);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(16512.499, 1.0)).isEqualTo(16512.0);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(-16512.499, 1.0)).isEqualTo(-16512.0);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(16511.501, 1.0)).isEqualTo(16512.0);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(-16511.501, 1.0)).isEqualTo(-16512.0);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(3920.3257, 32.0)).isEqualTo(3936.0);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(-3920.3257, 32.0)).isEqualTo(-3936.0);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(3951.7654, 32.0)).isEqualTo(3936.0);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(-3951.7654, 32.0)).isEqualTo(-3936.0);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(7.9990232355, 0.0009765625))
        .isEqualTo(7.9990234375);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(-7.9990232355, 0.0009765625))
        .isEqualTo(-7.9990234375);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(7.9990514315, 0.0009765625))
        .isEqualTo(7.9990234375);
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo(-7.9990514315, 0.0009765625))
        .isEqualTo(-7.9990234375);
  }

  @Test
  public void roundToMultipleOfPowerOfTwo_scaledInputExceedsLongRange_returnsInput() {
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo((double) (Long.MAX_VALUE - 1), 0.125))
        .isEqualTo((double) (Long.MAX_VALUE - 1));
    assertThat(SecureNoiseMath.roundToMultipleOfPowerOfTwo((double) (Long.MIN_VALUE + 1), 0.125))
        .isEqualTo((double) (Long.MIN_VALUE + 1));
  }

  @Test
  public void roundToMultiple_granularityIsNotPositive_throwsException() {
    assertThrows(IllegalArgumentException.class, () -> SecureNoiseMath.roundToMultiple(1, 0));
    assertThrows(IllegalArgumentException.class, () -> SecureNoiseMath.roundToMultiple(1, -1));
    assertThrows(
        IllegalArgumentException.class,
        () -> SecureNoiseMath.roundToMultiple(1, Integer.MIN_VALUE));
  }

  @Test
  public void roundToMultiple_granularityIsOne_returnsInput() {
    assertThat(SecureNoiseMath.roundToMultiple(0, 1)).isEqualTo(0);
    assertThat(SecureNoiseMath.roundToMultiple(1, 1)).isEqualTo(1);
    assertThat(SecureNoiseMath.roundToMultiple(-1, 1)).isEqualTo(-1);
    assertThat(SecureNoiseMath.roundToMultiple(648391, 1)).isEqualTo(648391);
    assertThat(SecureNoiseMath.roundToMultiple(-648391, 1)).isEqualTo(-648391);
  }

  @Test
  public void roundToMultiple_granularityIsEven_returnsCorrectResult() {
    assertThat(SecureNoiseMath.roundToMultiple(0, 4)).isEqualTo(0);
    assertThat(SecureNoiseMath.roundToMultiple(1, 4)).isEqualTo(0);
    assertThat(SecureNoiseMath.roundToMultiple(2, 4)).isEqualTo(4);
    assertThat(SecureNoiseMath.roundToMultiple(3, 4)).isEqualTo(4);
    assertThat(SecureNoiseMath.roundToMultiple(4, 4)).isEqualTo(4);
    assertThat(SecureNoiseMath.roundToMultiple(-1, 4)).isEqualTo(0);
    assertThat(SecureNoiseMath.roundToMultiple(-2, 4)).isEqualTo(0);
    assertThat(SecureNoiseMath.roundToMultiple(-3, 4)).isEqualTo(-4);
    assertThat(SecureNoiseMath.roundToMultiple(-4, 4)).isEqualTo(-4);
    assertThat(SecureNoiseMath.roundToMultiple(648389, 4)).isEqualTo(648388);
    assertThat(SecureNoiseMath.roundToMultiple(648390, 4)).isEqualTo(648392);
    assertThat(SecureNoiseMath.roundToMultiple(648391, 4)).isEqualTo(648392);
    assertThat(SecureNoiseMath.roundToMultiple(648392, 4)).isEqualTo(648392);
    assertThat(SecureNoiseMath.roundToMultiple(-648389, 4)).isEqualTo(-648388);
    assertThat(SecureNoiseMath.roundToMultiple(-648390, 4)).isEqualTo(-648388);
    assertThat(SecureNoiseMath.roundToMultiple(-648391, 4)).isEqualTo(-648392);
    assertThat(SecureNoiseMath.roundToMultiple(-648392, 4)).isEqualTo(-648392);
  }

  @Test
  public void roundToMultiple_granularityIsOdd_returnsCorrectResult() {
    assertThat(SecureNoiseMath.roundToMultiple(0, 3)).isEqualTo(0);
    assertThat(SecureNoiseMath.roundToMultiple(1, 3)).isEqualTo(0);
    assertThat(SecureNoiseMath.roundToMultiple(2, 3)).isEqualTo(3);
    assertThat(SecureNoiseMath.roundToMultiple(3, 3)).isEqualTo(3);
    assertThat(SecureNoiseMath.roundToMultiple(-1, 3)).isEqualTo(0);
    assertThat(SecureNoiseMath.roundToMultiple(-2, 3)).isEqualTo(-3);
    assertThat(SecureNoiseMath.roundToMultiple(-3, 3)).isEqualTo(-3);
    assertThat(SecureNoiseMath.roundToMultiple(648391, 3)).isEqualTo(648390);
    assertThat(SecureNoiseMath.roundToMultiple(648392, 3)).isEqualTo(648393);
    assertThat(SecureNoiseMath.roundToMultiple(648393, 3)).isEqualTo(648393);
    assertThat(SecureNoiseMath.roundToMultiple(-648391, 3)).isEqualTo(-648390);
    assertThat(SecureNoiseMath.roundToMultiple(-648392, 3)).isEqualTo(-648393);
    assertThat(SecureNoiseMath.roundToMultiple(-648393, 3)).isEqualTo(-648393);
  }

  @Test
  public void nextLargerDouble_inputRepresentableAsDouble_returnsDoubleOfSameValue() {
    assertThat(SecureNoiseMath.nextLargerDouble(0L)).isEqualTo(0.0);
    assertThat(SecureNoiseMath.nextLargerDouble(1L)).isEqualTo(1.0);
    assertThat(SecureNoiseMath.nextLargerDouble(-1L)).isEqualTo(-1.0);

    // Smallest positive long value for which the next double is a distance of 2 away
    assertThat(SecureNoiseMath.nextLargerDouble(9007199254740992L)).isEqualTo(9007199254740992.0);
    // Largest negative long value for which the previous double is a distance of 2 away
    assertThat(SecureNoiseMath.nextLargerDouble(-9007199254740992L)).isEqualTo(-9007199254740992.0);

    assertThat((long) SecureNoiseMath.nextLargerDouble(8646911284551352320L))
        .isEqualTo(8646911284551352320L);
    assertThat((long) SecureNoiseMath.nextLargerDouble(-8646911284551352320L))
        .isEqualTo(-8646911284551352320L);

    // Largest long value accurately representable as a double
    assertThat((long) SecureNoiseMath.nextLargerDouble(Long.MAX_VALUE - 1023))
        .isEqualTo(Long.MAX_VALUE - 1023);
    // Smallest long value accurately representable as a double
    assertThat((long) SecureNoiseMath.nextLargerDouble(Long.MIN_VALUE)).isEqualTo(Long.MIN_VALUE);
  }

  @Test
  public void nextLargerDouble_inputNotRepresentableAsDouble_returnsNextLargerDouble() {
    // Smallest positive long value not representable as a double
    assertThat((long) SecureNoiseMath.nextLargerDouble(9007199254740993L))
        .isEqualTo(9007199254740994L);
    // Largest negative long value not representable as a double
    assertThat((long) SecureNoiseMath.nextLargerDouble(-9007199254740993L))
        .isEqualTo(-9007199254740992L);

    assertThat((long) SecureNoiseMath.nextLargerDouble(8646911284551352321L))
        .isEqualTo(8646911284551353344L);
    assertThat((long) SecureNoiseMath.nextLargerDouble(8646911284551353343L))
        .isEqualTo(8646911284551353344L);
    assertThat((long) SecureNoiseMath.nextLargerDouble(-8646911284551352321L))
        .isEqualTo(-8646911284551352320L);
    assertThat((long) SecureNoiseMath.nextLargerDouble(-8646911284551353343L))
        .isEqualTo(-8646911284551352320L);

    // Casting Long.MAX_VALUE to double should result in the next larger double, i.e., 2^63
    assertThat(SecureNoiseMath.nextLargerDouble(Long.MAX_VALUE)).isEqualTo(Math.pow(2.0, 63.0));
  }

  @Test
  public void nextSmallerDouble_inputRepresentableAsDouble_returnsDoubleOfSameValue() {
    assertThat(SecureNoiseMath.nextSmallerDouble(0L)).isEqualTo(0.0);
    assertThat(SecureNoiseMath.nextSmallerDouble(1L)).isEqualTo(1.0);
    assertThat(SecureNoiseMath.nextSmallerDouble(-1L)).isEqualTo(-1.0);

    // Smallest positive long value for which the next double is a distance of 2 away
    assertThat(SecureNoiseMath.nextSmallerDouble(9007199254740992L)).isEqualTo(9007199254740992.0);
    // Largest negative long value for which the previous double is a distance of 2 away
    assertThat(SecureNoiseMath.nextSmallerDouble(-9007199254740992L))
        .isEqualTo(-9007199254740992.0);

    assertThat((long) SecureNoiseMath.nextSmallerDouble(8646911284551352320L))
        .isEqualTo(8646911284551352320L);
    assertThat((long) SecureNoiseMath.nextSmallerDouble(-8646911284551352320L))
        .isEqualTo(-8646911284551352320L);

    // Largest long value accurately representable as a double
    assertThat((long) SecureNoiseMath.nextSmallerDouble(Long.MAX_VALUE - 1023))
        .isEqualTo(Long.MAX_VALUE - 1023);
    // Smallest long value accurately representable as a double
    assertThat((long) SecureNoiseMath.nextSmallerDouble(Long.MIN_VALUE)).isEqualTo(Long.MIN_VALUE);
  }

  @Test
  public void nextSmallerDouble_inputNotRepresentableAsDouble_returnsNextSmallerDouble() {
    // Smallest positive long value not representable as a double
    assertThat((long) SecureNoiseMath.nextSmallerDouble(9007199254740993L))
        .isEqualTo(9007199254740992L);
    // Largest negative long value not representable as a double
    assertThat((long) SecureNoiseMath.nextSmallerDouble(-9007199254740993L))
        .isEqualTo(-9007199254740994L);

    assertThat((long) SecureNoiseMath.nextSmallerDouble(8646911284551352321L))
        .isEqualTo(8646911284551352320L);
    assertThat((long) SecureNoiseMath.nextSmallerDouble(8646911284551353343L))
        .isEqualTo(8646911284551352320L);
    assertThat((long) SecureNoiseMath.nextSmallerDouble(-8646911284551352321L))
        .isEqualTo(-8646911284551353344L);
    assertThat((long) SecureNoiseMath.nextSmallerDouble(-8646911284551353343L))
        .isEqualTo(-8646911284551353344L);
  }
}
