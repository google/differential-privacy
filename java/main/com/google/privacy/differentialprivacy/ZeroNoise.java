package com.google.privacy.differentialprivacy;

import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType;
import javax.annotation.Nullable;

/** {@link Noise} implementation that adds 0 noise. Should be used in tests only. */
public final class ZeroNoise implements Noise {

  @Override
  public double addNoise(
      double x, int l0Sensitivity, double lInfSensitivity, double epsilon, @Nullable Double delta) {
    return x;
  }

  @Override
  public long addNoise(
      long x, int l0Sensitivity, long lInfSensitivity, double epsilon, @Nullable Double delta) {
    return x;
  }

  @Override
  public ConfidenceInterval computeConfidenceInterval(
      double noisedX,
      int l0Sensitivity,
      double lInfSensitivity,
      double epsilon,
      @Nullable Double delta,
      double alpha) {
    return ConfidenceInterval.create(noisedX, noisedX);
  }

  @Override
  public ConfidenceInterval computeConfidenceInterval(
      long noisedX,
      int l0Sensitivity,
      long lInfSensitivity,
      double epsilon,
      @Nullable Double delta,
      double alpha) {
    return ConfidenceInterval.create(noisedX, noisedX);
  }

  @Override
  public MechanismType getMechanismType() {
    return MechanismType.MECHANISM_NONE;
  }

  @Override
  public double computeQuantile(
      double rank,
      double x,
      int l0Sensitivity,
      double lInfSensitivity,
      double epsilon,
      @Nullable Double delta) {
    return x;
  }
}
