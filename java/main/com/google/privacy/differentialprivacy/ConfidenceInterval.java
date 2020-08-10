package com.google.privacy.differentialprivacy;

import com.google.auto.value.AutoValue;

/** Stores the upper and lower bounds of a confidence interval. */
@AutoValue
public abstract class ConfidenceInterval {
  public abstract double lowerBound();

  public abstract double upperBound();

  public static ConfidenceInterval create(double lowerBound, double upperBound) {
    return new AutoValue_ConfidenceInterval(lowerBound, upperBound);
  }
}
