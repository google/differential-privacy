package com.google.privacy.differentialprivacy;

import com.google.auto.value.AutoValue;

/**
 * Valued-typed class that holds real bounds for the confidence interval.
 */
@AutoValue
public abstract class ConfidenceInterval{

    public abstract double getLowerBound();

    public abstract double getUpperBound();

    public static ConfidenceInterval create(double lowerBound, double upperBound){
        return new AutoValue_ConfidenceInterval(lowerBound, upperBound);
    }
}
