package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.google.privacy.differentialprivacy.pipelinedp4j.core.DpEngine
import com.google.privacy.differentialprivacy.pipelinedp4j.core.DpEngineBudgetSpec

fun DpEngine.Factory.createSparkEngine(budgetSpec: DpEngineBudgetSpec) =
    create(SparkEncoderFactory(), budgetSpec)
