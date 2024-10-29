/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.privacy.differentialprivacy.pipelinedp4j.core

import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalEncoderFactory

data class TestDataRow(val privacyId: String, val partitionKey: String, val value: Double = 0.0) {
  private constructor() : this("defaultPrivacyId", "defaultPartitionKey", 0.0)
}

val testDataExtractors = testDataExtractors(LocalEncoderFactory())

fun testDataExtractors(encoderFactory: EncoderFactory) =
  DataExtractors.from<TestDataRow, String, String>(
    { dataRow -> dataRow.privacyId },
    encoderFactory.strings(),
    { dataRow -> dataRow.partitionKey },
    encoderFactory.strings(),
    { dataRow -> dataRow.value },
  )
