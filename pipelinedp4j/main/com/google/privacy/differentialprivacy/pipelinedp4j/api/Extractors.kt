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

package com.google.privacy.differentialprivacy.pipelinedp4j.api

import java.io.Serializable

// TODO: add documentation on how to create extractors in different languages (Kotlin,
// Java, Scala). In Kotlin you have to write `StringExtractor { it.privacyUnit }` and not just `{
// it.privacyUnit }` otherwise function overload resolution fails.

/**
 * A function that extracts a string from a data row.
 *
 * It is just a wrapper for `(DataRowT) -> String` function to avoid conflicting function overloads.
 * To create a [StringExtractor], you can just pass a lambda `(DataRowT) -> String`, it is the same
 * type.
 */
fun interface StringExtractor<DataRowT : Any> : (DataRowT) -> String, Serializable

/**
 * A function that extracts an integer from a data row.
 *
 * It is just a wrapper for `(DataRowT) -> Int` function to avoid conflicting function overloads. To
 * create a [IntExtractor], you can just pass a lambda `(DataRowT) -> Int`, it is the same type.
 */
fun interface IntExtractor<DataRowT : Any> : (DataRowT) -> Int, Serializable

/**
 * A function that extracts a long integer from a data row.
 *
 * It is just a wrapper for `(DataRowT) -> Long` function to avoid conflicting function overloads.
 * To create a [LongExtractor], you can just pass a lambda `(DataRowT) -> Long`, it is the same
 * type.
 */
fun interface LongExtractor<DataRowT : Any> : (DataRowT) -> Long, Serializable
