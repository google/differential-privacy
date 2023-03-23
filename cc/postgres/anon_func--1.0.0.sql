/*
Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

\echo Use "CREATE EXTENSION anon_func" to load this file. \quit

/* Create the aggregates:
 *
 * ANON_COUNT(column, epsilon)
 * ANON_COUNT(column)
 *
 * where column is of any type.
 */

-- Accum for with epsilon.
CREATE FUNCTION anon_count_accum(internal, anyelement, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_count_accum'
LANGUAGE C IMMUTABLE;

-- Accum for no epsilon.
CREATE FUNCTION anon_count_accum(internal, anyelement)
RETURNS internal AS
  'anon_func','anon_count_accum'
LANGUAGE C IMMUTABLE;

-- Extract.
CREATE FUNCTION anon_count_extract(internal) RETURNS bigint AS
  'anon_func','anon_count_extract'
LANGUAGE C IMMUTABLE;

-- Aggregate for with epsilon.
CREATE AGGREGATE anon_count(anyelement, epsilon double precision) (
  SFUNC = anon_count_accum,
  STYPE = internal,
  FINALFUNC = anon_count_extract
);

-- Aggregate for no epsilon.
CREATE AGGREGATE anon_count(anyelement) (
  SFUNC = anon_count_accum,
  STYPE = internal,
  FINALFUNC = anon_count_extract
);


/* Create the aggregates:
 *
 * ANON_SUM(column, epsilon)
 * ANON_SUM(column)
 * ANON_SUM(column, lower, upper, epsilon)
 * ANON_SUM(column, lower, upper)
 *
 * where column can be double, bigint, integer, or smallint type.
 */

-- Accum for double type, auto bounding, with epsilon.
CREATE FUNCTION anon_sum_accum(internal, entry double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_sum_accum_double'
LANGUAGE C IMMUTABLE;

-- Accum for double type, auto bounding, no epsilon.
CREATE FUNCTION anon_sum_accum(internal, entry double precision)
RETURNS internal AS
  'anon_func','anon_sum_accum_double'
LANGUAGE C IMMUTABLE;

-- Accum for bigint type, auto bounding, with epsilon.
CREATE FUNCTION anon_sum_accum(internal, entry bigint, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_sum_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for bigint type, auto bounding, no epsilon.
CREATE FUNCTION anon_sum_accum(internal, entry bigint)
RETURNS internal AS
  'anon_func','anon_sum_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for integer type, auto bounding, with epsilon.
CREATE FUNCTION anon_sum_accum(internal, entry integer, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_sum_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for integer type, auto bounding, no epsilon.
CREATE FUNCTION anon_sum_accum(internal, entry integer)
RETURNS internal AS
  'anon_func','anon_sum_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for smallint type, auto bounding, with epsilon.
CREATE FUNCTION anon_sum_accum(internal, entry smallint, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_sum_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for smallint type, auto bounding, no epsilon.
CREATE FUNCTION anon_sum_accum(internal, entry smallint)
RETURNS internal AS
  'anon_func','anon_sum_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for double type, manual bounding, with epsilon.
CREATE FUNCTION anon_sum_with_bounds_accum(internal, entry double precision, lb double precision,
  ub double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_sum_with_bounds_accum_double'
LANGUAGE C IMMUTABLE;

-- Accum for double type, manual bounding, no epsilon.
CREATE FUNCTION anon_sum_with_bounds_accum(internal, entry double precision, lb double precision,
  ub double precision)
RETURNS internal AS
  'anon_func','anon_sum_with_bounds_accum_double'
LANGUAGE C IMMUTABLE;

-- Accum for bigint type, manual bounding, with epsilon.
CREATE FUNCTION anon_sum_with_bounds_accum(internal, entry bigint, lb double precision,
  ub double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_sum_with_bounds_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for bigint type, manual bounding, no epsilon.
CREATE FUNCTION anon_sum_with_bounds_accum(internal, entry bigint, lb double precision,
  ub double precision)
RETURNS internal AS
  'anon_func','anon_sum_with_bounds_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for integer type, manual bounding, with epsilon.
CREATE FUNCTION anon_sum_with_bounds_accum(internal, entry integer, lb double precision,
  ub double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_sum_with_bounds_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for integer type, manual bounding, no epsilon.
CREATE FUNCTION anon_sum_with_bounds_accum(internal, entry integer, lb double precision,
  ub double precision)
RETURNS internal AS
  'anon_func','anon_sum_with_bounds_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for smallint type, manual bounding, with epsilon.
CREATE FUNCTION anon_sum_with_bounds_accum(internal, entry smallint, lb double precision,
  ub double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_sum_with_bounds_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for smallint type, manual bounding, no epsilon.
CREATE FUNCTION anon_sum_with_bounds_accum(internal, entry smallint, lb double precision,
  ub double precision)
RETURNS internal AS
  'anon_func','anon_sum_with_bounds_accum_int'
LANGUAGE C IMMUTABLE;

-- Extract for double type.
CREATE FUNCTION anon_sum_extract_double(internal) RETURNS double precision AS
  'anon_func','anon_sum_extract_double'
LANGUAGE C IMMUTABLE;

-- Extract for int type.
CREATE FUNCTION anon_sum_extract_int(internal) RETURNS bigint AS
  'anon_func','anon_sum_extract_int'
LANGUAGE C IMMUTABLE;

-- Aggregate for double type, auto bounding, with epsilon.
CREATE AGGREGATE anon_sum(entry double precision, epsilon double precision) (
  SFUNC = anon_sum_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_double
);

-- Aggregate for double type, auto bounding, no epsilon.
CREATE AGGREGATE anon_sum(entry double precision) (
  SFUNC = anon_sum_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_double
);

-- Aggregate for bigint type, auto bounding, with epsilon.
CREATE AGGREGATE anon_sum(entry bigint, epsilon double precision) (
  SFUNC = anon_sum_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_int
);

-- Aggregate for bigint type, auto bounding, no epsilon.
CREATE AGGREGATE anon_sum(entry bigint) (
  SFUNC = anon_sum_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_int
);

-- Aggregate for integer type, auto bounding, with epsilon.
CREATE AGGREGATE anon_sum(entry integer, epsilon double precision) (
  SFUNC = anon_sum_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_int
);

-- Aggregate for integer type, auto bounding, no epsilon.
CREATE AGGREGATE anon_sum(entry integer) (
  SFUNC = anon_sum_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_int
);

-- Aggregate for smallint type, auto bounding, with epsilon.
CREATE AGGREGATE anon_sum(entry smallint, epsilon double precision) (
  SFUNC = anon_sum_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_int
);

-- Aggregate for smallint type, auto bounding, no epsilon.
CREATE AGGREGATE anon_sum(entry smallint) (
  SFUNC = anon_sum_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_int
);

-- Aggregate for bigint type, manual bounding, with epsilon.
CREATE AGGREGATE anon_sum_with_bounds(entry bigint, lb double precision, ub double precision,
  epsilon double precision) (
  SFUNC = anon_sum_with_bounds_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_int
);

-- Aggregate for bigint type, manual bounding, no epsilon.
CREATE AGGREGATE anon_sum_with_bounds(entry bigint, lb double precision, ub double precision) (
  SFUNC = anon_sum_with_bounds_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_int
);

-- Aggregate for double type, manual bounding, with epsilon.
CREATE AGGREGATE anon_sum_with_bounds(entry double precision, lb double precision,
    ub double precision, epsilon double precision) (
  SFUNC = anon_sum_with_bounds_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_double
);

-- Aggregate for double type, manual bounding, no epsilon.
CREATE AGGREGATE anon_sum_with_bounds(entry double precision, lb double precision,
    ub double precision) (
  SFUNC = anon_sum_with_bounds_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_double
);

-- Aggregate for integer type, manual bounding, with epsilon.
CREATE AGGREGATE anon_sum_with_bounds(entry integer, lb double precision, ub double precision,
    epsilon double precision) (
  SFUNC = anon_sum_with_bounds_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_int
);

-- Aggregate for integer type, manual bounding, no epsilon.
CREATE AGGREGATE anon_sum_with_bounds(entry integer, lb double precision, ub double precision) (
  SFUNC = anon_sum_with_bounds_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_int
);


-- Aggregate for smallint type, manual bounding, with epsilon.
CREATE AGGREGATE anon_sum_with_bounds(entry smallint, lb double precision, ub double precision,
    epsilon double precision) (
  SFUNC = anon_sum_with_bounds_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_int
);

-- Aggregate for smallint type, manual bounding, no epsilon.
CREATE AGGREGATE anon_sum_with_bounds(entry smallint, lb double precision, ub double precision) (
  SFUNC = anon_sum_with_bounds_accum,
  STYPE = internal,
  FINALFUNC = anon_sum_extract_int
);


/* Create the aggregates:
 *
 * ANON_AVG(column, epsilon)
 * ANON_AVG(column)
 * ANON_AVG(column, lower, upper, epsilon)
 * ANON_AVG(column, lower, upper)
 *
 * where column can be any numeric type smaller than double precision.
 */

-- Accum for auto bounding, with epsilon.
CREATE FUNCTION anon_avg_accum(internal, entry double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_avg_accum'
LANGUAGE C IMMUTABLE;

-- Accum for auto bounding, no epsilon.
CREATE FUNCTION anon_avg_accum(internal, entry double precision)
RETURNS internal AS
  'anon_func','anon_avg_accum'
LANGUAGE C IMMUTABLE;

-- Accum for manual bounding, with epsilon.
CREATE FUNCTION anon_avg_with_bounds_accum(internal, entry double precision, lb double precision,
  ub double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_avg_with_bounds_accum'
LANGUAGE C IMMUTABLE;

-- Accum for manual bounding, no epsilon.
CREATE FUNCTION anon_avg_with_bounds_accum(internal, entry double precision, lb double precision,
  ub double precision)
RETURNS internal AS
  'anon_func','anon_avg_with_bounds_accum'
LANGUAGE C IMMUTABLE;

-- Extract.
CREATE FUNCTION anon_avg_extract(internal) RETURNS double precision AS
  'anon_func','anon_avg_extract'
LANGUAGE C IMMUTABLE;

-- Aggregate for auto bounding, with epsilon.
CREATE AGGREGATE anon_avg(entry double precision, epsilon double precision) (
  SFUNC = anon_avg_accum,
  STYPE = internal,
  FINALFUNC = anon_avg_extract
);

-- Aggregate for auto bounding, no epsilon.
CREATE AGGREGATE anon_avg(entry double precision) (
  SFUNC = anon_avg_accum,
  STYPE = internal,
  FINALFUNC = anon_avg_extract
);

-- Aggregate for manual bounding, with epsilon.
CREATE AGGREGATE anon_avg_with_bounds(entry double precision, lb double precision,
    ub double precision, epsilon double precision) (
  SFUNC = anon_avg_with_bounds_accum,
  STYPE = internal,
  FINALFUNC = anon_avg_extract
);

-- Aggregate for manual bounding, no epsilon.
CREATE AGGREGATE anon_avg_with_bounds(entry double precision, lb double precision,
    ub double precision) (
  SFUNC = anon_avg_with_bounds_accum,
  STYPE = internal,
  FINALFUNC = anon_avg_extract
);


/* Create the aggregates:
 *
 * ANON_VAR(column, epsilon)
 * ANON_VAR(column)
 * ANON_VAR(column, lower, upper, epsilon)
 * ANON_VAR(column, lower, upper)
 *
 * where column can be any numeric type smaller than double precision.
 */

-- Accum for auto bounding, with epsilon.
CREATE FUNCTION anon_var_accum(internal, entry double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_var_accum'
LANGUAGE C IMMUTABLE;

-- Accum for auto bounding, no epsilon.
CREATE FUNCTION anon_var_accum(internal, entry double precision)
RETURNS internal AS
  'anon_func','anon_var_accum'
LANGUAGE C IMMUTABLE;

-- Accum for manual bounding, with epsilon.
CREATE FUNCTION anon_var_with_bounds_accum(internal, entry double precision, lb double precision,
  ub double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_var_with_bounds_accum'
LANGUAGE C IMMUTABLE;

-- Accum for manual bounding, no epsilon.
CREATE FUNCTION anon_var_with_bounds_accum(internal, entry double precision, lb double precision,
  ub double precision)
RETURNS internal AS
  'anon_func','anon_var_with_bounds_accum'
LANGUAGE C IMMUTABLE;

-- Extract.
CREATE FUNCTION anon_var_extract(internal) RETURNS double precision AS
  'anon_func','anon_var_extract'
LANGUAGE C IMMUTABLE;

-- Aggregate for auto bounding, with epsilon.
CREATE AGGREGATE anon_var(entry double precision, epsilon double precision) (
  SFUNC = anon_var_accum,
  STYPE = internal,
  FINALFUNC = anon_var_extract
);

-- Aggregate for auto bounding, no epsilon.
CREATE AGGREGATE anon_var(entry double precision) (
  SFUNC = anon_var_accum,
  STYPE = internal,
  FINALFUNC = anon_var_extract
);

-- Aggregate for manual bounding, with epsilon.
CREATE AGGREGATE anon_var_with_bounds(entry double precision, lb double precision,
    ub double precision, epsilon double precision) (
  SFUNC = anon_var_with_bounds_accum,
  STYPE = internal,
  FINALFUNC = anon_var_extract
);

-- Aggregate for manual bounding, no epsilon.
CREATE AGGREGATE anon_var_with_bounds(entry double precision, lb double precision,
    ub double precision) (
  SFUNC = anon_var_with_bounds_accum,
  STYPE = internal,
  FINALFUNC = anon_var_extract
);


/* Create the aggregates:
 *
 * ANON_STDDEV(column, epsilon)
 * ANON_STDDEV(column)
 * ANON_STDDEV(column, lower, upper, epsilon)
 * ANON_STDDEV(column, lower, upper)
 *
 * where column can be any numeric type smaller than double precision.
 */

-- Accum for auto bounding, with epsilon.
CREATE FUNCTION anon_stddev_accum(internal, entry double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_stddev_accum'
LANGUAGE C IMMUTABLE;

-- Accum for auto bounding, no epsilon.
CREATE FUNCTION anon_stddev_accum(internal, entry double precision)
RETURNS internal AS
  'anon_func','anon_stddev_accum'
LANGUAGE C IMMUTABLE;

-- Accum for manual bounding, with epsilon.
CREATE FUNCTION anon_stddev_with_bounds_accum(internal, entry double precision, lb double precision,
  ub double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_stddev_with_bounds_accum'
LANGUAGE C IMMUTABLE;

-- Accum for manual bounding, no epsilon.
CREATE FUNCTION anon_stddev_with_bounds_accum(internal, entry double precision, lb double precision,
  ub double precision)
RETURNS internal AS
  'anon_func','anon_stddev_with_bounds_accum'
LANGUAGE C IMMUTABLE;

-- Extract.
CREATE FUNCTION anon_stddev_extract(internal) RETURNS double precision AS
  'anon_func','anon_stddev_extract'
LANGUAGE C IMMUTABLE;

-- Aggregate for auto bounding, with epsilon.
CREATE AGGREGATE anon_stddev(entry double precision, epsilon double precision) (
  SFUNC = anon_stddev_accum,
  STYPE = internal,
  FINALFUNC = anon_stddev_extract
);

-- Aggregate for auto bounding, no epsilon.
CREATE AGGREGATE anon_stddev(entry double precision) (
  SFUNC = anon_stddev_accum,
  STYPE = internal,
  FINALFUNC = anon_stddev_extract
);

-- Aggregate for manual bounding, with epsilon.
CREATE AGGREGATE anon_stddev_with_bounds(entry double precision, lb double precision,
    ub double precision, epsilon double precision) (
  SFUNC = anon_stddev_with_bounds_accum,
  STYPE = internal,
  FINALFUNC = anon_stddev_extract
);

-- Aggregate for manual bounding, no epsilon.
CREATE AGGREGATE anon_stddev_with_bounds(entry double precision, lb double precision,
    ub double precision) (
  SFUNC = anon_stddev_with_bounds_accum,
  STYPE = internal,
  FINALFUNC = anon_stddev_extract
);


/* Create the aggregates:
 *
 * ANON_NTILE(column, lower, upper, epsilon)
 * ANON_NTILE(column, lower, upper)
 *
 * where column can be double, bigint, integer, or smallint type.
 */

-- Accum for double type, with epsilon.
CREATE FUNCTION anon_ntile_accum(internal, entry double precision, percentile double precision,
  lb double precision, ub double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_ntile_accum_double'
LANGUAGE C IMMUTABLE;

-- Accum for double type, no epsilon.
CREATE FUNCTION anon_ntile_accum(internal, entry double precision, percentile double precision,
  lb double precision, ub double precision)
RETURNS internal AS
  'anon_func','anon_ntile_accum_double'
LANGUAGE C IMMUTABLE;

-- Accum for bigint type, with epsilon.
CREATE FUNCTION anon_ntile_accum(internal, entry bigint, percentile double precision,
  lb double precision, ub double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_ntile_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for bigint type, no epsilon.
CREATE FUNCTION anon_ntile_accum(internal, entry bigint, percentile double precision,
  lb double precision, ub double precision)
RETURNS internal AS
  'anon_func','anon_ntile_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for integer type, with epsilon.
CREATE FUNCTION anon_ntile_accum(internal, entry integer, percentile double precision,
  lb double precision, ub double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_ntile_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for integer type, no epsilon.
CREATE FUNCTION anon_ntile_accum(internal, entry integer, percentile double precision,
  lb double precision, ub double precision)
RETURNS internal AS
  'anon_func','anon_ntile_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for smallint type, with epsilon.
CREATE FUNCTION anon_ntile_accum(internal, entry smallint, percentile double precision,
  lb double precision, ub double precision, epsilon double precision)
RETURNS internal AS
  'anon_func','anon_ntile_accum_int'
LANGUAGE C IMMUTABLE;

-- Accum for smallint type, no epsilon.
CREATE FUNCTION anon_ntile_accum(internal, entry smallint, percentile double precision,
  lb double precision, ub double precision)
RETURNS internal AS
  'anon_func','anon_ntile_accum_int'
LANGUAGE C IMMUTABLE;

-- Extract for double type.
CREATE FUNCTION anon_ntile_extract_double(internal) RETURNS double precision AS
  'anon_func','anon_ntile_extract_double'
LANGUAGE C IMMUTABLE;

-- Extract for int type.
CREATE FUNCTION anon_ntile_extract_int(internal) RETURNS bigint AS
  'anon_func','anon_ntile_extract_int'
LANGUAGE C IMMUTABLE;

-- Aggregate for double type, with epsilon.
CREATE AGGREGATE anon_ntile(entry double precision, percentile double precision,
    lb double precision, ub double precision, epsilon double precision) (
  SFUNC = anon_ntile_accum,
  STYPE = internal,
  FINALFUNC = anon_ntile_extract_double
);

-- Aggregate for double type, no epsilon.
CREATE AGGREGATE anon_ntile(entry double precision, percentile double precision,
  lb double precision, ub double precision) (
  SFUNC = anon_ntile_accum,
  STYPE = internal,
  FINALFUNC = anon_ntile_extract_double
);

-- Aggregate for bigint type, with epsilon.
CREATE AGGREGATE anon_ntile(entry bigint, percentile double precision, lb double precision,
    ub double precision, epsilon double precision) (
  SFUNC = anon_ntile_accum,
  STYPE = internal,
  FINALFUNC = anon_ntile_extract_int
);

-- Aggregate for bigint type, no epsilon.
CREATE AGGREGATE anon_ntile(entry bigint, percentile double precision, lb double precision,
    ub double precision) (
  SFUNC = anon_ntile_accum,
  STYPE = internal,
  FINALFUNC = anon_ntile_extract_int
);

-- Aggregate for integer type, with epsilon.
CREATE AGGREGATE anon_ntile(entry integer, percentile double precision,  lb double precision,
    ub double precision, epsilon double precision) (
  SFUNC = anon_ntile_accum,
  STYPE = internal,
  FINALFUNC = anon_ntile_extract_int
);

-- Aggregate for integer type, no epsilon.
CREATE AGGREGATE anon_ntile(entry integer, percentile double precision,  lb double precision,
    ub double precision) (
  SFUNC = anon_ntile_accum,
  STYPE = internal,
  FINALFUNC = anon_ntile_extract_int
);

-- Aggregate for smallint type, with epsilon.
CREATE AGGREGATE anon_ntile(entry smallint, percentile double precision,  lb double precision,
    ub double precision, epsilon double precision) (
  SFUNC = anon_ntile_accum,
  STYPE = internal,
  FINALFUNC = anon_ntile_extract_int
);

-- Aggregate for smallint type, no epsilon.
CREATE AGGREGATE anon_ntile(entry smallint, percentile double precision,  lb double precision,
    ub double precision) (
  SFUNC = anon_ntile_accum,
  STYPE = internal,
  FINALFUNC = anon_ntile_extract_int
);
