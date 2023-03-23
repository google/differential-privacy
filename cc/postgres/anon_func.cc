//
// Copyright 2019 Google LLC
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

extern "C" {

#include "postgres.h"
#include "fmgr.h"
#include "utils/datum.h"

PG_MODULE_MAGIC;

#define CHECK_AGG_CONTEXT(fcinfo)                                 \
  if (!AggCheckCallContext(fcinfo, NULL)) {                       \
    elog(ERROR, "Anon function called in non-aggregate context"); \
  }

/*
 * Headers for functions available in PostgreSQL.
 */

// ANON_COUNT
PG_FUNCTION_INFO_V1(anon_count_accum);
PG_FUNCTION_INFO_V1(anon_count_extract);

// ANON_SUM
PG_FUNCTION_INFO_V1(anon_sum_accum_double);
PG_FUNCTION_INFO_V1(anon_sum_accum_int);
PG_FUNCTION_INFO_V1(anon_sum_with_bounds_accum_double);
PG_FUNCTION_INFO_V1(anon_sum_with_bounds_accum_int);
PG_FUNCTION_INFO_V1(anon_sum_extract_double);
PG_FUNCTION_INFO_V1(anon_sum_extract_int);

// ANON_AVG
PG_FUNCTION_INFO_V1(anon_avg_accum);
PG_FUNCTION_INFO_V1(anon_avg_with_bounds_accum);
PG_FUNCTION_INFO_V1(anon_avg_extract);

// ANON_VAR
PG_FUNCTION_INFO_V1(anon_var_accum);
PG_FUNCTION_INFO_V1(anon_var_with_bounds_accum);
PG_FUNCTION_INFO_V1(anon_var_extract);

// ANON_STDDEV
PG_FUNCTION_INFO_V1(anon_stddev_accum);
PG_FUNCTION_INFO_V1(anon_stddev_with_bounds_accum);
PG_FUNCTION_INFO_V1(anon_stddev_extract);

// ANON_NTILE
PG_FUNCTION_INFO_V1(anon_ntile_accum_double);
PG_FUNCTION_INFO_V1(anon_ntile_accum_int);
PG_FUNCTION_INFO_V1(anon_ntile_extract_double);
PG_FUNCTION_INFO_V1(anon_ntile_extract_int);
}

#include "dp_func.h"

/*
 * Helper functions.
 */

template <typename DpFunction>
void add_arg_entry(PG_FUNCTION_ARGS, DpFunction* func, bool is_integral) {
  bool entry_added;
  if (is_integral) {
    entry_added = func->AddEntry(PG_GETARG_INT64(1));
  } else {
    // Double type.
    entry_added = func->AddEntry(PG_GETARG_FLOAT8(1));
  }
  if (!entry_added) {
    ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
            errmsg("Adding entry to dp function failed.")));
  }
}

// Common code for bounded accum functions.
template <typename DpFunction>
Datum bounded_accum(PG_FUNCTION_ARGS, bool with_bounds, bool is_integral) {
  CHECK_AGG_CONTEXT(fcinfo);
  DpFunction* arg0;

  // Create DpFunction if it doesn't exist.
  if (PG_ARGISNULL(0)) {
    // Grab the optional variables, if provided.
    float8 epsilon = 0, lower = 0, upper = 0;
    bool with_epsilon = false;
    if (with_bounds) {
      if (PG_NARGS() < 4) {
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
            errmsg("Bounds provided but unable to be retrieved.")));
      }
      lower = PG_GETARG_FLOAT8(2);
      upper = PG_GETARG_FLOAT8(3);
    }
    if (with_bounds && PG_NARGS() > 4) {
      epsilon = PG_GETARG_FLOAT8(4);
      with_epsilon = true;
    }
    if (!with_bounds && PG_NARGS() > 2) {
      epsilon = PG_GETARG_FLOAT8(2);
      with_epsilon = true;
    }

    // Construct the DP function.
    std::string err;
    arg0 = new DpFunction(&err, !with_epsilon, epsilon,
                          !with_bounds, lower, upper);
    if (!err.empty()) {
      ereport(ERROR,
              (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
              errmsg("%s", err.c_str())));
    }
  } else {
    arg0 = reinterpret_cast<DpFunction*>(PG_GETARG_POINTER(0));
  }

  add_arg_entry(fcinfo, arg0, is_integral);
  PG_RETURN_POINTER(arg0);
}

// Common extract code for returning integer values. Return null if error.
template <typename DpFunction>
Datum int_extract(PG_FUNCTION_ARGS){
  CHECK_AGG_CONTEXT(fcinfo);
  if (PG_ARGISNULL(0)) {
    PG_RETURN_NULL();
  }
  DpFunction* arg = reinterpret_cast<DpFunction*>(PG_GETARG_POINTER(0));
  std::string err;
  int64_t result = arg->ResultRounded(&err);
  if (!err.empty()) {
    ereport(INFO, (errmsg("%s Returning NULL.", err.c_str())));
    PG_RETURN_NULL();
  }
  delete arg;
  PG_RETURN_INT64(result);
}

// Common extract code for returning double values. Return null if error.
template <typename DpFunction>
Datum double_extract(PG_FUNCTION_ARGS){
  CHECK_AGG_CONTEXT(fcinfo);
  if (PG_ARGISNULL(0)) {
    PG_RETURN_NULL();
  }
  DpFunction* arg = reinterpret_cast<DpFunction*>(PG_GETARG_POINTER(0));
  std::string err;
  double result = arg->Result(&err);
  if (!err.empty()) {
    ereport(INFO, (errmsg("%s Returning NULL.", err.c_str())));
    PG_RETURN_NULL();
  }
  delete arg;
  PG_RETURN_FLOAT8(result);
}


/*
 * ANON_COUNT functions.
 */

Datum anon_count_accum(PG_FUNCTION_ARGS) {
  CHECK_AGG_CONTEXT(fcinfo);
  DpCount* arg0;
  if (PG_ARGISNULL(0)) {
    std::string err;
    if (PG_NARGS() > 2) {
      float8 epsilon = PG_GETARG_FLOAT8(2);
      arg0 = new DpCount(&err, /*default_epsilon=*/false, epsilon);
    } else {
      arg0 = new DpCount(&err);
    }
    if (!err.empty()) {
      ereport(ERROR,
              (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
              errmsg("%s", err.c_str())));
    }
  } else {
    arg0 = reinterpret_cast<DpCount*>(PG_GETARG_POINTER(0));
  }

  // Add a placeholder entry for each element counted.
  if (!arg0->AddEntry(1)) {
    ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
            errmsg("Adding entry to dp function failed.")));
  }
  PG_RETURN_POINTER(arg0);
}

Datum anon_count_extract(PG_FUNCTION_ARGS) {
  return int_extract<DpCount>(fcinfo);
}


/*
 * ANON_SUM functions.
 */


Datum anon_sum_accum_double(PG_FUNCTION_ARGS) {
  return bounded_accum<DpSum>(fcinfo, false, false);
}

Datum anon_sum_with_bounds_accum_double(PG_FUNCTION_ARGS) {
  return bounded_accum<DpSum>(fcinfo, true, false);
}

Datum anon_sum_accum_int(PG_FUNCTION_ARGS) {
  return bounded_accum<DpSum>(fcinfo, false, true);
}

Datum anon_sum_with_bounds_accum_int(PG_FUNCTION_ARGS) {
  return bounded_accum<DpSum>(fcinfo, true, true);
}

Datum anon_sum_extract_double(PG_FUNCTION_ARGS) {
  return double_extract<DpSum>(fcinfo);
}

Datum anon_sum_extract_int(PG_FUNCTION_ARGS) {
  return int_extract<DpSum>(fcinfo);
}


/*
 * ANON_AVG functions.
 */


Datum anon_avg_accum(PG_FUNCTION_ARGS) {
  return bounded_accum<DpMean>(fcinfo, false, false);
}

Datum anon_avg_with_bounds_accum(PG_FUNCTION_ARGS) {
  return bounded_accum<DpMean>(fcinfo, true, false);
}

Datum anon_avg_extract(PG_FUNCTION_ARGS) {
  return double_extract<DpMean>(fcinfo);
}


/*
 * ANON_VAR functions.
 */


Datum anon_var_accum(PG_FUNCTION_ARGS) {
  return bounded_accum<DpVariance>(fcinfo, false, false);
}

Datum anon_var_with_bounds_accum(PG_FUNCTION_ARGS) {
  return bounded_accum<DpVariance>(fcinfo, true, false);
}

Datum anon_var_extract(PG_FUNCTION_ARGS) {
  return double_extract<DpVariance>(fcinfo);
}


/*
 * ANON_STDDEV functions.
 */


Datum anon_stddev_accum(PG_FUNCTION_ARGS) {
  return bounded_accum<DpStandardDeviation>(fcinfo, false, false);
}

Datum anon_stddev_with_bounds_accum(PG_FUNCTION_ARGS) {
  return bounded_accum<DpStandardDeviation>(fcinfo, true, false);
}

Datum anon_stddev_extract(PG_FUNCTION_ARGS) {
  return double_extract<DpStandardDeviation>(fcinfo);
}



/*
 * ANON_NTILE functions.
 */

Datum ntile_accum(PG_FUNCTION_ARGS, bool is_integral) {
  CHECK_AGG_CONTEXT(fcinfo);
  DpNtile* arg0;

  // Construct algorithm if needed.
  if (PG_ARGISNULL(0)) {
    float8 percentile = PG_GETARG_FLOAT8(2);
    float8 lower = PG_GETARG_FLOAT8(3);
    float8 upper = PG_GETARG_FLOAT8(4);
    std::string err;
    if (PG_NARGS() > 5) {
      float8 epsilon = PG_GETARG_FLOAT8(5);
      arg0 = new DpNtile(&err, percentile, lower, upper,
                         /*default_epsilon=*/false, epsilon);
    } else {
      arg0 = new DpNtile(&err, percentile, lower, upper);
    }
    if (!err.empty()) {
      ereport(ERROR,
              (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("%s", err.c_str())));
    }
  } else {
    arg0 = reinterpret_cast<DpNtile*>(PG_GETARG_POINTER(0));
  }

  // Add the input.
  add_arg_entry(fcinfo, arg0, is_integral);
  PG_RETURN_POINTER(arg0);
}

Datum anon_ntile_accum_double(PG_FUNCTION_ARGS) {
  return ntile_accum(fcinfo, false);
}

Datum anon_ntile_accum_int(PG_FUNCTION_ARGS) {
  return ntile_accum(fcinfo, true);
}

Datum anon_ntile_extract_double(PG_FUNCTION_ARGS) {
  return double_extract<DpNtile>(fcinfo);
}

Datum anon_ntile_extract_int(PG_FUNCTION_ARGS) {
  return int_extract<DpNtile>(fcinfo);
}
