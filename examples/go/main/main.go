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

// This is a command line utility which helps to run different examples of how dp library can be used.
// Usage example:
// (From the examples/go/ directory)
// Linux: bazel run main/examples -- --scenario=CountVisitsPerHour --input_file=$(pwd)/data/day_data.csv --non_private_output_file=$(pwd)/non_private.csv --private_output_file=$(pwd)/private.csv
// Windows: bazel run main/examples -- --scenario=CountVisitsPerHour --input_file=%CD%/data/day_data.csv --non_private_output_file=%CD%/non_private.csv --private_output_file=%CD%/private.csv
// (Or alternatively, from the examples/go/main/ directory)
// Linux: bazel run examples -- --scenario=CountVisitsPerHour --input_file=$(pwd)/../data/day_data.csv --non_private_output_file=$(pwd)/non_private.csv --private_output_file=$(pwd)/private.csv
// Windows: bazel run examples -- --scenario=CountVisitsPerHour --input_file=%CD%/../data/day_data.csv --non_private_output_file=%CD%/non_private.csv --private_output_file=%CD%/private.csv
// If instead you'd like to build and run with native "go" command, you can run the following:
// (From the examples/go/ directory)
// go run --mod=mod ./main --scenario=CountVisitsPerHour --input_file=data/day_data.csv --non_private_output_file=non_private.csv --private_output_file=private.csv
// (Or alternatively, from the examples/go/main/ directory)
// go run --mod=mod . --scenario=CountVisitsPerHour --input_file=../data/day_data.csv --non_private_output_file=non_private.csv --private_output_file=private.csv
package main

import (
	"flag"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/examples/go"
)

var (
	scenario = flag.String("scenario", "", "Scenario ID:\n"+
		"CountVisitsPerHour - counts of visits per hour.\n"+
		"CountVisitsPerDay - counts of visits per week day.\n"+
		"SumRevenuePerDay - total revenue per week day.\n"+
		"CountVisitsPerDuration - counts of visits per certain duration.")
	inputFile                   = flag.String("input_file", "", "Input csv file name with raw data.")
	nonPrivateResultsOutputFile = flag.String("non_private_output_file", "", "Output csv file name for non private results.")
	privateResultsOutputFile    = flag.String("private_output_file", "", "Output csv file name for private results.")
)

const (
	countVisitsPerHourScenarioID  = "CountVisitsPerHour"
	countVisitsPerDayScenarioID   = "CountVisitsPerDay"
	sumRevenuePerDayScenarioID    = "SumRevenuePerDay"
	countVisitsPerCertainDuration = "CountVisitsPerDuration"
)

func main() {
	flag.Parse()

	log.Infof("The example was run with arguments: scenario = %q,"+
		" inputFile = %q, nonPrivateResultsOutputFile = %q, privateResultsOutputFile = %q",
		*scenario,
		*inputFile,
		*nonPrivateResultsOutputFile,
		*privateResultsOutputFile,
	)

	if *scenario == "" {
		log.Exit("No scenario was chosen")
	}

	if *inputFile == "" {
		log.Exit("No input file was chosen")
	}

	if *nonPrivateResultsOutputFile == "" {
		log.Exit("No output file for non-private results was chosen")
	}

	if *privateResultsOutputFile == "" {
		log.Exit("No output file for private results was chosen")
	}

	var err error
	var sc examples.Scenario
	switch id := *scenario; id {
	case countVisitsPerHourScenarioID:
		sc = &examples.CountVisitsPerHourScenario{}
	case countVisitsPerDayScenarioID:
		sc = &examples.CountVisitsPerDayScenario{}
	case sumRevenuePerDayScenarioID:
		sc = &examples.SumRevenuePerDayScenario{}
	case countVisitsPerCertainDuration:
		sc = &examples.CountVisitsPerCertainDurationScenario{}
	default:
		log.Exitf("There is no scenario with id = %s", id)
	}

	err = examples.RunScenario(sc, *inputFile, *nonPrivateResultsOutputFile, *privateResultsOutputFile)
	if err != nil {
		log.Exitf("Couldn't execute example, err = %v", err)
	}

	log.Infof("Successfully finished executing the example")
}
