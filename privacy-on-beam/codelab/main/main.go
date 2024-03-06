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

// package main runs the Privacy on Beam codelab.
// Example command to run:
// (From the codelab/ directory)
// Linux: bazel run main/codelab -- --example=count --input_file=$(pwd)/main/day_data.csv --output_stats_file=$(pwd)/stats.csv --output_chart_file=$(pwd)/chart.png
// Windows: bazel run main/codelab -- --example=count --input_file=%CD%/main/day_data.csv --output_stats_file=%CD%/stats.csv --output_chart_file=%CD%/chart.png
// (Or alternatively, from the codelab/main/ directory)
// Linux: bazel run codelab -- --example=count --input_file=$(pwd)/day_data.csv --output_stats_file=$(pwd)/stats.csv --output_chart_file=$(pwd)/chart.png
// Windows: bazel run codelab -- --example=count --input_file=%CD%/day_data.csv --output_stats_file=%CD%/stats.csv --output_chart_file=%CD%/chart.png
// If instead you'd like to build and run with native "go" command, you can run the following:
// (From the codelab/ directory)
// go run --mod=mod ./main  --example=count --input_file=main/day_data.csv --output_stats_file=stats.csv --output_chart_file=chart.png
// (From the codelab/main directory)
// go run --mod=mod . --example=count --input_file=day_data.csv --output_stats_file=stats.csv --output_chart_file=chart.png
// Replace 'example=count' with 'example=sum', 'example=mean' or 'example=public_partitions' to run other examples.
package main

import (
	"context"
	"fmt"
	"path"
	"strings"

	"flag"
	log "github.com/golang/glog"
	"github.com/google/differential-privacy/privacy-on-beam/v3/codelab"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/register"

	// The following import is required for accessing local files.
	_ "github.com/apache/beam/sdks/v2/go/pkg/beam/io/filesystem/local"

	"github.com/apache/beam/sdks/v2/go/pkg/beam/runners/direct"
)

func init() {
	register.Combiner3[outputAccumulator, pair, string](&normalizeOutputCombineFn{})
	register.Function2x2[int, beam.V, pair, error](convertToPairFn)
}

var (
	example = flag.String("example", "", "Example ID:\n"+
		"count - counts of visits per hour.\n"+
		"sum - total revenue per hour.\n"+
		"mean - average visit time per hour.\n"+
		"public_partitions - count of visits per hour with public partitions.")
	inputFile       = flag.String("input_file", "", "Input csv file name with raw data.")
	outputStatsFile = flag.String("output_stats_file", "", "Output csv file name for stats results.")
	outputChartFile = flag.String("output_chart_file", "", "Output png file name for chart with stats.")
)

func main() {
	flag.Parse()

	// beam.Init() is an initialization hook that must be called on startup. On
	// distributed runners, it is used to intercept control.
	beam.Init()

	// Flag validation.
	switch *example {
	case count, mean, sum, publicPartitions:
	case "":
		log.Exit("No example specified.")
	default:
		log.Exitf("Unknown example (%s) specified, please use one of 'count', 'sum', 'mean', 'public_partitions'", *example)
	}
	if *inputFile == "" {
		log.Exit("No input file specified.")
	}
	if *outputStatsFile == "" {
		log.Exit("No output stats file specified.")
	}
	if *outputChartFile == "" {
		log.Exit("No output chart file specified.")
	}

	// DP output file names.
	outputStatsFileDP := strings.ReplaceAll(*outputStatsFile, path.Ext(*outputStatsFile), "_dp"+path.Ext(*outputStatsFile))
	outputChartFileDP := strings.ReplaceAll(*outputChartFile, path.Ext(*outputChartFile), "_dp"+path.Ext(*outputChartFile))

	// Create a pipeline.
	p := beam.NewPipeline()
	s := p.Root()

	// Read and parse the input.
	visits := readInput(s, *inputFile)

	// Run the example pipeline.
	rawOutput := runRawExample(s, visits, *example)
	dpOutput := runDPExample(s, visits, *example)

	// Write the text output to file.
	log.Info("Writing text output.")
	writeOutput(s, rawOutput, *outputStatsFile)
	writeOutput(s, dpOutput, outputStatsFileDP)

	// Execute pipeline.
	_, err := direct.Execute(context.Background(), p)
	if err != nil {
		log.Exitf("Execution of pipeline failed: %v", err)
	}

	// Read the text output from file.
	hourToValue, err := readOutput(*outputStatsFile)
	if err != nil {
		log.Exitf("Reading output text file (%s) to plot bar charts failed: %v", *outputStatsFile, err)
	}
	dpHourToValue, err := readOutput(outputStatsFileDP)
	if err != nil {
		log.Exitf("Reading output text file (%s) to plot bar charts failed: %v", outputStatsFileDP, err)
	}

	// Draw the bar charts.
	if err = drawPlot(hourToValue, dpHourToValue, *example, *outputChartFile, outputChartFileDP); err != nil {
		log.Exitf("Drawing bar chart failed: %v", err)
	}

}

func runRawExample(s beam.Scope, col beam.PCollection, example string) beam.PCollection {
	switch example {
	case count:
		return codelab.CountVisitsPerHour(s, col)
	case mean:
		return codelab.MeanTimeSpent(s, col)
	case sum:
		return codelab.RevenuePerHour(s, col)
	case publicPartitions:
		return codelab.CountVisitsPerHour(s, col)
	default:
		log.Exitf("Unknown example %q specified, please use one of 'count', 'sum', 'mean', 'public_partitions'", example)
		return beam.PCollection{}
	}
}

func runDPExample(s beam.Scope, col beam.PCollection, example string) beam.PCollection {
	switch example {
	case count:
		return codelab.PrivateCountVisitsPerHour(s, col)
	case mean:
		return codelab.PrivateMeanTimeSpent(s, col)
	case sum:
		return codelab.PrivateRevenuePerHour(s, col)
	case publicPartitions:
		return codelab.PrivateCountVisitsPerHourWithPublicPartitions(s, col)
	default:
		log.Exitf("Unknown example %q specified, please use one of 'count', 'sum', 'mean', 'public_partitions'", example)
		return beam.PCollection{}
	}
}

type pair struct {
	K int
	V float64
}

func convertToPairFn(k int, v beam.V) (pair, error) {
	switch v := v.(type) {
	case int:
		return pair{K: k, V: float64(v)}, nil
	case int64:
		return pair{K: k, V: float64(v)}, nil
	case float64:
		return pair{K: k, V: v}, nil
	default:
		return pair{}, fmt.Errorf("expected int, int64 or float64 for value type, got %v", v)
	}
}

type outputAccumulator struct {
	HourToValue map[int]float64
}

type normalizeOutputCombineFn struct{}

func (fn *normalizeOutputCombineFn) CreateAccumulator() outputAccumulator {
	hourToValue := make(map[int]float64)
	for i := 0; i < 24; i++ {
		hourToValue[i] = 0
	}
	return outputAccumulator{hourToValue}
}

func (fn *normalizeOutputCombineFn) AddInput(a outputAccumulator, p pair) outputAccumulator {
	a.HourToValue[p.K] = p.V
	return a
}

func (fn *normalizeOutputCombineFn) MergeAccumulators(a, b outputAccumulator) outputAccumulator {
	for k, v := range b.HourToValue {
		if v != 0 {
			a.HourToValue[k] = v
		}
	}
	return a
}

func (fn *normalizeOutputCombineFn) ExtractOutput(a outputAccumulator) string {
	var lines []string
	for k, v := range a.HourToValue {
		lines = append(lines, fmt.Sprintf("%d %f", k, v))
	}
	return strings.Join(lines, "\n")
}
