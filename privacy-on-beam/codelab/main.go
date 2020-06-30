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
// bazel run codelab/codelab -- --example="count" --output_text_name=$(pwd)/codelab/count.txt --output_img_name=$(pwd)/codelab/count.png
package main

import (
	"context"
	"fmt"
	"path"
	"reflect"
	"strings"

	"flag"
	log "github.com/golang/glog"
	"github.com/google/differential-privacy/privacy-on-beam/codelab"
	"github.com/apache/beam/sdks/go/pkg/beam"

	// The following import is required for accessing local files.
	_ "github.com/apache/beam/sdks/go/pkg/beam/io/filesystem/local"

	"github.com/apache/beam/sdks/go/pkg/beam/runners/direct"
)

func init() {
	beam.RegisterType(reflect.TypeOf((*normalizeOutputCombineFn)(nil)))
	beam.RegisterType(reflect.TypeOf(outputAccumulator{}))
	beam.RegisterFunction(convertToPairFn)
}

var (
	// Set this option to choose which example to run.
	example = flag.String("example", "", "Privacy on Beam example to run, enter 'count', 'sum', 'mean'.")

	// By default, this reads from day_data.csv.
	inputName = flag.String("input_name", "codelab/day_data.csv", "File to read.")

	// By default, this writes to 'example_name.txt'.
	outputTextName = flag.String("output_text_name", "", "Output file. Set to '/tmp/{{example_name}}.txt' by default.")

	// By default, this writes to 'example_name.png'.
	outputImgName = flag.String("output_img_name", "", "Output file. Set to '/tmp/{{example_name}}.png' by default.")
)

func main() {
	flag.Parse()

	// beam.Init() is an initialization hook that must be called on startup. On
	// distributed runners, it is used to intercept control.
	beam.Init()

	// Flag validation.
	switch *example {
	case count, mean, sum:
	case "":
		log.Exit("No example specified.")
	default:
		log.Exitf("Unknown example (%s) specified, please use one of 'count', 'sum', 'mean'", *example)
	}
	if *inputName == "" {
		log.Exit("No input file specified.")
	}
	if *outputTextName == "" {
		*outputTextName = "/tmp/" + *example + ".txt"
	}
	if *outputImgName == "" {
		*outputImgName = "/tmp/" + *example + ".png"
	}

	// DP output file names.
	outputTextNameDP := strings.ReplaceAll(*outputTextName, path.Ext(*outputTextName), "_dp"+path.Ext(*outputTextName))
	outputImgNameDP := strings.ReplaceAll(*outputImgName, path.Ext(*outputImgName), "_dp"+path.Ext(*outputImgName))

	// Create a pipeline.
	p := beam.NewPipeline()
	s := p.Root()

	// Read and parse the input.
	visits := readInput(s, *inputName)

	// Run the example pipeline.
	rawOutput := runRawExample(s, visits, *example)
	dpOutput := runDPExample(s, visits, *example)

	// Write the text output to file.
	log.Info("Writing text output.")
	writeOutput(s, rawOutput, *outputTextName)
	writeOutput(s, dpOutput, outputTextNameDP)

	// Execute pipeline.
	err := direct.Execute(context.Background(), p)
	if err != nil {
		log.Exitf("Execution of pipeline failed: %v", err)
	}

	// Read the text output from file.
	hourToValue, err := readOutput(*outputTextName)
	if err != nil {
		log.Exitf("Reading output text file (%s) to plot bar charts failed: %v", *outputTextName, err)
	}
	dpHourToValue, err := readOutput(outputTextNameDP)
	if err != nil {
		log.Exitf("Reading output text file (%s) to plot bar charts failed: %v", outputTextNameDP, err)
	}

	// Draw the bar charts.
	if err = drawPlot(hourToValue, dpHourToValue, *example, *outputImgName, outputImgNameDP); err != nil {
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
	default:
		log.Exitf("Unknown example %q specified, please use one of 'count', 'sum', 'mean'", example)
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
	default:
		log.Exitf("Unknown example %q specified, please use one of 'count', 'sum', 'mean'", example)
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
