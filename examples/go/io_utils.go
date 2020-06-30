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

package examples

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"
)

func readVisitsFromCSV(inputFile string) ([]Visit, error) {
	csvFile, err := os.Open(inputFile)
	if err != nil {
		return nil, fmt.Errorf("couldn't open the csv file = %q, err = %v", inputFile, err)
	}

	defer csvFile.Close()

	visits := make([]Visit, 0)
	r := csv.NewReader(csvFile)
	skipLine := false
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}

		if err != nil {
			return nil, fmt.Errorf("couldn't read the csv file = %q, err = %v", inputFile, err)
		}

		if len(record) != 5 {
			return nil, fmt.Errorf("the csv file = %q has incorrect format", inputFile)
		}

		// Skip the first line in the csv file which contains the header.
		if !skipLine {
			skipLine = true
			continue
		}

		visitorID, err := toInt64(record[0])
		if err != nil {
			return nil, fmt.Errorf("couldn't read VisitorID = %s as int64 in the csv file = %q, err = %v", record[0], inputFile, err)
		}
		visitTime, err := toTime(record[1])
		if err != nil {
			return nil, fmt.Errorf("couldn't read VisitTime = %s as time (in 3:04PM format) in the csv file = %q, err = %v", record[1], inputFile, err)
		}
		minutesSpent, err := toInt64(record[2])
		if err != nil {
			return nil, fmt.Errorf("couldn't read MinutesSpent = %s as int64 in the csv file = %q, err = %v", record[2], inputFile, err)
		}
		eurosSpent, err := toInt64(record[3])
		if err != nil {
			return nil, fmt.Errorf("couldn't read EurosSpent = %s as int64 in the csv file = %q, err = %v", record[3], inputFile, err)
		}
		day, err := toInt(record[4])
		if err != nil {
			return nil, fmt.Errorf("couldn't read Day = %s as int in the csv file = %s, err = %v", record[4], inputFile, err)
		}

		visits = append(visits,
			Visit{
				VisitorID:    visitorID,
				VisitTime:    visitTime,
				MinutesSpent: minutesSpent,
				EurosSpent:   eurosSpent,
				Day:          day,
			})
	}

	return visits, nil
}

func writeResultsToCSV(results map[int64]int64, outputFile string) error {
	csvFile, err := os.Create(outputFile)
	if err != nil {
		return fmt.Errorf("couldn't open the csv file = %q, err = %v", outputFile, err)
	}

	writer := csv.NewWriter(csvFile)

	for key, value := range results {
		data := []string{toString(key), toString(value)}
		err := writer.Write(data)
		if err != nil {
			return fmt.Errorf(
				"couldn't write to the csv file = %q, err = %v",
				outputFile, combineErrors(err, csvFile.Close()))
		}
	}

	writer.Flush()
	err = writer.Error()

	if err != nil {
		return fmt.Errorf(
			"couldn't write to the csv file = %q, err = %v",
			outputFile, combineErrors(err, csvFile.Close()))
	}

	err = csvFile.Close()
	if err != nil {
		return fmt.Errorf("couldn't close the csv file = %q, err = %v", outputFile, err)
	}

	return nil
}

func toString(n int64) string {
	return strconv.FormatInt(n, 10)
}

func toInt64(str string) (int64, error) {
	return strconv.ParseInt(str, 10, 64)
}

func toInt(str string) (int, error) {
	res, err := strconv.ParseInt(str, 10, 32)
	if err == nil {
		return int(res), err
	}
	return 0, err
}

func toTime(str string) (time.Time, error) {
	return time.Parse(time.Kitchen, str)
}

func combineErrors(errors ...error) string {
	var nonNilErrors []error
	for _, err := range errors {
		if err != nil {
			nonNilErrors = append(nonNilErrors, err)
		}
	}
	return fmt.Sprintf("%+v", nonNilErrors)
}
