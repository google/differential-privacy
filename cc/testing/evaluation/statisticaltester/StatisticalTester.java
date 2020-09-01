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

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.privacy.differentialprivacy.testing.StatisticalTestsUtil;
import java.io.*;
import java.lang.*;
import java.util.*;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.Date;

public class StatisticalTester {

	public static void main(String[] args) {

		int ratio_min;
		int ratio_max;
		int numberOfSamples;
		String filename;
// Specify arguments on command line.
		if (args.length>0) {
			try {
				ratio_min = Integer.parseInt(args[0]);
				ratio_max = Integer.parseInt(args[1]);
				numberOfSamples = Integer.parseInt(args[2]);

				if (ratio_min>0 && ratio_min<1 && ratio_max>0 && ratio_max<1 && numberOfSamples>0 && args[3] instanceof String) {
					StatisticalTesterCount st_count = new StatisticalTesterCount();
					st_count.collectData(ratio_min,ratio_max,numberOfSamples,args[3]);

					StatisticalTesterMean st_mean = new StatisticalTesterMean();
					st_mean.collectData(ratio_min,ratio_max,numberOfSamples,args[4]);

					StatisticalTesterSum st_sum = new StatisticalTesterSum();
					st_sum.collectData(ratio_min,ratio_max,numberOfSamples,args[5]);
				}
				else {
					System.out.println("An invalid parameter was specified. Try again, please!");
				}
			}
			catch (NumberFormatException e) {
				System.err.println("Argument" + args[0]+", "+args[1]+", "+args[2]+" must be integers.");
				System.exit(1);
			}
		}
		else {
// Use default arguments.
			ratio_min = 80;
			ratio_max = 85;
			numberOfSamples = 100;
			filename = "statistical_tester_results_";

			StatisticalTesterCount st_count = new StatisticalTesterCount();
			st_count.collectData(numberOfSamples,ratio_min,ratio_max,filename+"_count.txt");

			StatisticalTesterSum st_sum = new StatisticalTesterSum();
			st_sum.collectData(numberOfSamples,ratio_min,ratio_max,filename+"_sum.txt");

			StatisticalTesterMean st_mean = new StatisticalTesterMean();
			st_mean.collectData(numberOfSamples,ratio_min,ratio_max,filename+"_mean.txt");

		}
	}
}