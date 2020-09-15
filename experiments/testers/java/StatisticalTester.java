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

		static int ratioMin;
		static int ratioMax;
		static int numberOfSamples;
		static String fileName;
		static int numberOfVotes = 7;

	public static void main(String[] args) {

// Specify arguments on command line.
		if (args.length>0) {
			try {
				ratioMin = Integer.parseInt(args[0]);
				ratioMax = Integer.parseInt(args[1]);
				numberOfSamples = Integer.parseInt(args[2]);

				if (ratioMin>0 && ratioMin<1 && ratioMax>0 && ratioMax<1 && numberOfSamples>0 && args[3] instanceof String) {
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
			ratioMin = 80;
			ratioMax = 85;
			numberOfSamples = 100;
			fileName = "statistical_tester_results_";
		}

		StatisticalTesterCount stCount = new StatisticalTesterCount();
		stCount.collectData(numberOfVotes,numberOfSamples,ratioMin,ratioMax,fileName+"_count.txt");

		StatisticalTesterSum stSum = new StatisticalTesterSum();
		stSum.collectData(numberOfVotes,numberOfSamples,ratioMin,ratioMax,fileName+"_sum.txt");

		StatisticalTesterMean stMean = new StatisticalTesterMean();
		stMean.collectData(numberOfVotes,numberOfSamples,ratioMin,ratioMax,fileName+"_mean.txt");
}
}