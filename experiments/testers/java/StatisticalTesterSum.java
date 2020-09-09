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

/** Runs the Statistical Tester on differentially private sums with
insufficient noise. Records the results in an output text file.
*/
public class StatisticalTesterSum {

	private static final String homedir = "boundedsumsamples/";

// Run each test case according to parameters specified by SumDpTest.java.
// If any one test fails to satisfy DP, the algorithm is considered not DP.
	private static int getOverallOutcome(int numberOfSamples, String ratio,
		int numberOfVotes) {

		int counter = 0;
		double smallEpsilon = 0.1;
		double mediumEpsilon = Math.log(3);
		double largeEpsilon = 2*Math.log(3);
		final int delta = 0;

		int testCase1 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario1/",numberOfSamples,
			mediumEpsilon,delta,0.02,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase2 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario2/",numberOfSamples,
			mediumEpsilon,delta,0.04,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase3 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario3/",numberOfSamples,
			mediumEpsilon,delta,0.04,numberOfVotes); 
		counter++;
		System.out.println(counter);

		int testCase4 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario4/",numberOfSamples,
			smallEpsilon,delta,0.0135,numberOfVotes); 
		counter++;
		System.out.println(counter);

		int testCase5 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario5/",numberOfSamples,
			largeEpsilon,delta,0.0135,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase6 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario6/",numberOfSamples,
			mediumEpsilon,delta,0.02,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase7 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario7/",numberOfSamples,
			mediumEpsilon,delta,0.04,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase8 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario8/",numberOfSamples,
			smallEpsilon,delta,0.0135,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase9 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario9/",numberOfSamples,
			largeEpsilon,delta,0.0135,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase10 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario10/",numberOfSamples,
			mediumEpsilon,delta,0.04,numberOfVotes); 
		counter++;
		System.out.println(counter);

		int testCase11 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario11/",numberOfSamples,
			mediumEpsilon,delta,0.04,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase12 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario12/",numberOfSamples,
			smallEpsilon,delta,0.023,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase13 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario13/",numberOfSamples,
			largeEpsilon,delta,0.023,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase14 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario14/",numberOfSamples,
			mediumEpsilon,delta,0.04,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase15 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario15/",numberOfSamples,
			mediumEpsilon,delta,0.04,numberOfVotes); 
		counter++;
		System.out.println(counter);

		int testCase16 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario16/",numberOfSamples,
				smallEpsilon,delta,0.023,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase17 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario17/",numberOfSamples,
			largeEpsilon,delta,0.023,numberOfVotes);
		counter++;
		System.out.println(counter);

		int allCases = testCase1+testCase2+testCase3+testCase4+testCase5+testCase6
		+testCase7+testCase8+testCase9+testCase10+testCase11+testCase12+testCase13
		+testCase14+testCase15+testCase16+testCase17;
		System.out.println("The algorithm incorrectly passed "+Integer.toString(allCases)+" out of 17 tests.");

		if (allCases < 17) {
			System.out.println("The algorithm fails according to this Tester.");
			return 0;
		}
		else { 
			System.out.println("The algorithm passes according to this Tester.");
			return 1;
		}
	}

	protected static void collectData(int numberOfVotes, int numberOfSamples, int ratioMin,
		int ratioMax, String fileName) {
		String testName = "insufficient_noise";
		String algorithmType = "bounded_sum";
		String expected = "0";
		String numDatasets = "17";
		String numSamples = Integer.toString(numberOfSamples);
		String time = "null";
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss");

    PrintWriter pw = null;
    try {
      pw = new PrintWriter(new File("../results/"+fileName));
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }

    StringBuilder builder = new StringBuilder();
    Timestamp timestamp = new Timestamp(System.currentTimeMillis());
    builder.append("Results run on: "+timestamp + "\n");
    String columnNamesList = "test_name,algorithm,expected,actual,ratio,num_datasets,num_samples,time(sec)";
    builder.append(columnNamesList +"\n");

		for (int i = ratioMin; i <= ratioMax; i++) {
			String r = Integer.toString(i);
			String Outcome = Integer.toString(getOverallOutcome(numberOfSamples,r,numberOfVotes));
	    builder.append(testName+","+algorithmType+","+expected+","+Outcome+","+r+","+numDatasets+","+numSamples+","+time);
	    builder.append('\n');
    }
    pw.write(builder.toString());
 		pw.close();
	}
}
