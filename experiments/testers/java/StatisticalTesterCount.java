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

/** Runs the Statistical Tester on differentially private counts with
insufficient noise. Records the results in an output text file.
*/
public class StatisticalTesterCount {

	private static final String homedir = "countsamples/";

// Run each test case according to parameters specified by CountDpTest.java.
// If any one test fails to satisfy DP, the algorithm is considered not DP.
	private static int getOverallOutcome(int numberOfSamples, String ratio,
		int numberOfVotes) {

		int counter = 0;
		double smallEpsilon = 0.01;
		double mediumEpsilon = Math.log(3);
		double largeEpsilon = 2*Math.log(3);
		final int delta = 0;

		int testCase1 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario1/",numberOfSamples,
			mediumEpsilon,delta,0.0025,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase2 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario2/",numberOfSamples,
			mediumEpsilon,delta,0.0035,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase3 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario3/",numberOfSamples,
			mediumEpsilon,delta,0.02,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase4 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario4/",numberOfSamples,
			smallEpsilon,delta,0.0131,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase5 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario5/",numberOfSamples,
			largeEpsilon,delta,0.0035,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase6 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario6/",numberOfSamples,
			mediumEpsilon,delta,0.0025,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase7 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario7/",numberOfSamples,
			mediumEpsilon,delta,0.0035,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase8 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario8/",numberOfSamples,
			mediumEpsilon,delta,0.0025,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase9 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario9/",numberOfSamples,
			mediumEpsilon,delta,0.02,numberOfVotes);
		counter++;
		System.out.println(counter);

		int testCase10 = StatisticalUtils.getMajorityVote(homedir+"R"+ratio+"/Scenario10/",numberOfSamples,
			smallEpsilon,delta,0.0131,numberOfVotes);
		counter++;
		System.out.println(counter);

		int allCases = testCase1+testCase2+testCase3+testCase4+testCase5+testCase6
			+testCase7+testCase8+testCase9+testCase10;
		System.out.println("The algorithm incorrectly passed "
			+Integer.toString(allCases)+" out of 10 tests.");

		if (allCases < 10) {
			System.out.println("The algorithm fails according to this Tester.");
			return 0;
		}

		else { 
			System.out.println("The algorithm passes according to this Tester.");
			return 1;
		}
	}

	protected static void collectData(int numberOfVotes,
		int numberOfSamples, int ratioMin, int ratioMax,
		String fileName) {
		String testName = "insufficient_noise";
		String algorithmType = "count";
		String expected = "0";
		String numDatasets = "10";
		String numSamples = Integer.toString(numberOfSamples);
//		numberOfVotes = 7;
// TODO: Add time elapsed to be consistent with Stochastic Tester output file.
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
			System.out.println(i);
			String r = Integer.toString(i);
			String Outcome = Integer.toString(getOverallOutcome(numberOfSamples,r,numberOfVotes));
	    builder.append(testName+","+algorithmType+","+expected+","+Outcome+","+r+","
	    	+String.valueOf(numDatasets)+","+String.valueOf(numSamples)+","+time);
	    builder.append('\n');
    }
    pw.write(builder.toString());
 		pw.close();
	}
}
