package com.yahoo.labs.yamall.spark;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.Learner;
import com.yahoo.labs.yamall.parser.VWParser;
import com.yahoo.labs.yamall.spark.helper.Evaluate;
import com.yahoo.labs.yamall.spark.helper.FileWriterToHDFS;
import com.yahoo.labs.yamall.spark.helper.ModelSerializationToHDFS;
import com.yahoo.labs.yamall.spark.helper.SparkLearnerFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.zip.GZIPInputStream;

/**
 * Created by busafekete on 8/28/17.
 */
public class SparkSingleCoreTrain {
    protected static String inputDir;
    protected static String inputDirTest;
    protected static String outputDir;

    protected static String logFile = "";

    protected static int bitsHash = 22;
    protected static StringBuilder strb = new StringBuilder("");

    protected static String method = null;
    protected static Learner learner = null;
    protected static VWParser vwparser = null;

    protected static double minPrediction = -50.0;
    protected static double maxPrediction = 50.0;

    protected static double cumLoss = 0.0;
    protected static long numSamples = 0;
    protected static int evalPeriod = 5000000;
    protected static int evalNumInstances = 5000000;
    protected static boolean saveModelFlag = false;

    protected static ArrayList<Path>  featureFilePaths = null;
    protected static ArrayList<Path>  featureFilePathsTest = null;

    public static void init(SparkConf sparkConf) throws IOException {
        outputDir = sparkConf.get("spark.myapp.outdir");
        inputDir = sparkConf.get("spark.myapp.input");
        inputDirTest = sparkConf.get("spark.myapp.test");

        method = sparkConf.get("spark.myapp.method");
        evalPeriod = Integer.parseInt(sparkConf.get("spark.myapp.evalperiod", "5000000"));
        saveModelFlag = Boolean.parseBoolean(sparkConf.get("spark.myapp.save_model", "false"));
        evalNumInstances = Integer.parseInt(sparkConf.get("spark.myapp.testsize", "1000000"));
        // create learner
        learner = SparkLearnerFactory.getLearnerForSpark(sparkConf);
        vwparser = new VWParser(bitsHash, null, false);
        logFile = outputDir + "/log.txt";



        // Get a list of all the files in the inputPath directory. We will read these files one at a time
        //the second boolean parameter here sets the recursion to true

        FileSystem hdfs = FileSystem.get(new Configuration());

        featureFilePaths = new ArrayList<>();
        RemoteIterator<LocatedFileStatus> fileStatusListIterator = hdfs.listFiles(
                new Path(inputDir), true);

        while (fileStatusListIterator.hasNext()) {
            LocatedFileStatus fileStatus = fileStatusListIterator.next();
            String fileName = fileStatus.getPath().getName();
            if (fileName.contains(".gz") || fileName.contains(".txt"))
                featureFilePaths.add(fileStatus.getPath());
        }

        // pick some file for testing
        Collections.shuffle(featureFilePaths);

        // test files
        featureFilePathsTest = new ArrayList<>();
        fileStatusListIterator = hdfs.listFiles(
                new Path(inputDirTest), true);

        while (fileStatusListIterator.hasNext()) {
            LocatedFileStatus fileStatus = fileStatusListIterator.next();
            String fileName = fileStatus.getPath().getName();
            if (fileName.contains(".gz") || fileName.contains(".txt"))
                featureFilePathsTest.add(fileStatus.getPath());
        }

        strb.append("--- Input: " + inputDir + "\n");
        strb.append("--- Output: " + outputDir + "\n");
        strb.append("--- Number of train files: " + featureFilePaths.size() + "\n");
        strb.append("--- Number of test files: " + featureFilePathsTest.size() + "\n");
        strb.append("--- Eval period: " + evalPeriod + "\n");
        strb.append("--- Num of eval instaces: " + evalNumInstances + "\n");

        System.out.println(strb.toString());
        saveLog();
    }

    protected static void saveLog() throws IOException {
        FileWriterToHDFS.writeToHDFS(logFile, strb.toString());
    }


    public static void train() throws IOException {
        SparkConf sparkConf = new SparkConf().setAppName("spark yamall (single core training)");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
        FileSystem hdfs = FileSystem.get(sparkContext.hadoopConfiguration());

        init(sparkConf);


        long clusterStartTime = System.currentTimeMillis();

        for (Path featureFile : featureFilePaths) {
            System.out.println("----- Starting file " + featureFile + " Samp. num: " + numSamples);
            strb.append("----- Starting file " + featureFile + " Samp. num: " + numSamples + "\n");

            BufferedReader br = null;
            if (featureFile.getName().contains(".gz"))
                br = new BufferedReader(new InputStreamReader(new GZIPInputStream(hdfs.open(featureFile))));
            else
                br = new BufferedReader(new InputStreamReader(hdfs.open(featureFile)));

            for (; ; ) { // forever
                String strLine = br.readLine();

                Instance sample;


                if (strLine != null) {
                    sample = vwparser.parse(strLine);
                } else
                    break;

                double score = learner.update(sample);
                score = Math.min(Math.max(score, minPrediction), maxPrediction);

                cumLoss += learner.getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();

                numSamples++;

                if (numSamples % evalPeriod == 0) {
                    double trainLoss = cumLoss / (double) numSamples;
                    //double testLoss = eval(hdfs, vwparser);
                    double testLoss = Evaluate.getLoss(sparkContext,inputDirTest,learner, bitsHash);
                    long clusteringRuntime = System.currentTimeMillis() - clusterStartTime;
                    double elapsedTime = clusteringRuntime / 1000.0;
                    double elapsedTimeInhours = elapsedTime / 3600.0;

                    String line = String.format("%d %f %f %f\n", numSamples, trainLoss, testLoss, elapsedTimeInhours);
                    strb.append(line);
                    System.out.print(method + " " + line);

                    saveLog();
                }

            }
        }


        double trainLoss = cumLoss / (double) numSamples;
        //double testLoss = eval(hdfs, vwparser);
        double testLoss = Evaluate.getLoss(sparkContext,inputDirTest,learner, bitsHash);
        long clusteringRuntime = System.currentTimeMillis() - clusterStartTime;
        double elapsedTime = clusteringRuntime / 1000.0;
        double elapsedTimeInhours = elapsedTime / 3600.0;

        String line = String.format("%d %f %f %f\n", numSamples, trainLoss, testLoss, elapsedTimeInhours);
        strb.append(line);
        System.out.print(method + " " + line);

        saveLog();


        if (saveModelFlag) {
            String modelFile = "/model.bin";
            ModelSerializationToHDFS.saveModel(outputDir, modelFile, learner);
        }

    }

//    public static double eval(FileSystem hdfs, VWParser vwparser) throws IOException {
//        int numSamples = 0;
//        double score;
//        double cumLoss = 0.0;
//
//        Collections.shuffle(featureFilePathsTest);
//
//        for (Path featureFile : featureFilePathsTest) {
//            System.out.println("----- Starting test file " + featureFile + " Samp. num: " + numSamples);
//            strb.append("----- Starting test file " + featureFile + " Samp. num: " + numSamples + "\n");
//
//            BufferedReader br = null;
//            if (featureFile.getName().contains(".gz"))
//                br = new BufferedReader(new InputStreamReader(new GZIPInputStream(hdfs.open(featureFile))));
//            else
//                br = new BufferedReader(new InputStreamReader(hdfs.open(featureFile)));
//
//            for (; ; ) { // forever
//                String strLine = br.readLine();
//
//                Instance sample;
//
//                if (strLine != null) {
//                    sample = vwparser.parse(strLine);
//                } else
//                    break;
//
//
//                score = learner.predict(sample);
//                score = Math.min(Math.max(score, minPrediction), maxPrediction);
//
//                cumLoss += learner.getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
//
//                numSamples++;
//
//                if (numSamples>evalNumInstances) break;
//            }
//            if (numSamples>evalNumInstances) break;
//        }
//
//        return cumLoss / (double) numSamples;
//    }


    public static void main( String[] args ) throws IOException {
        train();
    }

}
