package com.yahoo.labs.yamall.hadoop.sparkcore;

import com.yahoo.labs.yamall.ml.Learner;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaStreamingContext;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;

/**
 * Created by busafekete on 7/24/17.
 */
public class FileStreamingTrain implements Serializable {
    protected String inputDir;
    protected String outputDir;

    protected String logFile = "";

    public static int bitsHash = 22;
    public long N = 0;
    StringBuilder strb = new StringBuilder("");

    protected int mainloops = 10;
    protected int batchsize = 10000;
    protected int testsize = 1000;

    protected String method = null;
    protected Learner learner = null;

    public static double minPrediction = -50.0;
    public static double maxPrediction = 50.0;

    protected double cumLoss = 0.0;
    protected long numSamples = 0;


    public FileStreamingTrain() {
        SparkConf sparkConf = new SparkConf().setAppName("spark yamall (training)");

        this.outputDir = sparkConf.get("spark.myapp.outdir");
        this.inputDir = sparkConf.get("spark.myapp.input");

        this.mainloops = Integer.parseInt(sparkConf.get("spark.myapp.mainloops", "10"));
        this.batchsize = Integer.parseInt(sparkConf.get("spark.myapp.batchsize", "10000"));
        this.testsize = Integer.parseInt(sparkConf.get("spark.myapp.testsize", "1000"));

        this.logFile = this.outputDir + "log.txt";

        strb.append("Input: " + this.inputDir + "\n");
        strb.append("Output: " + this.outputDir + "\n");
        strb.append("main loops: " + this.mainloops + "\n");
        strb.append("batch size: " + this.batchsize + "\n");
        strb.append("test size: " + this.testsize + "\n");
    }

    public void train() throws IOException {
        SparkConf sparkConf = new SparkConf().setAppName("spark yamall (training)");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
        JavaRDD<String> input = sparkContext.textFile(inputDir);

        JavaStreamingContext ssc = new JavaStreamingContext(sparkContext, Durations.seconds(1));
        ssc.textFileStream(inputDir);

        long clusterStartTime = System.currentTimeMillis();
        FileSystem hdfs = FileSystem.get(sparkContext.hadoopConfiguration());

        // Get a list of all the files in the inputPath directory. We will read these files one at a time
        Path[] featureFilePaths = FileUtil.stat2Paths(hdfs.listStatus(new Path(inputDir)));


        // Loop through the feature files
        for (Path featureFile : featureFilePaths) {
            System.out.println("----- Starting file " + featureFile + " -----");
            //BinaryInputStream inputStream = new BinaryInputStream(new BZip2CompressorInputStream(hdfs.open(featureFile)));
            BufferedReader br = new BufferedReader(new InputStreamReader(hdfs.open(featureFile)));
            for(;;) { // forever
                String line = br.readLine();

            }
        }
        long clusteringRuntime = System.currentTimeMillis() - clusterStartTime;




    }
}