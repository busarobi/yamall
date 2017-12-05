package com.yahoo.labs.yamall.spark.core;

import com.yahoo.labs.yamall.spark.helper.Evaluate;
import com.yahoo.labs.yamall.spark.helper.FileWriterToHDFS;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.testng.annotations.Test;

/**
 * Created by busafekete on 9/16/17.
 */
public class PerCoordinateSVRGSparkTest {
    protected static String inputDir = "/Users/busafekete/work/DistOpt/Clkb_data/";
    protected static String inputDirTest = "/Users/busafekete/work/DistOpt/Clkb_data_test/";
    protected static String outDir = "/Users/busafekete/work/DistOpt/tmp/";
    protected static String logFile = outDir + "log.txt";
    protected static int bits = 20;
    @Test
    public void testTrain() throws Exception {
        SparkConf sparkConf = new SparkConf().setAppName("spark yamall (parallel training)");

        sparkConf.set("spark.myapp.outdir", outDir);
        sparkConf.set("spark.myapp.iter", "10");
        sparkConf.set("executor-memory", "14G");
        sparkConf.set("driver-memory", "14G");
        sparkConf.set("spark.driver.maxResultSize", "14G" );
        sparkConf.set("spark.myapp.batchsize", "500000" );

        JavaSparkContext sparkContext = new JavaSparkContext("local[*]", "Test");

        JavaRDD<String> input = null;
        input = sparkContext.textFile(inputDir);
        StringBuilder strb = new StringBuilder("");

        PerCoordinateSVRGSpark learner = new PerCoordinateSVRGSpark(sparkConf,strb,bits);
        learner.train(input);


        double testLoss = Evaluate.getLoss(sparkContext,inputDirTest,learner, bits);
        String line = String.format("Test loss: %f\n", testLoss);
        strb.append(line);

        Evaluate.computeResult(strb,sparkContext,inputDirTest,learner, bits);


        Evaluate.computeResult(strb,sparkContext,inputDirTest,learner,bits);
        FileWriterToHDFS.writeToHDFS(logFile, strb.toString());

    }

}