package com.yahoo.labs.yamall.spark;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.Learner;
import com.yahoo.labs.yamall.parser.VWParser;
import com.yahoo.labs.yamall.spark.helper.Evaluate;
import com.yahoo.labs.yamall.spark.helper.FileWriterToHDFS;
import com.yahoo.labs.yamall.spark.helper.ModelSerializationToHDFS;
import com.yahoo.labs.yamall.spark.helper.SparkLearnerFactory;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;

import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by busafekete on 8/29/17.
 */
public class Train {
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
    protected static boolean saveModelFlag = false;

    protected static ArrayList<Path>  featureFilePaths = null;
    protected static ArrayList<Path>  featureFilePathsTest = null;
    protected static int iter = 10;

    public static void init(SparkConf sparkConf) throws IOException {
        outputDir = sparkConf.get("spark.myapp.outdir");
        inputDir = sparkConf.get("spark.myapp.input");
        inputDirTest = sparkConf.get("spark.myapp.test");

        iter = Integer.parseInt(sparkConf.get("spark.myapp.iter"));

        method = sparkConf.get("spark.myapp.method");
        saveModelFlag = Boolean.parseBoolean(sparkConf.get("spark.myapp.save_model", "false"));

        // create learner
        learner = SparkLearnerFactory.getLearnerForSpark(sparkConf);
        vwparser = new VWParser(bitsHash, null, false);
        logFile = outputDir + "/log.txt";


        strb.append("--- Input: " + inputDir + "\n");
        strb.append("--- Output: " + outputDir + "\n");
        strb.append("--- Number of train files: " + featureFilePaths.size() + "\n");
        strb.append("--- Number of test files: " + featureFilePathsTest.size() + "\n");
        strb.append("--- Iter: " + iter + "\n");

        System.out.println(strb.toString());
        saveLog();

    }

    protected static void saveLog() throws IOException {
        FileWriterToHDFS.writeToHDFS(logFile, strb.toString());
    }


    public static void train() throws IOException {
        SparkConf sparkConf = new SparkConf().setAppName("spark yamall (single core training)");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

        init(sparkConf);

        JavaRDD<String> input = sparkContext.textFile(inputDir);
        input.persist(StorageLevel.MEMORY_AND_DISK());
        //long lineNum = input.count();

        //String line = "--- Number of training instance: " + lineNum + "\n";
        //System.out.println(line);
        //strb.append(line);

        double fraction = 1.0 / (iter + 1.0);
        System.out.println("--Fraction: " + fraction);

        //input.cache();
        //input.persist();


        // save example to hdfs
        JavaRDD<String> subsampTrain = input.sample(false, fraction);
        subsampTrain.persist(StorageLevel.MEMORY_AND_DISK());
        //JavaRDD<String> subsampTest = input.sample(false,fraction);
        //DataFrame wordsDataFrame = spark.createDataFrame(subsamp, String.class);


        long lineNumGrad = subsampTrain.count();
        line = "--- Number of instances for the gradient step: " + lineNumGrad + "\n";
        strb.append(line);


        long clusterStartTime = System.currentTimeMillis();

        for (Path featureFile : featureFilePaths) {
            System.out.println("----- Starting file " + featureFile + " Samp. num: " + numSamples);
            strb.append("----- Starting file " + featureFile + " Samp. num: " + numSamples + "\n");

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // compute gradient
            JavaRDD<String> subsamp = input.sample(false, fraction);
            BatchGradObject batchgradient = subsamp.treeAggregate(new BatchGradObject(bitsHash, prev_w, vwparser), new SeqOp(), new CombOp(), 11);
            batchgradient.normalizeBatchGradient();

            ind = checkIsInf(batchgradient.getGbatch());
            if (ind >= 0) {
                line = "--- Infinite value in batch grad vector \n";
                strb.append(line);
                saveLog(0);
                System.exit(0);
            }
            numSamples += batchgradient.getNum();
            line = "--- Gbatch step: " + batchgradient.gatherGradIter + " Cum loss: " + batchgradient.cumLoss + "\n";
            strb.append(line);
            saveLog(0);



            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            if (numSamples % evalPeriod == 0) {
                double trainLoss = cumLoss / (double) numSamples;
                double testLoss = Evaluate.eval(sparkContext, inputDirTest, learner, bitsHash);
                long clusteringRuntime = System.currentTimeMillis() - clusterStartTime;
                double elapsedTime = clusteringRuntime / 1000.0;
                double elapsedTimeInhours = elapsedTime / 3600.0;

                String line = String.format("%d %f %f %f\n", numSamples, trainLoss, testLoss, elapsedTimeInhours);
                strb.append(line);
                System.out.print(method + " " + line);

                saveLog();
            }

        }

        if (saveModelFlag) {
            String modelFile = "/model.bin";
            ModelSerializationToHDFS.saveModel(outputDir, modelFile, learner);
        }

    }

    public static void main(String[] args ) throws IOException {
        Train.train();
    }


}
