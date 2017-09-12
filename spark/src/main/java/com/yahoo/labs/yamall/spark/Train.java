package com.yahoo.labs.yamall.spark;

import com.yahoo.labs.yamall.parser.VWParser;
import com.yahoo.labs.yamall.spark.core.PerCoordinateSVRGSpark;
import com.yahoo.labs.yamall.spark.core.SparkLearner;
import com.yahoo.labs.yamall.spark.gradient.BatchGradient;
import com.yahoo.labs.yamall.spark.helper.Evaluate;
import com.yahoo.labs.yamall.spark.helper.FileWriterToHDFS;
import com.yahoo.labs.yamall.spark.helper.ModelSerializationToHDFS;
import com.yahoo.labs.yamall.spark.helper.PosteriorComputer;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

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

    protected static int bitsHash = 23;
    protected static StringBuilder strb = new StringBuilder("");

    protected static String method = null;
    protected static SparkLearner learner = null;
    protected static VWParser vwparser = null;

    protected static double minPrediction = -50.0;
    protected static double maxPrediction = 50.0;

    protected static double cumLoss = 0.0;
    protected static long numSamples = 0;
    protected static boolean saveModelFlag = false;
    protected static int inputPartition = 0;
    protected static int iter = 10;

    public static void init(SparkConf sparkConf) throws IOException {
        FSDataInputStream fs ;

        outputDir = sparkConf.get("spark.myapp.outdir");
        inputDir = sparkConf.get("spark.myapp.input");
        inputDirTest = sparkConf.get("spark.myapp.test");

        iter = Integer.parseInt(sparkConf.get("spark.myapp.iter"));
        bitsHash = Integer.parseInt(sparkConf.get("spark.myapp.bitshash", "23"));

        inputPartition = Integer.parseInt(sparkConf.get("spark.myapp.inputpartition", "0"));

        method = sparkConf.get("spark.myapp.method");
        saveModelFlag = Boolean.parseBoolean(sparkConf.get("spark.myapp.save_model", "false"));

        // create learner
        vwparser = new VWParser(bitsHash, null, false);
        logFile = outputDir + "/log.txt";


        strb.append("--- Input: " + inputDir + "\n");
        strb.append("--- Input test: " + inputDirTest + "\n");
        strb.append("--- Output: " + outputDir + "\n");
        strb.append("--- Log file: " + logFile + "\n");

        strb.append("--- Input partition: " + inputPartition + "\n" );
        strb.append("--- Method: " + method + "\n");
        strb.append("--- Iter: " + iter + "\n");
        strb.append("--- Bits hash: " + bitsHash + "\n");

        System.out.println(strb.toString());
        saveLog();

    }

    protected static void saveLog() throws IOException {
        FileWriterToHDFS.writeToHDFS(logFile, strb.toString());
    }

    protected static Class[] getKyroclassArray(){
        ArrayList<Class> array = new ArrayList<>();

        array.add(BatchGradient.class);
        array.add(BatchGradient.CombOp.class);
        array.add(BatchGradient.SeqOp.class);
        array.add(PosteriorComputer.class);

        array.add(SparkLearner.class);
        array.add(PerCoordinateSVRGSpark.class);


        return (Class[]) array.toArray();
    }

    public static void run() throws IOException {
        SparkConf sparkConf = new SparkConf().setAppName("spark yamall (parallel training)");
        //sparkConf.registerKryoClasses(getKyroclassArray());
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

        init(sparkConf);

        JavaRDD<String> input = null;
        if (inputPartition>0) {
            input = sparkContext.textFile(inputDir);
            input = input.repartition(inputPartition);
        } else
            input = sparkContext.textFile(inputDir);


        //long lineNum = input.count();
        learner = new PerCoordinateSVRGSpark(sparkConf,strb,bitsHash);
        learner.train(input);

        if (! inputDirTest.isEmpty()){
            double testLoss = Evaluate.getLoss(sparkContext,inputDirTest,learner, bitsHash);
            String line = String.format("%d %f %f %f\n", numSamples, testLoss);
            strb.append(line);

            System.out.print(method + " " + line);
            Evaluate.computeResult(strb,sparkContext,inputDirTest,learner, bitsHash);
            saveLog();


            if (saveModelFlag) {
                ModelSerializationToHDFS.saveModel(outputDir, learner);
            }


            Evaluate.computeResult(strb,sparkContext,inputDirTest,learner,bitsHash);
        }

        if (saveModelFlag) {
            String modelFile = "/model.bin";
            ModelSerializationToHDFS.saveModel(outputDir, learner);
        }

    }


    public static void main(String[] args ) throws IOException {
        Train.run();
    }


}
