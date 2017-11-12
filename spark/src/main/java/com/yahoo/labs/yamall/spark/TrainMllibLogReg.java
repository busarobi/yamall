package com.yahoo.labs.yamall.spark;

import com.yahoo.labs.yamall.spark.core.LearnerSpark;
import com.yahoo.labs.yamall.spark.helper.FileWriterToHDFS;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.io.IOException;

/**
 * Created by busafekete on 11/6/17.
 */
public class TrainMllibLogReg {
    protected static String inputDir;
    protected static String inputDirTest;
    protected static String outputDir;

    protected static String logFile = "";

    protected static StringBuilder strb = new StringBuilder("");

    protected static String method = null;
    protected static LearnerSpark learner = null;


    protected static boolean saveModelFlag = false;
    protected static int inputPartition = 0;

    public static void init(SparkConf sparkConf) throws IOException {
        FSDataInputStream fs ;

        outputDir = sparkConf.get("spark.myapp.outdir");
        inputDir = sparkConf.get("spark.myapp.input");
        inputDirTest = sparkConf.get("spark.myapp.test","");


        inputPartition = Integer.parseInt(sparkConf.get("spark.myapp.inputpartition", "0"));

        method = sparkConf.get("spark.myapp.method");
        saveModelFlag = Boolean.parseBoolean(sparkConf.get("spark.myapp.save_model", "false"));

        // create learner

        logFile = outputDir + "/log.txt";


        strb.append("--- Input: " + inputDir + "\n");
        strb.append("--- Input test: " + inputDirTest + "\n");
        strb.append("--- Output: " + outputDir + "\n");
        strb.append("--- Log file: " + logFile + "\n");

        strb.append("--- Input partition: " + inputPartition + "\n" );
        strb.append("--- Method: " + method + "\n");


        System.out.println(strb.toString());
        saveLog();

    }

    protected static void saveLog() throws IOException {
        FileWriterToHDFS.writeToHDFS(logFile, strb.toString());
    }

    static class LibSVMConverter implements Function<String,LabeledPoint> {
        @Override
        public LabeledPoint call(String line) throws Exception {
            String[] parts = line.split(" ");
            int initialIndex = 1;
            if ((parts.length>1) && (parts[1].charAt(0) == 'q')) {
                initialIndex = 2;
            }
            double label = Double.parseDouble(parts[0]);

            double[] v = new double[parts.length - initialIndex];
            int[] idx = new int[parts.length - initialIndex];
            for (int i = initialIndex; i < parts.length; i++) {
                idx[i - initialIndex] = Integer.parseInt(parts[i].split(":")[0])-1;
                v[i-initialIndex] = Double.parseDouble(parts[i].split(":")[1]);
            }
            int size = 123;
            //int size = idx[idx.length-1]+1;
            return (new LabeledPoint(label, Vectors.sparse(size, idx, v)));
        }
    }



    public static void run(SparkConf sparkConf, JavaSparkContext sparkContext) throws IOException {
        init(sparkConf);

//        JavaRDD<LabeledPoint> training = MLUtils.loadLibSVMFile(sparkContext, inputDir).toJavaRDD();
//        JavaRDD<LabeledPoint> test = MLUtils.loadLibSVMFile(sparkContext, inputDirTest).toJavaRDD();

        JavaRDD<String> trainingRdd = sparkContext.textFile(inputDir);
        JavaRDD<String> testRdd = sparkContext.textFile(inputDirTest);


        JavaRDD<LabeledPoint> training = trainingRdd.map(new LibSVMConverter());
        JavaRDD<LabeledPoint> test = testRdd.map(new LibSVMConverter());

//        JavaRDD<org.apache.spark.ml.feature.LabeledPoint> test = testRdd.map(new LibSVMConverter());
//        SparkSession spark = SparkSession
//                .builder()
//                .appName("SparkSession")
//                .getOrCreate();
//
//        Dataset<Row> training = spark.createDataFrame( trainingRdd, org.apache.spark.ml.feature.LabeledPoint.class);
        //Dataset<Row> test = spark.createDataFrame( testRdd, LabeledPoint.class);

//        JavaRDD<LabeledPoint> training = MLUtils.loadLibSVMFile(sparkContext, inputDir).toJavaRDD();
//        JavaRDD<LabeledPoint> test = MLUtils.loadLibSVMFile(sparkContext, inputDirTest).toJavaRDD();


        // Run training algorithm to build the model.
        LogisticRegressionWithLBFGS lrModel= new LogisticRegressionWithLBFGS();
//        LogisticRegressionWithSGD lrModel= new LogisticRegressionWithSGD();
        lrModel.optimizer().setRegParam(0.000001);
        lrModel.optimizer().setNumIterations(100);
        lrModel.optimizer().setConvergenceTol(10e-7);
        LogisticRegressionModel model = lrModel.run(training.rdd());

//        LogisticRegression lrModel = new LogisticRegression();
//        org.apache.spark.ml.classification.LogisticRegressionModel model = lrModel.fit(training);

        //lrModel.setAggregationDepth(10);


        //SparkSession spark = SparkSession.builder().sparkContext(sparkContext).getOrCreate();
        //Dataset<org.apache.spark.ml.feature.LabeledPoint> df = spark.createDataFrame(training.map( a -> new org.apache.spark.ml.feature.LabeledPoint(a.label(), a.features().asML())));
        //LogisticRegressionWithSGD lrModel= new LogisticRegressionWithSGD();
        //LogisticRegressionModel model = lrModel.fit(df);


        // Compute raw scores on the test set.
        JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));

        // Get evaluation metrics.
//        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
//        double accuracy = metrics.accuracy();
//        System.out.println("Accuracy = " + accuracy);

        // Save and load model
        //model.save(sparkContext, outputDir + "/javaLogisticRegressionWithLBFGSModel");

        int numBins = 100;
        // Get evaluation metrics.
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(predictionAndLabels.rdd(), numBins);

        // Precision by threshold
        JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
        System.out.println("Precision by threshold: " + precision.collect());
        strb.append("Precision by threshold: " + precision.collect() + "\n");

        // Recall by threshold
        JavaRDD<?> recall = metrics.recallByThreshold().toJavaRDD();
        System.out.println("Recall by threshold: " + recall.collect());
        strb.append("Recall by threshold: " + recall.collect() + "\n");

        // F Score by threshold
        JavaRDD<?> f1Score = metrics.fMeasureByThreshold().toJavaRDD();
        System.out.println("F1 Score by threshold: " + f1Score.collect());
        strb.append("F1 Score by threshold: " + f1Score.collect() + "\n" );

        JavaRDD<?> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD();
        System.out.println("F2 Score by threshold: " + f2Score.collect());
        strb.append("F2 Score by threshold: " + f2Score.collect() + "\n");

        // Precision-recall curve
        JavaRDD<?> prc = metrics.pr().toJavaRDD();
        strb.append("Precision-recall curve: " + prc.collect()+"\n");

        // Thresholds
        JavaRDD<Double> thresholds = precision.map(t -> Double.parseDouble(t._1().toString()));

        // ROC Curve
        JavaRDD<?> roc = metrics.roc().toJavaRDD();
        strb.append("+++ ROC curve: " + roc.collect() + "\n" );

        // AUPRC
        strb.append("+++ Area under precision-recall curve = " + metrics.areaUnderPR() + "\n" );
        // AUC
        strb.append("+++ Area under ROC = " + metrics.areaUnderROC() + "\n");

        saveLog();
    }

    public static void main(String[] args) throws IOException {
        SparkConf sparkConf = new SparkConf().setAppName("spark yamall (parallel training)");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

        run( sparkConf, sparkContext);
    }
}
