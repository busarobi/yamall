package com.yahoo.labs.yamall.spark;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.LogisticLoss;
import com.yahoo.labs.yamall.parser.VWParser;
import com.yahoo.labs.yamall.spark.helper.Evaluate;
import com.yahoo.labs.yamall.spark.helper.FileWriterToHDFS;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.DoubleFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;

/**
 * Created by busafekete on 11/8/17.
 */
public class TrainSparkMLLogReg {
    protected enum INPUT_TYPE { LIBSVM, VW, CLKB};
    protected static String inputDir;
    protected static String inputDirTest;
    protected static String outputDir;

    protected static String logFile = "";
    protected static int sparkIter = 10;

    protected static StringBuilder strb = new StringBuilder("");

    protected static String method = null;
    protected static int bitsHash = 23;

    protected static double regParameter = 0.0;
    protected static boolean saveModelFlag = false;
    protected static INPUT_TYPE inputType = INPUT_TYPE.LIBSVM;
    protected static int inputPartition = 0;

    public static void init(SparkConf sparkConf) throws IOException {
        FSDataInputStream fs ;

        outputDir = sparkConf.get("spark.myapp.outdir");
        inputDir = sparkConf.get("spark.myapp.input");
        inputDirTest = sparkConf.get("spark.myapp.test","");

        sparkIter = Integer.parseInt(sparkConf.get("spark.myapp.iter", "10"));
        inputPartition = Integer.parseInt(sparkConf.get("spark.myapp.inputpartition", "0"));
        regParameter = Double.parseDouble(sparkConf.get("spark.myapp.reg", "0.0"));

        method = sparkConf.get("spark.myapp.method");
        saveModelFlag = Boolean.parseBoolean(sparkConf.get("spark.myapp.save_model", "false"));

        String typeOFInput = sparkConf.get("spark.myapp.inputtype");
        switch (typeOFInput) {
            case "libsvm":
                inputType = INPUT_TYPE.LIBSVM;
                break;
            case "vw":
                inputType = INPUT_TYPE.VW;
                break;
            case "clkb":
                inputType = INPUT_TYPE.CLKB;
                break;
            default:
                inputType = INPUT_TYPE.LIBSVM;
                break;
        }


        bitsHash = Integer.parseInt(sparkConf.get("spark.myapp.bitshash", "23"));

        // create learner

        logFile = outputDir + "/log.txt";


        strb.append("--- Input: " + inputDir + "\n");
        strb.append("--- Input test: " + inputDirTest + "\n");
        strb.append("--- Output: " + outputDir + "\n");
        strb.append("--- Log file: " + logFile + "\n");

        strb.append("--- Iter: " + sparkIter + "\n");
        strb.append("--- Regularization: " + regParameter + "\n");
        strb.append("--- Input partition: " + inputPartition + "\n" );
        strb.append("--- Method: " + method + "\n");

        strb.append("--- Input type: " + inputType.toString() + "\n");
        strb.append("--- Bits hash: " + bitsHash + "\n");



        System.out.println(strb.toString());
        saveLog();

    }

    protected static void saveLog() throws IOException {
        FileWriterToHDFS.writeToHDFS(logFile, strb.toString());
    }


    static class VWToMLLabeledPoint implements Function<String,org.apache.spark.mllib.regression.LabeledPoint> {
        protected int bitHash = 23;
        protected int size = 2;
        protected VWParser parser = null;

        public VWToMLLabeledPoint(int bitHash){
            parser = new VWParser(bitsHash, null, false);
            this.bitHash = bitHash;
            size = 2;
            size <<= bitHash;
        }

        @Override
        public org.apache.spark.mllib.regression.LabeledPoint call(String line) throws Exception {
            line = line.trim();
            Instance sample = parser.parse(line);
            TreeMap<Integer, Double> data = new TreeMap<>();

            for(Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
                int key = entry.getIntKey();
                double val = entry.getDoubleValue();
                data.put(key, val);
            }

            double[] v = new double[data.size()];
            int[] idx = new int[data.size()];

            int i = 0;
            for(Map.Entry<Integer, Double> entry : data.entrySet()){
                idx[i] = entry.getKey();
                v[i] = entry.getValue();
                i++;
            }


            return (new org.apache.spark.mllib.regression.LabeledPoint(sample.getLabel(), Vectors.sparse(size, idx, v)));
        }
    }

    static class VWToRow implements Function<String,Row> {
        protected int bitHash = 23;
        protected int size = 2;
        protected VWParser parser = null;


        VWToRow(int bitHash){
            parser = new VWParser(bitsHash, null, false);
            this.bitHash = bitHash;
            this.size = 2;
            this.size <<= bitHash;

        }

        @Override
        public Row call(String line) throws Exception {
            line = line.trim();
            Instance sample = parser.parse(line);
            TreeMap<Integer, Double> data = new TreeMap<>();

            for(Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
                int key = entry.getIntKey();
                double val = entry.getDoubleValue();
                data.put(key, val);

            }
            double[] v = new double[data.size()];
            int[] idx = new int[data.size()];

            int i = 0;
            for(Map.Entry<Integer, Double> entry : data.entrySet()){
                idx[i] = entry.getKey();
                v[i] = entry.getValue();
                i++;
            }

            Row points = RowFactory.create(sample.getLabel(),org.apache.spark.ml.linalg.Vectors.sparse(size, idx, v));
            return points;
        }
    }

    static class MaxFeatIndexClkb implements DoubleFunction<String> {
        @Override
        public double call(String line) throws Exception {
            String features = line.split("\\|")[1].trim();
            String[] parts = features.split(" ");
            double max = 0;
            for(String p: parts) {
                double idx = Double.parseDouble(p);
                if (idx>max) max = idx;
            }
            return max;
        }
    }

    static class ClkbToRow implements Function<String,Row> {
        protected int size = 0;

        ClkbToRow(int numOfFeatures){
            this.size = numOfFeatures;
        }

        @Override
        public Row call(String line) throws Exception {
            String labelString = line.split("\\|")[0].split(" ")[0];
            double label = (Double.parseDouble(labelString)+1.0)/2.0;

            String[] parts = line.split("\\|")[1].trim().split(" ");


            TreeMap<Integer, Double> data = new TreeMap<>();

            for(String p: parts) {
                int idx = Integer.parseInt(p)-1;
                data.put(idx, 1.0);

            }
            double[] v = new double[data.size()];
            int[] idx = new int[data.size()];

            int i = 0;
            for(Map.Entry<Integer, Double> entry : data.entrySet()){
                idx[i] = entry.getKey();
                v[i] = entry.getValue();
                i++;
            }

            Row points = RowFactory.create(label,org.apache.spark.ml.linalg.Vectors.sparse(size, idx, v));
            return points;
        }
    }


    static class ClkbToMLLabeledPoint implements Function<String,org.apache.spark.mllib.regression.LabeledPoint> {
        protected int size = 0;

        ClkbToMLLabeledPoint(int numOfFeatures){
            this.size = numOfFeatures;
        }

        @Override
        public org.apache.spark.mllib.regression.LabeledPoint call(String line) throws Exception {
            String labelString = line.split("\\|")[0].split(" ")[0];
            double label = (Double.parseDouble(labelString)+1.0)/2.0;
            String[] parts = line.split("\\|")[1].trim().split(" ");


            TreeMap<Integer, Double> data = new TreeMap<>();

            for(String p: parts) {
                int idx = Integer.parseInt(p)-1;
                data.put(idx, 1.0);

            }
            double[] v = new double[data.size()];
            int[] idx = new int[data.size()];

            int i = 0;
            for(Map.Entry<Integer, Double> entry : data.entrySet()){
                idx[i] = entry.getKey();
                v[i] = entry.getValue();
                i++;
            }

            return (new org.apache.spark.mllib.regression.LabeledPoint(label, Vectors.sparse(size, idx, v)));
        }
    }




    static class LibSVMToMLLabeledPoint implements Function<String,org.apache.spark.mllib.regression.LabeledPoint> {
        protected int size = 0;

        public LibSVMToMLLabeledPoint(int numOfFeatures){
            this.size = numOfFeatures;
        }

        @Override
        public org.apache.spark.mllib.regression.LabeledPoint call(String line) throws Exception {
            line = line.split("#")[0];
            line = line.trim();

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

            return (new org.apache.spark.mllib.regression.LabeledPoint(label, Vectors.sparse(size, idx, v)));
        }
    }

    static class LibSVMToRow implements Function<String,Row> {
        protected int size = 0;

        LibSVMToRow(int size){
            this.size = size;
        }

        @Override
        public Row call(String line) throws Exception {
            line = line.split("#")[0];
            line = line.trim();

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
            //Vector[] points = {org.apache.spark.ml.linalg.Vectors.sparse(size, idx, v)};
            //int size = idx[idx.length-1]+1;
            Row points = RowFactory.create(label,org.apache.spark.ml.linalg.Vectors.sparse(size, idx, v));
            return points;
        }
    }

    static class MaxFeatIndex implements DoubleFunction<String> {
        @Override
        public double call(String line) throws Exception {
            line = line.split("#")[0];
            line = line.trim();

            String[] parts = line.split(" ");
            int initialIndex = 1;
            if ((parts.length>1) && (parts[1].charAt(0) == 'q')) {
                initialIndex = 2;
            }
            double max = Double.parseDouble(parts[parts.length-1].split(":")[0]);
            return max;
        }
    }



    public static void run(SparkConf sparkConf, JavaSparkContext sparkContext) throws IOException {
        init(sparkConf);

        SparkSession spark = SparkSession
                .builder()
                .appName("Spark ML training")
                .getOrCreate();

        JavaRDD<String> trainingRdd = sparkContext.textFile(inputDir);
        JavaRDD<String> testingRdd = sparkContext.textFile(inputDirTest);

        // number of features
        Integer numOfFeatures = 0;
        if (inputType == INPUT_TYPE.LIBSVM) {
            JavaDoubleRDD maxVals = trainingRdd.mapToDouble(new MaxFeatIndex());
            numOfFeatures = maxVals.max().intValue();
        } else if (inputType == INPUT_TYPE.VW) {
            numOfFeatures = 2;
            numOfFeatures <<= bitsHash;
        } else if (inputType == INPUT_TYPE.CLKB) {
            JavaDoubleRDD maxVals = trainingRdd.mapToDouble(new MaxFeatIndexClkb());
            numOfFeatures = maxVals.max().intValue();
        }

        System.out.println( "Number of features: " + numOfFeatures );
        strb.append( "Number of features: " + numOfFeatures + "\n");
        saveLog();

        //
        JavaRDD<Row> pointsTrain = null;
        JavaRDD<Row> pointsTest = null;

        if (inputType == INPUT_TYPE.LIBSVM) {
            pointsTrain = trainingRdd.map(new LibSVMToRow(numOfFeatures));
            pointsTest = testingRdd.map(new LibSVMToRow(numOfFeatures));
        } else if (inputType == INPUT_TYPE.VW) {
            pointsTrain = trainingRdd.map(new VWToRow(bitsHash));
            pointsTest = testingRdd.map(new VWToRow(bitsHash));
        } else if (inputType == INPUT_TYPE.CLKB) {
            pointsTrain = trainingRdd.map(new ClkbToRow(numOfFeatures));
            pointsTest = testingRdd.map(new ClkbToRow(numOfFeatures));
        }


        StructField[] fields = {
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())};
        StructType schema = new StructType(fields);
        Dataset<Row> trainingdf = spark.createDataFrame(pointsTrain, schema);
        Dataset<Row> testingdf = spark.createDataFrame(pointsTest, schema);

        long clusterStartTime = System.currentTimeMillis();

        // Run training algorithm to build the model.
        LogisticRegression lr = new LogisticRegression().setMaxIter(sparkIter)
                .setRegParam(regParameter)
                .setStandardization(false)
                .setAggregationDepth(5)
                .setTol(10e-9)
                .setElasticNetParam(0.5);

        LogisticRegressionModel lrModel = lr.train(trainingdf);

        long clusteringRuntime = System.currentTimeMillis() - clusterStartTime;
        double elapsedTime = clusteringRuntime / 1000.0;
        double elapsedTimeInhours = elapsedTime / 3600.0;


        // Print the coefficients and intercepts for logistic regression with multinomial family
//        System.out.println("Multinomial coefficients: " + lrModel.coefficientMatrix()
//                + "\nMultinomial intercepts: " + lrModel.interceptVector());
//        strb.append("Multinomial coefficients: " + lrModel.coefficientMatrix() + "\n");
//        strb.append("Multinomial intercepts: " +lrModel.interceptVector() + "\n");
        // Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
        // example

//        strb.append( "#0 bias " + lrModel.intercept() + "\n");
//        double[] coeffs = lrModel.coefficients().toArray();
//        for(int i = 0; i < coeffs.length; i++ ){
//            strb.append( "#"+(i+1) + "\t" + String.format("%.10f", coeffs[i]) + "\n");
//        }
//
//        saveLog();

        LogisticRegressionTrainingSummary trainingSummary = lrModel.summary();

        // Obtain the loss per iteration.
        strb.append("Loss: \n" );
        double[] objectiveHistory = trainingSummary.objectiveHistory();
        for (double lossPerIteration : objectiveHistory) {
            System.out.println(lossPerIteration);
            strb.append(lossPerIteration + "\n");
        }
        strb.append("\n");
        // Obtain the metrics useful to judge performance on test data.
        // We cast the summary to a BinaryLogisticRegressionSummary since the problem is a binary
        // classification problem.
        BinaryLogisticRegressionSummary binarySummary =
                (BinaryLogisticRegressionSummary) trainingSummary;

        // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
        Dataset roc = binarySummary.roc();
        roc.show();
        roc.select("FPR").show();
        System.out.println(binarySummary.areaUnderROC());
        strb.append( "AUC on train: " + binarySummary.areaUnderROC() + "\n");

        Dataset<Row> predictions = lrModel.transform(testingdf);

        // Select example rows to display.
        BinaryClassificationEvaluator binaryEvaluator = new BinaryClassificationEvaluator();
        double result = binaryEvaluator.evaluate(predictions);
        System.out.println( "AUC on test: " + result);

        strb.append("Elapsed time in hours: " + elapsedTimeInhours + "\n");
        strb.append("AUC on test: " + result + "\n");
        saveLog();

        //FileDeleter.directoryDeleter(outputDir + "/scores_df");
        //predictions.javaRDD().saveAsTextFile(outputDir + "/scores_Df");
        //
        JavaRDD<String> testRdd = sparkContext.textFile(inputDirTest);
        JavaRDD<org.apache.spark.mllib.regression.LabeledPoint> test = null;


        if (inputType == INPUT_TYPE.LIBSVM) {
            test = testRdd.map(new LibSVMToMLLabeledPoint(numOfFeatures));
        } else if (inputType == INPUT_TYPE.VW) {
            test = testRdd.map(new VWToMLLabeledPoint(bitsHash));
        } else if (inputType == INPUT_TYPE.CLKB) {
            test = testRdd.map(new ClkbToMLLabeledPoint(numOfFeatures));
        }


        JavaPairRDD<Double,Double> predictionAndLabels = test.mapToPair(p -> new Tuple2<Double,Double>(lrModel.predictRaw(p.features().asML()).toArray()[1], p.label()));
        JavaDoubleRDD losses = predictionAndLabels.mapToDouble( scoreandlabel -> (new LogisticLoss()).lossValue(scoreandlabel._1(),(2.0*scoreandlabel._2())-1.0));
        Evaluate.computeResult(strb, predictionAndLabels.mapToPair(a -> new Tuple2<Object, Object>(a._1(),a._2())), 10);

        double avgloss = losses.mean();
        strb.append("+++ Avg. loss on test: " + avgloss + "\n");

        saveLog();

    }



    public static void main(String[] args) throws IOException {
        SparkConf sparkConf = new SparkConf().setAppName("spark yamall (parallel training)");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

        run( sparkConf, sparkContext);
    }

}
