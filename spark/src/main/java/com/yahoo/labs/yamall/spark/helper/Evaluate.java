package com.yahoo.labs.yamall.spark.helper;

import com.yahoo.labs.yamall.ml.Learner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import scala.Tuple2;

import java.io.IOException;
import java.util.List;

/**
 * Created by busafekete on 8/29/17.
 */
public class Evaluate {

    public static double getLoss(JavaSparkContext sparkContext, String inputDir, Learner learner, int bitsHash) {
        JavaRDD<String> input = sparkContext.textFile(inputDir );
        JavaPairRDD<String, Tuple2> lossesAndLables = input.mapToPair(new LossComputer(learner, bitsHash));
        JavaRDD<Double> losses = lossesAndLables.map( (Function<Tuple2<String,Tuple2>,Double>) tup -> ((double)tup._2()._1()) );
        List<Double> loss = losses.collect();

        double avg = 0.0;
        for (Double l : loss) avg += l;
        return (avg / (double)loss.size());
    }

//    public static void getLossAndCount(JavaSparkContext sparkContext, String inputDir, Learner learner, int bitsHash, StringBuffer strb ) {
//        JavaRDD<String> input = sparkContext.textFile(inputDir );
//        JavaPairRDD<String, Tuple2> posteriorsAndLables = input.mapToPair(new LossComputer(learner, bitsHash));
//        JavaRDD<Double> posteriors = posteriorsAndLables.map( (Function<Tuple2<String,Tuple2>,Double>) tup -> ((double)tup._2()._1()) );
//
//        AverageFunction.AvgCount avg = posteriors.treeAggregate(new AverageFunction.AvgCount(0.0, 0), new AverageFunction.AddandCount(), new AverageFunction.Combine());
//        strb.append("" + avg.avg() + " " + avg.getNum() );
//    }

    public static void computeResult(StringBuilder strb, JavaSparkContext sparkContext, String inputDir, Learner learner, int bitsHash) {
        JavaRDD<String> input = sparkContext.textFile(inputDir );
        JavaPairRDD<String, Tuple2> posteriorsAndLables = input.mapToPair(new PosteriorComputer(learner, bitsHash));
        JavaPairRDD<Object, Object> predictionAndLabels = posteriorsAndLables.values().mapToPair((PairFunction<Tuple2, Object, Object>) tup -> new Tuple2<>(tup._1(),tup._2()));

        // Get evaluation metrics.
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(predictionAndLabels.rdd());

        // Precision by threshold
//        JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
//        System.out.println("Precision by threshold: " + precision.collect());
//        strb.append("Precision by threshold: " + precision.collect() + "\n");
//
//        // Recall by threshold
//        JavaRDD<?> recall = metrics.recallByThreshold().toJavaRDD();
//        System.out.println("Recall by threshold: " + recall.collect());
//        strb.append("Recall by threshold: " + recall.collect() + "\n");
//
//        // F Score by threshold
//        JavaRDD<?> f1Score = metrics.fMeasureByThreshold().toJavaRDD();
//        System.out.println("F1 Score by threshold: " + f1Score.collect());
//        strb.append("F1 Score by threshold: " + f1Score.collect() + "\n" );
//
//        JavaRDD<?> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD();
//        System.out.println("F2 Score by threshold: " + f2Score.collect());
//        strb.append("F2 Score by threshold: " + f2Score.collect() + "\n");
//
        // Precision-recall curve
//        JavaRDD<?> prc = metrics.pr().toJavaRDD();
//        strb.append("Precision-recall curve: " + prc.collect()+"\n");
//
//        // Thresholds
//        JavaRDD<Double> thresholds = precision.map(t -> Double.parseDouble(t._1().toString()));

        // ROC Curve
        // JavaRDD<?> roc = metrics.roc().toJavaRDD();
        // strb.append("+++ ROC curve: " + roc.collect() + "\n" );

        // AUPRC
        strb.append("+++ Area under precision-recall curve = " + metrics.areaUnderPR() + "\n" );
        // AUC
        strb.append("+++ Area under ROC = " + metrics.areaUnderROC() + "\n");
    }


    public static void main(String[] args) throws IOException {
        SparkConf sparkConf = new SparkConf().setAppName("test evaluate");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

        String modelDir = sparkConf.get("spark.myapp.modeldir");
        String modelFname = sparkConf.get("spark.myapp.modelfile");

        Learner learner = ModelSerializationToHDFS.loadModel(modelDir,modelFname);

        String inputDir = sparkConf.get("spark.myapp.inputdir");
        String output = sparkConf.get("spark.myapp.output");
        int bitsHash = Integer.parseInt(sparkConf.get("spark.myapp.bitshash", "23"));


        StringBuilder strb = new StringBuilder("");

        strb.append( "Model: " + modelDir + "/" + modelFname + "\n" );
        strb.append( "Input: " + inputDir + "\n" );
        strb.append( "Output: " + output + "\n" );

        double loss = getLoss(sparkContext, inputDir, learner, bitsHash);
        strb.append( "Loss: " + loss + "\n");

        computeResult(strb, sparkContext, inputDir, learner, bitsHash);

        System.out.println( strb.toString());
        FileWriterToHDFS.writeToHDFS( output, strb.toString());
    }

}
