package com.yahoo.labs.yamall.spark.helper;

import com.yahoo.labs.yamall.ml.Learner;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import scala.Tuple2;

/**
 * Created by busafekete on 8/29/17.
 */
public class Evaluate {

    public static double eval(JavaSparkContext sparkContext, String inputDir, Learner learner, int bitsHash) {
        JavaRDD<String> input = sparkContext.textFile(inputDir );
        JavaPairRDD<String, Tuple2> posteriorsAndLables = input.mapToPair(new PosteriorComputer(learner, bitsHash));
        JavaRDD<Double> posteriors = posteriorsAndLables.map( (Function<Tuple2<String,Tuple2>,Double>) tup -> ((double)tup._2()._2()) );
        AverageFunction.AvgCount avg = posteriors.treeAggregate(new AverageFunction.AvgCount(0.0, 0), new AverageFunction.AddandCount(), new AverageFunction.Combine());
        return avg.avg();
    }

}
