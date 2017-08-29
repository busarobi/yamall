package com.yahoo.labs.yamall.spark.helper;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.Learner;
import com.yahoo.labs.yamall.ml.LogisticLinkFunction;
import com.yahoo.labs.yamall.parser.VWParser;
import org.apache.commons.lang.SerializationUtils;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

/**
 * Created by busafekete on 8/29/17.
 */
public class PosteriorComputer implements PairFunction<String, String, Tuple2> {
    public static double minPrediction = -50.0;
    public static double maxPrediction = 50.0;

    Learner learner = null;
    VWParser vwparser = null;
    LogisticLinkFunction link = new LogisticLinkFunction();

    public PosteriorComputer(Learner learner, int bitsHash) {
        this.learner = (Learner) SerializationUtils.clone(learner);
        vwparser = new VWParser(bitsHash, null, false);
    }

    @Override
    public Tuple2<String,Tuple2> call(String s) throws Exception {
        Instance sample = vwparser.parse(s);

        double score = learner.predict(sample);
        score = Math.min(Math.max(score, minPrediction), maxPrediction);
        score = link.apply(score);

        Tuple2 tup = new Tuple2(score,sample.getLabel());
        return new Tuple2<>(sample.getTag(), tup);
    }
}
