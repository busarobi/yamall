package com.yahoo.labs.yamall.spark.helper;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.Learner;
import com.yahoo.labs.yamall.parser.VWParser;
import org.apache.commons.lang3.SerializationUtils;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

/**
 * Created by busafekete on 8/30/17.
 */
public class LossComputer implements PairFunction<String, String, Tuple2> {
    public final double minPrediction = -50.0;
    public final double maxPrediction = 50.0;

    Learner learner = null;
    VWParser vwparser = null;

    public LossComputer(Learner learner, int bitsHash) {
        this.learner = (Learner) SerializationUtils.clone(learner);
        //this.learner = learner;
        vwparser = new VWParser(bitsHash, null, false);
    }

    @Override
    public Tuple2<String,Tuple2> call(String s) throws Exception {
        Instance sample = vwparser.parse(s);

        double score = learner.predict(sample);
        score = Math.min(Math.max(score, minPrediction), maxPrediction);

        double loss = learner.getLoss().lossValue(score, sample.getLabel()) * sample.getWeight();
        Tuple2 tup = new Tuple2(loss,sample.getLabel());
        return new Tuple2<>(sample.getTag(), tup);
    }
}
