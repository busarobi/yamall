package com.yahoo.labs.yamall.spark.gradient;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.ml.LogisticLoss;
import com.yahoo.labs.yamall.ml.Loss;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function2;

import java.io.Serializable;

/**
 * Created by busafekete on 8/28/17.
 */
public class BatchGradient {

    public static class BatchGradientData implements Serializable{
        public static double minPrediction = -50.0;
        public static double maxPrediction = 50.0;

        protected double[] localGbatch;
        protected int size_hash;
        protected double[] localw;

        protected double[] featureMax;
        protected long[] featureCounts;

        protected Loss lossFnc = new LogisticLoss();
        public long gatherGradIter = 0;
        public double cumLoss = 0.0;
        protected int bits = 0;
        protected boolean normalizationFlag = false;


        BatchGradientData (int b, double[] weights) {
            bits = b;
            size_hash = 1 << bits;
            localw = new double[size_hash];
            for (int i = 0; i < size_hash; i++) localw[i] = weights[i];

            localGbatch = new double[size_hash];
            featureCounts = new long[size_hash];
            featureMax = new double[size_hash];
        }

        public double accumulateGradient(Instance sample) {
            gatherGradIter++;

            double pred = predict(sample);
            //pred = Math.min(Math.max(pred, minPrediction), maxPrediction);

            final double grad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());


            if (Math.abs(grad) > 1e-8) {
                sample.getVector().addScaledSparseVectorToDenseVector(localGbatch, grad);
            }
            cumLoss += lossFnc.lossValue(pred, sample.getLabel()) * sample.getWeight();

            for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
                int key = entry.getIntKey();
                double absVal = Math.abs(entry.getDoubleValue());
                if ( absVal > featureMax[key])
                    featureMax[key] = absVal;

                featureCounts[key]++;
            }


            return pred;
        }

        public double predict(Instance sample) {
            return sample.getVector().dot(localw);
        }

        public void aggregate(BatchGradientData  obj2) {
//            System.out.println("Before Cum loss obj1: " + cumLoss);
//            System.out.println("Before Cum loss obj2: " + obj2.cumLoss);
//
//            this.normalizeBatchGradient();
//            obj2.normalizeBatchGradient();
//
//            System.out.println("After Cum loss obj1: " + cumLoss);
//            System.out.println("After Cum loss obj2: " + obj2.cumLoss);
//
//            double sum = (double) (gatherGradIter + obj2.gatherGradIter);
//            if (sum > 0.0) {
//                for (int i = 0; i < size_hash; i++)
//                    localGbatch[i] = (gatherGradIter * localGbatch[i] + obj2.gatherGradIter * obj2.localGbatch[i]) / sum;
//                cumLoss = (gatherGradIter * cumLoss + obj2.gatherGradIter * obj2.cumLoss) / sum;
//                gatherGradIter += obj2.gatherGradIter;
//            }
//
//            System.out.println("After aggregation Cum loss obj1: " + cumLoss);


            gatherGradIter += obj2.gatherGradIter;
            for (int i = 0; i < size_hash; i++) {
                localGbatch[i] += obj2.localGbatch[i];
                featureCounts[i] += obj2.featureCounts[i];
                featureMax[i] = Math.max(featureMax[i], obj2.featureMax[i]);
            }
            cumLoss += obj2.cumLoss;




        }

        protected void normalizeBatchGradient() {
            if (normalizationFlag == false) {
                if (gatherGradIter > 0) {
                    for (int i = 0; i < size_hash; i++) localGbatch[i] /= (double) gatherGradIter;
                    cumLoss /= (double) gatherGradIter;
                    normalizationFlag = true;
                }
            }
        }

        public double[] getGbatch() {
            return localGbatch;
        }

        public long getNum() {
            return gatherGradIter;
        }

        public long[] getFeatureCounts() { return featureCounts; }

        public double[] getFeatureMax() { return featureMax; }
    }

    public static class CombOp implements Function2<BatchGradientData, BatchGradientData, BatchGradientData> {

        @Override
        public BatchGradientData call(BatchGradientData v1, BatchGradientData v2) throws Exception {
            v1.aggregate(v2);
//            v2 = null;
            return v1;
        }
    }

    public static class SeqOp implements Function2<BatchGradientData, Instance, BatchGradientData> {

        @Override
        public BatchGradientData call(BatchGradientData v1, Instance v2) throws Exception {
            v1.accumulateGradient(v2);
            return v1;
        }
    }


    public static BatchGradientData computeGradient(JavaRDD<Instance> data, int bit, double[] w ){
        BatchGradientData batchgradient = data.treeAggregate(new BatchGradientData(bit, w), new SeqOp(), new CombOp(), 5);
        //BatchGradientData batchgradient = data.aggregate(new BatchGradientData(bit, w), new SeqOp(), new CombOp());
        //batchgradient.normalizeBatchGradient();
        return batchgradient;
    }
}
