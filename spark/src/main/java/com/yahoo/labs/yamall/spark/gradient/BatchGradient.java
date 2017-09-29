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

        protected long[] lastUpdated;


        BatchGradientData (int b, double[] weights) {
            bits = b;
            size_hash = 1 << bits;
            localw = new double[size_hash];
            for (int i = 0; i < size_hash; i++) localw[i] = weights[i];

            localGbatch = new double[size_hash];
            featureCounts = new long[size_hash];
            featureMax = new double[size_hash];
            lastUpdated = new long[size_hash];
        }

        public double accumulateGradient(Instance sample) {
//            gatherGradIter++;
//
//            double pred = predict(sample);
//            //pred = Math.min(Math.max(pred, minPrediction), maxPrediction);
//
//            final double grad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());
//
//
//            if (Math.abs(grad) > 1e-8) {
//                sample.getVector().addScaledSparseVectorToDenseVector(localGbatch, grad);
//            }
//            cumLoss += lossFnc.lossValue(pred, sample.getLabel()) * sample.getWeight();
//
//            for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
//                int key = entry.getIntKey();
//                double absVal = Math.abs(entry.getDoubleValue());
//                if ( absVal > featureMax[key])
//                    featureMax[key] = absVal;
//
//                featureCounts[key]++;
//            }
//


            //maybe more stable averaging
            gatherGradIter++;

            double pred = predict(sample);

            final double negativeGrad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());

            for(Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
                int key = entry.getIntKey();

                double absVal = Math.abs(entry.getDoubleValue());
                if ( absVal > featureMax[key])
                    featureMax[key] = absVal;
                featureCounts[key]++;

                double x_i = entry.getDoubleValue();
                double negativeGrad_i = x_i * negativeGrad;
                double currentAverageGrad = localGbatch[key];
                if(lastUpdated[key] < gatherGradIter - 1) {
                    currentAverageGrad *= ((double) lastUpdated[key] / (double) (gatherGradIter - 1));
                }

                currentAverageGrad += (negativeGrad_i - currentAverageGrad) / gatherGradIter;
                lastUpdated[key] = gatherGradIter;
                localGbatch[key] = currentAverageGrad;
            }
            double currentLoss = lossFnc.lossValue(pred, sample.getLabel()) * sample.getWeight();
            cumLoss += ( (cumLoss - currentLoss) / (double) gatherGradIter);
            return pred;
        }

        public double predict(Instance sample) {
            return sample.getVector().dot(localw);
        }

        public void aggregate(BatchGradientData  obj2) {
//            this.normalizeBatchGradient();
//            obj2.normalizeBatchGradient();

            this.doLazyUpdates();
            obj2.doLazyUpdates();

            double sum = (double) (gatherGradIter + obj2.gatherGradIter);
            for (int i = 0; i < size_hash; i++) {
                localGbatch[i] = ((double)gatherGradIter/(gatherGradIter + obj2.gatherGradIter)) * localGbatch[i] + ((double)obj2.gatherGradIter/(gatherGradIter + obj2.gatherGradIter)) * obj2.localGbatch[i];
                featureCounts[i] += obj2.featureCounts[i];
                featureMax[i] = Math.max(featureMax[i], obj2.featureMax[i]);
                lastUpdated[i] = gatherGradIter + obj2.gatherGradIter;

            }
            gatherGradIter += obj2.gatherGradIter;
            cumLoss = (gatherGradIter * cumLoss + obj2.gatherGradIter * obj2.cumLoss) / sum;
//            gatherGradIter += obj2.gatherGradIter;
            normalizationFlag = true;

//            // naive aggregation which requires normalization after
//            gatherGradIter += obj2.gatherGradIter;
//            for (int i = 0; i < size_hash; i++) {
//                localGbatch[i] += obj2.localGbatch[i];
//                featureCounts[i] += obj2.featureCounts[i];
//                featureMax[i] = Math.max(featureMax[i], obj2.featureMax[i]);
//            }
//            cumLoss += obj2.cumLoss;
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

        protected void doLazyUpdates() {
            for(int i=0; i<size_hash; i++) {
                if (lastUpdated[i] < gatherGradIter) {
                    localGbatch[i] *= (double) lastUpdated[i] / (double) gatherGradIter;
                }
                lastUpdated[i] = gatherGradIter;
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
//        BatchGradientData batchgradient = data.aggregate(new BatchGradientData(bit, w), new SeqOp(), new CombOp());
//        batchgradient.normalizeBatchGradient();
        return batchgradient;
    }
}
