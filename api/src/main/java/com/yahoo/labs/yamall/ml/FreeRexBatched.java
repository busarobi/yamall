// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.labs.yamall.ml;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.core.SparseVector;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * created by acutkosky on 8/3/2017
 */

@SuppressWarnings("serial")
public class FreeRexBatched implements Learner {

    private double inverseEtaSq = 0.0;
    private double maxNegativeGrad = 0.0;
    private double[] sumNegativeGrads;
    private int[] lastUpdated;
    private transient double[] weightScaling;
    private transient double[] center;
    private transient double[] w;

    private double scaling = 1.0;

    private double sumNegativeGradDotGBatch = 0.0;
    private int sumNegativeGradDotGBatch_Iteration = 0;

    private double sumNegativeGradNormSq = 0.0;
    private int sumNegativeGradNormSq_Iteration = 0;

    private double[] negativeBatchGrad;

    private double negativeBatchGradNormSq = 0.0;

    private boolean useScaling = false;
    private boolean useWeightScaling = false;

    private double k_inv = 0.45; // sqrt(1/5)
    private Loss lossFnc;
    public int iter = 0;
    private int size_hash = 0;

    public FreeRexBatched(
            int bits) {
        size_hash = 1 << bits;

        sumNegativeGrads = new double[size_hash];

        w = new double[size_hash];
        center = new double[size_hash]; //default 0, but allows for FTRL with arbitrary centering.

        negativeBatchGrad = new double[size_hash];
        lastUpdated = new int[size_hash];
        weightScaling = new double[size_hash];

    }

    public void setCenter(double[] center) {
        for(int i=0; i < size_hash; i++) {
            this.center[i] = center[i];
            this.w[i] = center[i];
        }
    }

    public void setNegativeBatchGrad(double[] negativeBatchGrad) {
        negativeBatchGradNormSq = 0.0;
        for(int i=0; i< size_hash; i++) {
            this.negativeBatchGrad[i] = negativeBatchGrad[i];
            negativeBatchGradNormSq += negativeBatchGrad[i] * negativeBatchGrad[i];
        }
    }

    public void resetLastUpdated() {
        for (int i=0; i<size_hash; i++) {
            lastUpdated[i] = 0;
        }
    }
    public void reset() {
        for(int i=0; i< size_hash; i++) {
            lastUpdated[i] = 0;
            sumNegativeGrads[i] = 0;
            w[i] = center[i];
        }
        sumNegativeGradNormSq = 0.0;
        sumNegativeGradDotGBatch = 0.0;
        sumNegativeGradDotGBatch_Iteration = 0;
        sumNegativeGradNormSq_Iteration = 0;
        inverseEtaSq = 0.0;
        scaling = 1.0;
        iter = 0;
    }

    protected double updateSumNegativeGradDotNegativeBatchGrad(SparseVector sparseNegativeGrad) {
        /**
         * Assume gradient is g + G where g is sparse and G is dense.
         * We want to compute (S + g + G) * G where * is the dot-product,
         * given a pre-computed S * G and |G|^2. This is equal to
         * S * G + g * G + |G|^2 which we can do in O(sparsity) time.
         */
        sumNegativeGradDotGBatch = sumNegativeGradDotGBatch + sparseNegativeGrad.dot(negativeBatchGrad) + negativeBatchGradNormSq;
        sumNegativeGradDotGBatch_Iteration++;
        return sumNegativeGradDotGBatch;
    }

    protected double getFullNegativeGradNormSquared(SparseVector sparseNegativeGrad) {
        /**
         * Assume gradient is g + G where g is sparse and G is dense.
         * We want to compute |g+G|^2 given a pre-computed |G|^2.
         * This is just |g|^2 + 2g * G + |G|^2
         */
        double dotProduct = sparseNegativeGrad.dot(negativeBatchGrad);
        return sparseNegativeGrad.squaredL2Norm() + 2 * dotProduct + negativeBatchGradNormSq;
    }

    protected double updateSumNegativeGradNormSq(SparseVector sparseNegativeGrad) {
        /**
         * TO MAKE Tth UPDATE, REQUIRES ALL DEPENDENCIES TO BE AT T-1th VALUE.
         * (I.E. CALL THIS UPDATE FIRST)
         *
         * We want to compute |S + g + G|^2 where g is sparse,
         * given pre-computed S * G and |S|^2. This is
         * |S|^2 + |g+G|^2 + 2 S*g + 2 S*G
         */
        assert(sumNegativeGradDotGBatch_Iteration == sumNegativeGradNormSq_Iteration);
        double fullNegativeGradNormSquared = getFullNegativeGradNormSquared(sparseNegativeGrad);
        double dotProduct = sparseNegativeGrad.dot(sumNegativeGrads);
        sumNegativeGradNormSq = sumNegativeGradNormSq + fullNegativeGradNormSquared
                + 2 * dotProduct + 2 * sumNegativeGradDotGBatch;
        sumNegativeGradNormSq_Iteration++;

        return sumNegativeGradNormSq;
    }

    public void lazySparseUpdateSumNegativeGrads(SparseVector feature, int updateTo) {
        /**
         * Updates the coordinates of sumNegativeGrads that are present in the feature.
         * Also updates the relevant coordinates of w.
         *
         * REQUIRES ALL SCALAR VALUES UP-TO-DATE
         *
         * Updates to the updateTo`th iteration.
         * Usually want updateTo to be T when used to produce the Tth prediction.
         *
         * For each coordinate:
         * Need to add updateTo - lastUpdated - 1 multiples of negativeBatchGrad to sumNegativeGrads.
         *
         * A subtlety:
         * Because of this lazy updating, sumNegativeGradNormSq and sumNegativeGradDotnegativeBatchGrad
         * will NOT be consistent with the current sumNegativeGrads array. This is because
         * sumNegativeGrads is usually not up-to-date. sumNegativeGradNormSq and sumNegativeGradDotnegativeBatchGrad should
         * always be kept up-to-date.
         */

        assert(sumNegativeGradDotGBatch_Iteration == sumNegativeGradNormSq_Iteration);
        assert(sumNegativeGradDotGBatch_Iteration == updateTo-1);

        double scalarOffset = getOffsetWithoutWeightScaling();
        for (Int2DoubleMap.Entry entry : feature.int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            int missedSteps = updateTo - lastUpdated[key] - 1;
            if (missedSteps < 0) {
                System.out.printf("missed steps negative... updateTo: %d, lastUpdated: %d, key: %d\n", updateTo, lastUpdated[key], key);
            }
            if (missedSteps > 0) {
                sumNegativeGrads[key] += missedSteps * negativeBatchGrad[key];
                lastUpdated[key] = updateTo - 1;
                w[key] = getCoordinateOffset(scalarOffset, key) + center[key];
            }

        }
    }

    public void lazyUpdateSumNegativeGrads(int updateTo) {
        /**
         * updates the coordinate coord of sumNegativeGrads and w to updateTo time step.
         */
        assert(sumNegativeGradDotGBatch_Iteration == sumNegativeGradNormSq_Iteration);
        assert(sumNegativeGradDotGBatch_Iteration == updateTo);
        double scalarOffset = getOffsetWithoutWeightScaling();
        for (int i=0; i<size_hash; i++) {
            int missedSteps = updateTo - lastUpdated[i];
            if(missedSteps > 0) {
                sumNegativeGrads[i] += missedSteps * negativeBatchGrad[i];
                w[i] = getCoordinateOffset(scalarOffset, i) + center[i];
            }
            lastUpdated[i] = updateTo;
        }

    }

    protected void createW() {
        /**
         * REQUIRES FULLY UP-TO-DATE sumNegativeGrads.
         *
         * recreates the entire W vector from scratch.
         */
        if (inverseEtaSq > 1e-7) {
            double scalarOffset = getOffsetWithoutWeightScaling();
            for (int i = 0; i < size_hash; i++) {
                double offset = getCoordinateOffset(scalarOffset, i);
                w[i] = offset + center[i];
            }
        }
    }

    protected double getOffsetWithoutWeightScaling() {
        /**
         * REQUIRES ALL SCALAR VALUES UP-TO-DATE
         */
        double sumNegativeGradNorm = Math.sqrt(sumNegativeGradNormSq);
        double offset = 0.0;
        if (inverseEtaSq > 1e-7 && sumNegativeGradNorm > 1e-7) {
            offset = Math.exp(k_inv * sumNegativeGradNorm / Math.sqrt(inverseEtaSq)) - 1.0;
            offset /= sumNegativeGradNorm;
            if (useScaling) {
                offset /= scaling;
            }
        }

        return offset;
    }

    protected double getCoordinateOffset(double scalarOffset, int key) {
        /**
         * scalarOffset is output of getOffsetWithoutWeightScaling.
         * key is coordinate to get offset for.
         */

        double offset = scalarOffset;
        if (useWeightScaling) {
            if(weightScaling[key] > 1e-7) {
                offset /= weightScaling[key];
            } else {
                offset = 0.0;
            }
        }

        return sumNegativeGrads[key] * offset;
    }

    public void updateFromSparseNegativeGrad(SparseVector sparseNegativeGrad, int iteration) {
        updateSumNegativeGradNormSq(sparseNegativeGrad);
        updateSumNegativeGradDotNegativeBatchGrad(sparseNegativeGrad);

        double gradNormSquared = getFullNegativeGradNormSquared(sparseNegativeGrad);
        double sumNegativeGradNorm = Math.sqrt(sumNegativeGradNormSq);



        maxNegativeGrad = Math.max(maxNegativeGrad, Math.sqrt(gradNormSquared));
        inverseEtaSq = inverseEtaSq + 2 * gradNormSquared;//Math.max(inverseEtaSq + 2 * gradNormSquared, maxNegativeGrad * sumNegativeGradNorm);


        if (useScaling && maxNegativeGrad > 1e-8) {
            scaling = Math.max(scaling, inverseEtaSq/(maxNegativeGrad * maxNegativeGrad));
        }

        double scalarOffset = getOffsetWithoutWeightScaling();

        for (Int2DoubleMap.Entry entry :  sparseNegativeGrad.int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            double negativeGrad_i = entry.getDoubleValue() + negativeBatchGrad[key];

            sumNegativeGrads[key] += negativeGrad_i;
            lastUpdated[key] = iteration;

            double offset = getCoordinateOffset(scalarOffset, key);

            w[key] = offset + center[key];
        }


    }




    public double update(Instance sample) {
        iter++;

        double pred = predict(sample);

        final double negativeGrad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());
        updateWeightScaling(sample);

        updateFromSparseNegativeGrad(sample.getVector().scale(negativeGrad), iter);

        return pred;
    }

    public void updateWeightScalingVector( double[] sc ) {
        for(int i=0; i< size_hash; i++) {
            if (sc[i] > weightScaling[i] )
                weightScaling[i] = sc[i];
        }
    }

    public void updateWeightScaling( Instance sample ) {
        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            double value = Math.abs(entry.getDoubleValue());
            if (value > weightScaling[key] )
                weightScaling[key] = value;
        }
    }


    public double predict(Instance sample) {
        return sample.getVector().dot(w);
    }

    public Loss getLoss() {
        return lossFnc;
    }

    public SparseVector getWeights() {
        return SparseVector.dense2Sparse(w);
    }

    public void setLoss(Loss lossFnc) {
        this.lossFnc = lossFnc;
    }

    public void setLearningRate(double k_inv) {
        this.k_inv = k_inv;
    }

    public void useScaling(boolean flag) {
        if (this.useWeightScaling && flag) {
            System.out.println("Scaling and weight scaling cannot be used together! ");
            System.exit(-1);
        }
        this.useScaling = flag;
    }

    public void useWeightScaling(boolean flag ){
        if (this.useScaling && flag) {
            System.out.println("Scaling and weight scaling cannot be used together! ");
            System.exit(-1);
        }
        this.useWeightScaling = flag;
    }

    /**
     * The following code is hopefully useful for an SVRG implementation.
     */
    public double[] getDenseWeights() {
        // useful for extracting w_prev in SVRG
        return w;
    }


    public String toString() {
        String tmp = "Using FreeRex optimizer\n";
        tmp = tmp + "learning rate = " + k_inv + "\n";
        tmp = tmp + "Loss function = " + getLoss().toString();
        return tmp;
    }

    private void writeObject(ObjectOutputStream o) throws IOException {
        o.defaultWriteObject();
        //o.writeObject(SparseVector.dense2Sparse(maxNegativeGrads));
        //o.writeObject(SparseVector.dense2Sparse(scaling));
        //o.writeObject(SparseVector.dense2Sparse(sumNegativeGrads));
        //o.writeObject(SparseVector.dense2Sparse(inverseEtaSq));
        //o.writeObject(SparseVector.dense2Sparse(center));
        o.writeObject(SparseVector.dense2Sparse(w));
    }

    private void readObject(ObjectInputStream o) throws IOException, ClassNotFoundException {
        o.defaultReadObject();
        //maxNegativeGrads = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        //scaling = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        //sumNegativeGrads = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        //inverseEtaSq = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        //center = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        w = ((SparseVector) o.readObject()).toDenseVector(size_hash);
    }

}
