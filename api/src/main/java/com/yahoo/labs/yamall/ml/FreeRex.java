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
 * created by acutkosky on 7/20/2017
 */

@SuppressWarnings("serial")
public class FreeRex implements DenseSVRGLearner {
    private transient double maxGradNorm = 0;
    private transient double inverseEtaSq = 0;
    private transient double[] negativeGradSum;
    private transient double[] weightScaling;
    private double scaling = 1.0;
    private transient double[] center;
    private transient double[] w;

    private boolean useScaling = false;
    private boolean useWeightScaling = true;

    private double k_inv = 0.45; // sqrt(1/5)
    private Loss lossFnc;
    public int iterations = 0;
    private int size_hash = 0;

    private double[] negativeBatchGrad;
    private double batchGradDotGradSum = 0;
    private int batchGradDotGradSum_Iterations = 0;
    private double batchGradNormSquared = 0;
    private double gradSumNormSquared = 0;

    private int[] lastUpdated;

    public FreeRex(
            int bits) {
        size_hash = 1 << bits;

        negativeGradSum = new double[size_hash];
        negativeBatchGrad = new double[size_hash];
        lastUpdated = new int[size_hash];

        // probably in practice we don't ever need the following two arrays
        weightScaling = new double[size_hash];

        w = new double[size_hash];
        center = new double[size_hash]; //default 0, but allows for FTRL with arbitrary centering.

    }

    public double[] getDenseWeights() {
        return w;
    }

    public void setCenter(double center[]) {
        for (int i = 0; i < size_hash; i++) {
            this.center[i] = center[i];
        }
    }

    public void reset() {
        for (int i = 0; i < size_hash; i++) {
            this.inverseEtaSq = 0.0;
            this.negativeGradSum[i] = 0.0;
            this.w[i] = center[i];
        }
    }


    /**
     * S = sum of gradients.
     * S is going to update to S + g + G where g is sparse (given by negativeGrad), and G is dense
     * (given by negativeBatchGrad).
     * We want to compute the new value of S * G, given the old value and the value of G*G
     * This is simply
     * S * G + g * G + G * G
     * <p>
     * THIS SHOULD BE CALLED AFTER updateGradSumNormSquared
     *
     * @param negativeGrad
     */
    private double updateBatchGradDotGradSum(SparseVector negativeGrad) {
        batchGradDotGradSum_Iterations++;
        batchGradDotGradSum += negativeGrad.dot(negativeBatchGrad) + batchGradNormSquared;
        return batchGradDotGradSum;
    }

    /**
     * We want to compute (g + G)^2 where g is sparse, given by negativeGrad, and G^2 is known.
     * This is
     * g^2 + 2g*G + G^2
     *
     * @param negativeGrad
     * @return
     */
    private double getGradientNormSquared(SparseVector negativeGrad) {
        return negativeGrad.squaredL2Norm() + 2 * negativeGrad.dot(negativeBatchGrad) + batchGradNormSquared;
    }

    /**
     * S = sum of gradients
     * S is going to update to S + g + G where g is sparse and G is dense.
     * We want to compute the new value of S^2 given the old value, the value of S*G and (g+G)^2
     * This is
     * S^2 + (g+G)^2 + 2S*g + 2S*G
     * <p>
     * THIS SHOULD BE CALLED BEFORE updateBatchGradDotGradSum
     *
     * @param negativeGrad
     */
    private double updateGradSumNormSquared(SparseVector negativeGrad) {
        gradSumNormSquared += getGradientNormSquared(negativeGrad) +
                2 * negativeGrad.dot(negativeGradSum) +
                2 * batchGradDotGradSum;
        return gradSumNormSquared;
    }

    public void setNegativeBatchGrad(double [] gBatch) {
        batchGradDotGradSum = 0;
        batchGradNormSquared = 0;
        for(int i=0; i<size_hash; i++) {
            negativeBatchGrad[i] = gBatch[i];
            batchGradNormSquared += gBatch[i] * gBatch[i];
            batchGradDotGradSum += gBatch[i] * negativeGradSum[i];
        }
    }

    private void updateGradSum(SparseVector negativeGrad) {
        for (Int2DoubleMap.Entry entry : negativeGrad.int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            int missedSteps = iterations - lastUpdated[key];
            negativeGradSum[key] += missedSteps * negativeBatchGrad[key] + entry.getDoubleValue();
            lastUpdated[key] = iterations;
        }
    }


    public double update(Instance sample) {
        if(useWeightScaling) {
            updateScalingVector(sample);
        }
        lazySparseUpdate(sample.getVector());
        double pred = predict(sample);

        final double negativeGrad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());
        int[] keys = new int[sample.getVector().size()];
        double[] values = new double[sample.getVector().size()];
        int i = 0;
        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            double negativeGrad_i = entry.getDoubleValue() * negativeGrad;
            keys[i] = entry.getIntKey();
            values[i] = negativeGrad_i;
            i++;
        }
        updateFromNegativeGrad(sample, new SparseVector(keys, values));

        return pred;
    }

    /*
    This scaling is similar to the one applied in normalized gradient descent, see SGD_VW
     */
    public void updateScalingVector(Instance sample) {
        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            double value = Math.abs(entry.getDoubleValue());
            if (value > weightScaling[key])
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

    public void useWeightScaling(boolean flag) {
        if (this.useScaling && flag) {
            System.out.println("Scaling and weight scaling cannot be used together! ");
            System.exit(-1);
        }
        this.useWeightScaling = flag;
    }

    public void updateFromNegativeGrad(Instance sample, SparseVector negativeGrad) {

        if (useWeightScaling)
            updateScalingVector(sample);

        iterations++;
        updateGradSumNormSquared(negativeGrad);
        updateBatchGradDotGradSum(negativeGrad);
        updateGradSum(negativeGrad);
        double gradNormSquared = getGradientNormSquared(negativeGrad);
        double gradSumNorm = Math.sqrt(gradSumNormSquared);

        maxGradNorm = Math.max(Math.sqrt(gradNormSquared), maxGradNorm);

        inverseEtaSq = Math.max(inverseEtaSq + 2 * gradNormSquared, maxGradNorm * gradSumNorm);
        if (useScaling) {
            scaling = Math.max(scaling, inverseEtaSq / (maxGradNorm * maxGradNorm));
        }

        for (Int2DoubleMap.Entry entry : negativeGrad.int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            double offset = getOffset(key);
            w[key] = center[key] + offset;
        }


    }

    private double getOffset(int key) {
        double gradSumNorm = Math.sqrt(gradSumNormSquared);
        double offset = 0;
        if(inverseEtaSq > 1e-8) {
            offset = negativeGradSum[key] / gradSumNorm * (Math.exp(k_inv * gradSumNorm / Math.sqrt(inverseEtaSq)) - 1.0);
            if (useWeightScaling)
                offset /= weightScaling[key];

            if (useScaling)
                offset /= scaling;
        }
        return offset;
    }

    public void lazySparseUpdate(SparseVector sampleVector) {
        for (Int2DoubleMap.Entry entry : sampleVector.int2DoubleEntrySet()) {
            int i = entry.getIntKey();
            double offset = getOffset(i);
            w[i] = center[i] + offset;
        }
    }

    public void lazyUpdate() {
        for (int i = 0; i < size_hash; i++) {
            double offset = getOffset(i);
            w[i] = center[i] + offset;
        }
    }


    public String toString() {
        String tmp = "Using FreeRex optimizer\n";
        tmp = tmp + "learning rate = " + k_inv + "\n";
        tmp = tmp + "Loss function = " + getLoss().toString();
        return tmp;
    }

    private void writeObject(ObjectOutputStream o) throws IOException {
        o.defaultWriteObject();
        o.writeObject(SparseVector.dense2Sparse(weightScaling));
        o.writeObject(SparseVector.dense2Sparse(negativeGradSum));
        o.writeObject(SparseVector.dense2Sparse(center));
        o.writeObject(SparseVector.dense2Sparse(w));
    }

    private void readObject(ObjectInputStream o) throws IOException, ClassNotFoundException {
        o.defaultReadObject();
        weightScaling = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        negativeGradSum = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        center = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        w = ((SparseVector) o.readObject()).toDenseVector(size_hash);
    }

}