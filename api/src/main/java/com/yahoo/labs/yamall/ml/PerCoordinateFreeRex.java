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
public class PerCoordinateFreeRex implements SVRGLearner {
    private double[] maxGrads;
    private double[] inverseEtaSq;
    private double[] sumGrads;
    private double[] weightScaling;
    private double[] regretScaling;
    private double[] center;
    private double[] w;

    private double guessedMaxGrad = 1.0;

    private boolean useRegretScaling = false;
    private boolean useWeightScaling = true;

    private double k_inv = 0.45; // sqrt(1/5)
    private Loss lossFnc;
    public double iter = 0;
    private int size_hash = 0;

    public PerCoordinateFreeRex(
            int bits) {
        size_hash = 1 << bits;
        inverseEtaSq = new double[size_hash];
        sumGrads = new double[size_hash];

        regretScaling = new double[size_hash];
        weightScaling = new double[size_hash];
        maxGrads = new double[size_hash];

        w = new double[size_hash];
        center = new double[size_hash]; //default 0, but allows for FTRL with arbitrary centering.

    }

    public void setGuessedMaxGrad(double guess) {
        guessedMaxGrad = guess;
    }

    public double[] getDenseWeights() {
        return w;
    }
    public void setCenter(double center[]) {
        for (int i=0; i<size_hash; i++) {
            this.center[i] = center[i];
        }
    }

    public void reset() {
        for (int i=0; i<size_hash; i++) {
            this.sumGrads[i] = 0.0;
            this.w[i] = center[i];
        }
    }

    public double update(Instance sample) {
        iter++;

        if (this.useWeightScaling)
            updateScalingVector(sample.getVector());

        double pred = predict(sample);

        final double negativeGrad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());

        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            double negativeGrad_i = entry.getDoubleValue() * negativeGrad;
            batchUpdateCoordinate(entry.getIntKey(),negativeGrad_i, 1);
        }

        return pred;
    }

    /*
    This scaling is similar to the one applied in normalized gradient descent, see SGD_VW
     */
    public void updateScalingVector(SparseVector featureVector) {
        for (Int2DoubleMap.Entry entry : featureVector.int2DoubleEntrySet()) {
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

    public void useRegretScaling(boolean flag) {
        this.useRegretScaling = flag;
    }

    public void useWeightScaling(boolean flag ){
        this.useWeightScaling = flag;
    }

    public void updateFromNegativeGrad(SparseVector featureVector, SparseVector negativeGrad) {
        updateScalingVector(featureVector);
        for (Int2DoubleMap.Entry entry : negativeGrad.int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            double negativeGrad_i = entry.getDoubleValue();
            batchUpdateCoordinate(key, negativeGrad_i, 1);
        }

    }

    double getOffset(int key) {
        double initEta = guessedMaxGrad;
        if (useWeightScaling) {
            initEta *= weightScaling[key];
        }
        double inverseEta_i = Math.sqrt(inverseEtaSq[key] + initEta);
        double sumGrads_i = sumGrads[key];

        if (Math.abs(inverseEta_i) < 1e-7) {
            return 0.0;
        }

        double offset = (Math.signum(sumGrads_i)) * (Math.exp(k_inv * Math.abs(sumGrads_i) / inverseEta_i) - 1.0);
        if (useWeightScaling) {
            offset /= weightScaling[key];
        }

        if (useRegretScaling) {
            offset /= regretScaling[key];
        }

        return offset;
    }

    public void batchUpdateCoordinate(int key, double negativeGrad, int missed_steps) {
        // useful for SVRG lazy update.
        if (Math.abs(negativeGrad) > 1e-8 && missed_steps > 0) {
            double sumGrads_i = sumGrads[key];
            double maxGrads_i = maxGrads[key];
            double inverseEtaSq_i = inverseEtaSq[key];

            sumGrads_i += missed_steps * negativeGrad;
            maxGrads_i = Math.max(maxGrads_i, Math.abs(negativeGrad));
            inverseEtaSq_i = Math.max(inverseEtaSq_i + 2 * missed_steps * negativeGrad * negativeGrad, Math.abs(sumGrads_i) * maxGrads_i);

            maxGrads[key] = maxGrads_i;
            sumGrads[key] = sumGrads_i;

            inverseEtaSq[key] = inverseEtaSq_i;

            if (inverseEtaSq_i>1e-7) {

                if (useRegretScaling) {
                    double scaling_i = regretScaling[key];
                    scaling_i = Math.max(scaling_i, inverseEtaSq_i / (maxGrads_i * maxGrads_i));
                    regretScaling[key] = scaling_i;
                }

                double offset = getOffset(key);


                double update = offset + center[key];

                w[key] = update;
            }


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
        o.writeObject(SparseVector.dense2Sparse(maxGrads));
//        o.writeObject(SparseVector.dense2Sparse(scaling));
        o.writeObject(SparseVector.dense2Sparse(sumGrads));
        o.writeObject(SparseVector.dense2Sparse(inverseEtaSq));
        o.writeObject(SparseVector.dense2Sparse(center));
        o.writeObject(SparseVector.dense2Sparse(w));
    }

    private void readObject(ObjectInputStream o) throws IOException, ClassNotFoundException {
        o.defaultReadObject();
        maxGrads = ((SparseVector) o.readObject()).toDenseVector(size_hash);
//        scaling = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        sumGrads = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        inverseEtaSq = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        center = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        w = ((SparseVector) o.readObject()).toDenseVector(size_hash);
    }

}