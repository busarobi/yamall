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
 * created by acutkosky on 8/15/2017
 */

@SuppressWarnings("serial")
public class SVRG implements Learner {

    private static final int BURN_IN = 0;
    private static final int GATHER_GRADIENT = 1;
    private static final int SGD_PHASE = 2;

    private int state = 0;
    private transient double[] negativeBatchGradient;
    private double[] w;
    private transient double[] w_previous;

    private transient int [] lastUpdated;

    private int gatherGradientIter = 0;
    private int SGDIter = 0;
    private int BurnInIter = 0;

    private double batchGradVariance = 0.0; //this is probably useful for testing

    private Loss lossFnc;
    public double iter = 0;
    private int size_hash = 0;

    private double eta = 0.01;
    private int batchSize = 1000;

    public SVRG(int bits) {
        size_hash = 1<< bits;

        negativeBatchGradient = new double[size_hash];
        w = new double[size_hash];
        w_previous = new double[size_hash];
        lastUpdated = new int[size_hash];
    }

    private int getSGDPhaseLength() {
        return batchSize/10;
    }

    private int getBurnInLength() {
        return batchSize;
    }

    private double updateBatchGradient(Instance sample) {

        gatherGradientIter++;

        double pred_prev = predict_previous(sample);

        double negativeGrad = lossFnc.negativeGradient(pred_prev, sample.getLabel(), sample.getWeight());

        if (Math.abs(negativeGrad) > 1e-8) {
            sample.getVector().addScaledSparseVectorToDenseVector(negativeBatchGradient, negativeGrad);
            batchGradVariance += sample.getVector().squaredL2Norm() * negativeGrad * negativeGrad;
        }

        return pred_prev;
    }

    /**
     * update the key'th coordinate of w to the updateTo'th iteration.
     *
     * @param key
     * @param updateTo
     */
    private void lazyUpdate(int key, int updateTo) {
        int missedSteps = updateTo - lastUpdated[key];
        if (missedSteps > 0) {
            w[key] += negativeBatchGradient[key] * missedSteps * eta;
        }
    }

    private void lazyUpdateFromSample(Instance sample, int updateTo) {
        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            lazyUpdate(key, updateTo);
        }
    }

    private double updateBurnIn(Instance sample) {
        BurnInIter ++;
        double pred = predict(sample);
        double negativeGrad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());
        sample.getVector().addScaledSparseVectorToDenseVector(w, negativeGrad * eta);
        return pred;
    }

    private double updateSGDStep(Instance sample) {

        SGDIter++;


        lazyUpdateFromSample(sample, SGDIter - 1);

        double pred = predict(sample);
        double pred_prev = predict_previous(sample);

        double negativeGrad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());
        double negativeGrad_prev = lossFnc.negativeGradient(pred_prev, sample.getLabel(), sample.getWeight());

        for(Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            double x_i = entry.getDoubleValue();
            double varianceReducedNegativeGrad_i = x_i*(negativeGrad - negativeGrad_prev) + negativeBatchGradient[key];
            if (Math.abs(varianceReducedNegativeGrad_i) > 1e-8) {
                w[key] += varianceReducedNegativeGrad_i * eta;
            }
            lastUpdated[key] = SGDIter;
        }

        return pred;
    }

    private void endBatchPhase() {

        double negativeBatchGradientNorm = 0.0;

        for (int i=0; i<size_hash; i++) {
            negativeBatchGradient[i] /= gatherGradientIter;
            negativeBatchGradientNorm += negativeBatchGradient[i]*negativeBatchGradient[i];
            lastUpdated[i] = 0;
        }
        SGDIter = 0;
        batchGradVariance = (batchGradVariance/gatherGradientIter - negativeBatchGradientNorm)/(gatherGradientIter - 1);
    }

    private void endSGDPhase() {

        for (int i=0; i<size_hash; i++) {
            lazyUpdate(i, SGDIter);
            w_previous[i] = w[i];
            negativeBatchGradient[i] = 0.0;
        }
        gatherGradientIter = 0;
        batchGradVariance = 0.0;
    }

    private double chooseActionFromState(Instance sample) {
        double pred = 0;
        switch(state) {
            case BURN_IN:
                pred = updateBurnIn(sample);
                break;
            case GATHER_GRADIENT:
                pred = updateBatchGradient(sample);
                break;
            case SGD_PHASE:
                pred = updateSGDStep(sample);
                break;
        }
        return pred;
    }

    private void endBurnInPhase() {

        for (int i=0; i<size_hash; i++) {
            w_previous[i] = w[i];
            negativeBatchGradient[i] = 0.0;
        }
        gatherGradientIter = 0;
        batchGradVariance = 0.0;
    }

    private void chooseState() {
        switch(state) {
            case BURN_IN:
                if (BurnInIter >= getBurnInLength()) {
                    endBurnInPhase();
                    state = GATHER_GRADIENT;
                }
                break;
            case GATHER_GRADIENT:
                if (gatherGradientIter >= batchSize) {
                    endBatchPhase();
                    state = SGD_PHASE;
                }
                break;
            case SGD_PHASE:
                if (SGDIter >= getSGDPhaseLength()) {
                    endSGDPhase();
                    state = GATHER_GRADIENT;
                }
        }
    }

    public double update(Instance sample) {
        iter++;
        chooseState();
        double pred = chooseActionFromState(sample);
        return pred;
    }


    public void setLoss(Loss lossFnc) {
        this.lossFnc = lossFnc;
    }

    public void setLearningRate(double eta) {
        this.eta = eta;
    }

    public double predict(Instance sample) {
        return sample.getVector().dot(w);
    }

    public double predict_previous(Instance sample) { return sample.getVector().dot(w_previous); }

    public Loss getLoss() {
        return lossFnc;
    }

    public SparseVector getWeights() {
        return SparseVector.dense2Sparse(w);
    }

    public String toString() {
        String tmp = "Using streaming SVRG optimizer\n";
        tmp = tmp + "learning rate = " + eta + "\n";
        tmp = tmp + "batch size = " + batchSize + "\n";
        tmp = tmp + "Loss function = " + getLoss().toString();
        return tmp;
    }

    private void writeObject(ObjectOutputStream o) throws IOException {
        o.defaultWriteObject();
        o.writeObject(SparseVector.dense2Sparse(w));
    }

    private void readObject(ObjectInputStream o) throws IOException, ClassNotFoundException {
        o.defaultReadObject();
        w = ((SparseVector) o.readObject()).toDenseVector(size_hash);
    }
}
