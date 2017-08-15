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
public class PerCoordinateSVRG implements Learner {

    private static final int BURN_IN = 0;
    private static final int GATHER_GRADIENT = 1;
    private static final int SGD_PHASE = 2;

    private int state = 0;
    private transient double[] negativeBatchGradient;
    private transient double[] w_previous;

    private double[] featureCounts;
    private int totalSamplesSeen = 0;

    private boolean doUseReset = false;

    SVRGLearner baseLearner = null;

    private int gatherGradientIter = 0;
    private int SGDIter = 0;
    private int BurnInIter = 0;

    private double batchGradVariance = 0.0; //this is probably useful for testing

    private Loss lossFnc;
    public double iter = 0;
    private int size_hash = 0;

    private double eta = 0.01;
    private int batchSize = 1000;

    public PerCoordinateSVRG(int bits) {
        this.baseLearner = new PerCoordinateFreeRex(bits);
        initialize(bits);
    }

    public PerCoordinateSVRG(int bits, SVRGLearner baseLearner) {
        this.baseLearner = baseLearner;
        initialize(bits);
    }

    private void initialize(int bits) {
        size_hash = 1<< bits;

        negativeBatchGradient = new double[size_hash];
        w_previous = new double[size_hash];
        featureCounts = new double[size_hash];
    }

    private void useReset(boolean flag) { this.doUseReset = flag; }

    private int getSGDPhaseLength() {
        return batchSize;
    }

    private int getBurnInLength() {
        return batchSize;
    }

    private double updateBatchGradient(Instance sample) {

        gatherGradientIter++;

        baseLearner.updateFromNegativeGrad(sample, new SparseVector());

        double pred_prev = predict_previous(sample);

        double negativeGrad = lossFnc.negativeGradient(pred_prev, sample.getLabel(), sample.getWeight());

        sample.getVector().addScaledSparseVectorToDenseVector(negativeBatchGradient, negativeGrad);
        batchGradVariance += sample.getVector().squaredL2Norm() * negativeGrad * negativeGrad;


        return pred_prev;
    }

    private double updateBurnIn(Instance sample) {
        BurnInIter ++;

        double pred = baseLearner.update(sample);

        return pred;
    }

    public void setBatchSize(int size) {
        this.batchSize = size;
    }

    private double updateSGDStep(Instance sample) {

        SGDIter++;

        double pred = predict(sample);
        double pred_prev = predict_previous(sample);

        double negativeGrad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());
        double negativeGrad_prev = lossFnc.negativeGradient(pred_prev, sample.getLabel(), sample.getWeight());

        int keys[] = new int[sample.getVector().size()];
        double values[] = new double[sample.getVector().size()];
        int i = 0;

        for(Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            double x_i = entry.getDoubleValue();
            keys[i] = key;
            double varianceReducedNegativeGrad_i = x_i*(negativeGrad - negativeGrad_prev) + negativeBatchGradient[key] * totalSamplesSeen / featureCounts[key];

            if (Math.abs(varianceReducedNegativeGrad_i) > 1e-8) {
                values[i] = varianceReducedNegativeGrad_i;
            }
            i++;
        }

        baseLearner.updateFromNegativeGrad(sample, new SparseVector(keys, values));

        return pred;
    }

    private void endBatchPhase() {

        double negativeBatchGradientNorm = 0.0;
        for (int i=0; i<size_hash; i++) {
            negativeBatchGradient[i] /= gatherGradientIter;
            negativeBatchGradientNorm += negativeBatchGradient[i]*negativeBatchGradient[i];
        }
        SGDIter = 0;
        batchGradVariance = (batchGradVariance/gatherGradientIter - negativeBatchGradientNorm)/(gatherGradientIter - 1);

        if (doUseReset) {
            baseLearner.setCenter(w_previous);
            baseLearner.reset();
        }
    }

    private void endSGDPhase() {

        double temp[] = baseLearner.getDenseWeights();
        for (int i=0; i<size_hash; i++) {
            w_previous[i] = temp[i];
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

        double temp[] = baseLearner.getDenseWeights();
        for (int i=0; i<size_hash; i++) {
            w_previous[i] = temp[i];
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

    private void updateFeatureCounts(Instance sample) {
        totalSamplesSeen ++;
        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            featureCounts[key] += 1;
        }
    }

    public double update(Instance sample) {
        iter++;
        updateFeatureCounts(sample);
        chooseState();
        double pred = chooseActionFromState(sample);
        return pred;
    }


    public void setLoss(Loss lossFnc) {
        this.lossFnc = lossFnc;
        baseLearner.setLoss(lossFnc);
    }

    public void setLearningRate(double eta) {
        this.eta = eta;
        baseLearner.setLearningRate(eta);
    }

    public double predict(Instance sample) {
        return baseLearner.predict(sample);
    }

    public double predict_previous(Instance sample) { return sample.getVector().dot(w_previous); }

    public Loss getLoss() {
        return lossFnc;
    }

    public SparseVector getWeights() {
        return baseLearner.getWeights();
    }

    public String toString() {
        String tmp = "Using per-coordinate SVRG optimizer\n";
        tmp = tmp + "learning rate = " + eta + "\n";
        tmp = tmp + "batch size = " + batchSize + "\n";
        tmp = tmp + "Loss function = " + getLoss().toString();
        return tmp;
    }

    private void writeObject(ObjectOutputStream o) throws IOException {
        o.defaultWriteObject();
        o.writeObject(SparseVector.dense2Sparse(featureCounts));
        o.writeObject(SparseVector.dense2Sparse(w_previous));
    }

    private void readObject(ObjectInputStream o) throws IOException, ClassNotFoundException {
        o.defaultReadObject();
        featureCounts = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        w_previous = ((SparseVector) o.readObject()).toDenseVector(size_hash);
    }
}
