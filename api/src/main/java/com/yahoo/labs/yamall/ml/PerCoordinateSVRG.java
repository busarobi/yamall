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
    protected transient double[] negativeBatchGradient;
    protected transient double[] w_previous;
    private transient double[] batchGradSquared;

    protected transient double[] featureScalings;

    double regularizationScaling = 0.0;

    protected double[] featureCounts;
    protected long totalSamplesSeen = 0;

    protected long[] lastUpdated;

    protected boolean doUseReset = false;

    protected SVRGLearner baseLearner = null;

    protected long gatherGradientIter = 0;
    protected int SGDIter = 0;
    private int BurnInIter = 0;
    private int totalSGDIter = 0;

    //private double batchGradVariance = 0.0; //this is probably useful for testing

    private Loss lossFnc;
    public double iter = 0;
    protected int size_hash = 0;

    private double eta = 0.01;
    private int SGDSize = 1000;

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
        lastUpdated = new long[size_hash];
        batchGradSquared = new double[size_hash];
        featureScalings = new double[size_hash];
    }

    private void useReset(boolean flag) { this.doUseReset = flag; }

    protected int getSGDPhaseLength() {
        return SGDSize;
    }

    private int getBatchLength() {
        return Math.max(SGDSize, totalSGDIter);
    }

    protected int getBurnInLength() {
        return SGDSize;
    }

    private void updateFeatureScaling(SparseVector featureVector) {
        for (Int2DoubleMap.Entry entry : featureVector.int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            double value = entry.getDoubleValue();
            featureScalings[key] = Math.max(featureScalings[key], value);
        }
    }

    private double updateBatchGradient(Instance sample) {

        gatherGradientIter++;

        updateFeatureScaling(sample.getVector());

        double pred_prev = predict_previous(sample);

        double negativeGrad = lossFnc.negativeGradient(pred_prev, sample.getLabel(), sample.getWeight());

        for(Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            double x_i = entry.getDoubleValue();
            double negativeGrad_i = x_i * negativeGrad;
            double currentAverageGrad = negativeBatchGradient[key];
            double currentAverageGradSquare = batchGradSquared[key];
            if(lastUpdated[key] < gatherGradientIter - 1) {
                currentAverageGrad *= ((double) lastUpdated[key] / (double) (gatherGradientIter - 1));
                currentAverageGradSquare *= ((double) lastUpdated[key] / (double) (gatherGradientIter - 1));
            }

            currentAverageGrad += (negativeGrad_i - currentAverageGrad) / gatherGradientIter;
            currentAverageGradSquare += (negativeGrad_i * negativeGrad_i - currentAverageGradSquare) / gatherGradientIter;
            lastUpdated[key] = gatherGradientIter;
            negativeBatchGradient[key] = currentAverageGrad;
            batchGradSquared[key] = currentAverageGradSquare;
        }

        return pred_prev;
    }

    protected double updateBurnIn(Instance sample) {
        BurnInIter ++;
        totalSGDIter ++;

        double pred = baseLearner.update(sample);

        return pred;
    }

    public void setRegularizationScaling(double reg) {
        regularizationScaling = reg;
    }
    public void setSGDSize(int size) {
        this.SGDSize = size;
    }

    protected double updateSGDStep(Instance sample) {

        SGDIter++;
        totalSGDIter++;

        double pred = predict(sample);
        double pred_prev = predict_previous(sample);

        updateBatchGradient(sample);

        double negativeGrad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());
        double negativeGrad_prev = lossFnc.negativeGradient(pred_prev, sample.getLabel(), sample.getWeight());

        int keys[] = new int[sample.getVector().size()];
        double values[] = new double[sample.getVector().size()];
        int i = 0;

        double [] weights = baseLearner.getDenseWeights();

        for(Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            double x_i = entry.getDoubleValue();
            keys[i] = key;
            double varianceReducedNegativeGrad_i = x_i*(negativeGrad - negativeGrad_prev) + negativeBatchGradient[key] * totalSamplesSeen / featureCounts[key];

            double batchGradVariance_i = Math.sqrt((batchGradSquared[key] - negativeBatchGradient[key]*negativeBatchGradient[key])/(gatherGradientIter - 1));
            double regularization_i = regularizationScaling * Math.log(size_hash) * batchGradVariance_i * Math.signum(weights[key]) * totalSamplesSeen / featureCounts[key];
            varianceReducedNegativeGrad_i -= regularization_i;
            if (Math.abs(varianceReducedNegativeGrad_i) > 1e-8) {
                values[i] = varianceReducedNegativeGrad_i;
            }
            i++;
        }

        baseLearner.updateFromNegativeGrad(sample.getVector(), new SparseVector(keys, values));

        return pred;
    }

    private void endBatchPhase() {
        for (int i=0; i<size_hash; i++) {

            if (lastUpdated[i] < gatherGradientIter) {
                negativeBatchGradient[i] *= (double) lastUpdated[i] / (double) gatherGradientIter;
                batchGradSquared[i] *= (double) lastUpdated[i] / (double) gatherGradientIter;
            }
            lastUpdated[i] = gatherGradientIter;
        }
        SGDIter = 0;

        if (doUseReset) {
            baseLearner.setCenter(w_previous);
            baseLearner.reset();
        }

        // perform one full batch grad step. Also update feature scalings.
        baseLearner.updateFromNegativeGrad(SparseVector.dense2Sparse(featureScalings),  SparseVector.dense2Sparse(negativeBatchGradient));
    }

    private void endSGDPhase() {

        double temp[] = baseLearner.getDenseWeights();
        for (int i=0; i<size_hash; i++) {
            w_previous[i] = temp[i];
            negativeBatchGradient[i] = 0.0;
            lastUpdated[i] = 0;

            batchGradSquared[i] = 0.0;
            featureScalings[i] = 0.0;

        }

        gatherGradientIter = 0;
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
            batchGradSquared[i] = 0.0;
        }
        gatherGradientIter = 0;
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
                if (gatherGradientIter >= getBatchLength()) {
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
        tmp = tmp + "batch size = " + SGDSize + "\n";
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
