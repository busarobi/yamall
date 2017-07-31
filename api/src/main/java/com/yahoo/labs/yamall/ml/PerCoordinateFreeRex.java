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
public class PerCoordinateFreeRex implements Learner {
    private transient double[] maxGrads;
    private transient double[] inverseEtaSq;
    private transient double[] sumGrads;
    private transient double[] scaling;
    private transient double[] center;
    private transient double[] w;

    private boolean use_scaling = false;
    private double k_inv = 0.45; // sqrt(1/5)
    private Loss lossFnc;
    public double iter = 0;
    private int size_hash = 0;

    public PerCoordinateFreeRex(
            int bits) {
        size_hash = 1 << bits;
        inverseEtaSq = new double[size_hash];
        sumGrads = new double[size_hash];

        // probably in practice we don't ever need the following two arrays
        scaling = new double[size_hash];
        maxGrads = new double[size_hash];

        w = new double[size_hash];
        center = new double[size_hash]; //default 0, but allows for FTRL with arbitrary centering.

    }

    public void setCenter(double[] center) {
        for(int i=0; i < size_hash; i++) {
            this.center[i] = center[i];
            this.w[i] = center[i];
        }
    }

    public void use_scaling(boolean use_scaling) {
        this.use_scaling = use_scaling;
    }

    public void reset() {
        for(int i=0; i< size_hash; i++) {
            //maxGrads[i] = 0;
            inverseEtaSq[i] = 0;
            sumGrads[i] = 0;
            scaling[i] = 0;
            w[i] = center[i];
        }
    }

    public void setLoss(Loss lossFnc) {
        this.lossFnc = lossFnc;
    }

    public void setLearningRate(double k_inv) {
        this.k_inv = k_inv;
    }

    public double update(Instance sample) {
        iter++;

        double pred = predict(sample);

        final double negativeGrad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());

        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            double negativeGrad_i = entry.getDoubleValue() * negativeGrad;

            if (Math.abs(negativeGrad_i) > 1e-8) {
                int key = entry.getIntKey();
                double maxGrads_i = maxGrads[key];
                double sumGrads_i = sumGrads[key];
                double inverseEtaSq_i = inverseEtaSq[key];

                sumGrads_i += negativeGrad_i;
                maxGrads_i = Math.max(maxGrads_i, Math.abs(negativeGrad_i));

                //The max below makes the worst-case bounds a little better. I've never seen the second argument actually be bigger.
                inverseEtaSq_i = Math.max(inverseEtaSq_i + 2.0 * negativeGrad_i * negativeGrad_i, Math.abs(sumGrads_i) * maxGrads_i);

                maxGrads[key] = maxGrads_i;
                sumGrads[key] = sumGrads_i;
                inverseEtaSq[key] = inverseEtaSq_i;

                if (inverseEtaSq_i>1e-7) {
                    double update = (Math.signum(sumGrads_i)) * (Math.exp(k_inv * Math.abs(sumGrads_i) / Math.sqrt(inverseEtaSq_i)) - 1.0) + center[key];
                    w[key] = update;
                    if (Double.isInfinite(update)){
                        System.out.printf( "key: %d\n", key);
                        System.out.printf( "inverseEtaSq_i: %f\n", inverseEtaSq_i );
                        System.out.printf( "sumGrads_i: %f\n", sumGrads_i );
                    }
                }

                if (use_scaling) { //In practice I suspect this is a bad trade-off
                    double scaling_i = scaling[key];
                    scaling_i = Math.max(scaling_i, inverseEtaSq_i / (maxGrads_i * maxGrads_i));
                    scaling[key] = scaling_i;
                    w[key] /= scaling_i;
                }

            }
        }

        return pred;
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


    /**
     * The following code is hopefully useful for an SVRG implementation.
     */
    public double[] getDenseWeights() {
        // useful for extracting w_prev in SVRG
        return w;
    }

    public void batch_update_coord(int key, double negativeGrad, int missed_steps) {
        // useful for SVRG lazy update.
        if (Math.abs(negativeGrad) > 1e-8 && missed_steps > 0) {
            double sumGrads_i = sumGrads[key];
            double maxGrads_i = maxGrads[key];
            double inverseEtaSq_i = inverseEtaSq[key];
            double scaling_i = scaling[key];
            sumGrads_i += missed_steps * negativeGrad;
            maxGrads_i = Math.max(maxGrads_i, Math.abs(negativeGrad));
            inverseEtaSq_i = Math.max(inverseEtaSq_i + 2 * missed_steps * negativeGrad * negativeGrad, Math.abs(sumGrads_i) * maxGrads_i);
            scaling_i = Math.max(scaling_i, inverseEtaSq_i / (maxGrads_i * maxGrads_i));
            maxGrads[key] = maxGrads_i;
            sumGrads[key] = sumGrads_i;
            scaling[key] = scaling_i;
            inverseEtaSq[key] = inverseEtaSq_i;
            if (inverseEtaSq_i>1e-7)
                w[key] = (Math.signum(sumGrads_i)) * ( Math.exp( k_inv *Math.abs(sumGrads_i) / Math.sqrt(inverseEtaSq_i)) - 1.0) + center[key];
            if (use_scaling) { //In practice I suspect this is a bad trade-off
                w[key] /= scaling_i;
            }
        }
    }

    public void updateFromNegativeGrad(SparseVector negativeGrad) {
        // potentially useful for SVRG updates eventually.
        for (Int2DoubleMap.Entry entry : negativeGrad.int2DoubleEntrySet()) {
            double negativeGrad_i = entry.getDoubleValue();

            if (Math.abs(negativeGrad_i) > 1e-8) {
                int key = entry.getIntKey();
                double maxGrads_i = maxGrads[key];
                double sumGrads_i = sumGrads[key];
                double scaling_i = scaling[key];
                double inverseEtaSq_i = inverseEtaSq[key];

                sumGrads_i += negativeGrad_i;
                maxGrads_i = Math.max(maxGrads_i, Math.abs(negativeGrad_i));

                inverseEtaSq_i = Math.max(inverseEtaSq_i + 2 * negativeGrad_i * negativeGrad_i, Math.abs(sumGrads_i) * maxGrads_i);
                scaling_i = Math.max(scaling_i, inverseEtaSq_i / (maxGrads_i * maxGrads_i));

                maxGrads[key] = maxGrads_i;
                sumGrads[key] = sumGrads_i;
                scaling[key] = scaling_i;
                inverseEtaSq[key] = inverseEtaSq_i;
                if (inverseEtaSq_i>1e-7)
                    w[key] = (Math.signum(sumGrads_i)) * ( Math.exp( k_inv * Math.abs(sumGrads_i) / Math.sqrt(inverseEtaSq_i)) - 1.0) + center[key];
                if (use_scaling) { //In practice I suspect this is a bad trade-off
                    w[key] /= scaling_i;
                }
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
        //o.writeObject(SparseVector.dense2Sparse(maxGrads));
        //o.writeObject(SparseVector.dense2Sparse(scaling));
        //o.writeObject(SparseVector.dense2Sparse(sumGrads));
        //o.writeObject(SparseVector.dense2Sparse(inverseEtaSq));
        //o.writeObject(SparseVector.dense2Sparse(center));
        o.writeObject(SparseVector.dense2Sparse(w));
    }

    private void readObject(ObjectInputStream o) throws IOException, ClassNotFoundException {
        o.defaultReadObject();
        //maxGrads = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        //scaling = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        //sumGrads = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        //inverseEtaSq = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        //center = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        w = ((SparseVector) o.readObject()).toDenseVector(size_hash);
    }

}
