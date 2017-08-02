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

    private boolean useScaling = false;
    private boolean useWeightScaling = false;

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

    public void reset() {
        for(int i=0; i< size_hash; i++) {
            //maxGrads[i] = 0;
            inverseEtaSq[i] = 0;
            sumGrads[i] = 0;
            if (this.useScaling )
                scaling[i] = 1.0;

            w[i] = center[i];
        }
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
                    double offset = (Math.signum(sumGrads_i)) * (Math.exp(k_inv * Math.abs(sumGrads_i) / Math.sqrt(inverseEtaSq_i)) - 1.0);

                    if (useScaling) {
                        double scaling_i = scaling[key];
                        scaling_i = Math.max(scaling_i, inverseEtaSq_i / (maxGrads_i * maxGrads_i));
                        scaling[key] = scaling_i;
                        if (scaling_i > 0.0)
                            offset /= scaling_i;
                    }

                    if (useWeightScaling) {
                        double scaling_i = scaling[key];
                        double absValue = Math.abs(entry.getDoubleValue());
                        if (absValue > scaling_i) {
                            scaling_i = absValue;
                            scaling[key] = scaling_i;
                        }
                        if (scaling_i > 0.0)
                            offset /= scaling_i;
                    }

                    double update = offset + center[key];

                    w[key] = update;
//                    if (Double.isInfinite(update)){
//                        System.out.printf( "key: %d\n", key);
//                        System.out.printf( "inverseEtaSq_i: %f\n", inverseEtaSq_i );
//                        System.out.printf( "sumGrads_i: %f\n", sumGrads_i );
//                    }
                }

            }
        }

        return pred;
    }

    public void updateScaleingVector( double[] sc ) {
        for(int i=0; i< size_hash; i++) {
            if (sc[i] > scaling[i] )
                scaling[i] = sc[i];
        }
    }

    public void updateScaleingVector( Instance sample ) {
        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            double value = Math.abs(entry.getDoubleValue());
            if (value > scaling[key] )
                scaling[key] = value;
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

    public void batch_update_coord(int key, double negativeGrad, int missed_steps) {
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
                double offset = (Math.signum(sumGrads_i)) * (Math.exp(k_inv * Math.abs(sumGrads_i) / Math.sqrt(inverseEtaSq_i)) - 1.0);

                if (useScaling) {
                    double scaling_i = scaling[key];
                    scaling_i = Math.max(scaling_i, inverseEtaSq_i / (maxGrads_i * maxGrads_i));
                    scaling[key] = scaling_i;
                    if (scaling_i > 0.0)
                        offset /= scaling[key];
                }

                if (useWeightScaling) {
                    double scaling_i = scaling[key];
                    if (scaling_i > 0.0)
                        offset /= scaling_i;
                }


                double update = offset + center[key];

                w[key] = update;
            }


        }
    }

    public void updateFromNegativeGrad(SparseVector negativeGrad) {
        // potentially useful for SVRG updates eventually.
        for (Int2DoubleMap.Entry entry : negativeGrad.int2DoubleEntrySet()) {
            double negativeGrad_i = entry.getDoubleValue();

            batch_update_coord(entry.getIntKey(), negativeGrad_i, 1);
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
