package com.yahoo.labs.yamall.ml;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.core.SparseVector;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * Created by busafekete on 7/11/17.
 */
public class SGD implements Learner {
    private double eta = .5;
    private transient double[] w;
    private Loss lossFnc;
    private double iter = 0;
    private int size_hash = 0;

    public SGD(
            int bits) {
        size_hash = 1 << bits;
        w = new double[size_hash];
    }

    public void setLoss(Loss lossFnc) {
        this.lossFnc = lossFnc;
    }

    public void setLearningRate(double eta) {
        this.eta = eta;
    }

    public double update(Instance sample) {
        iter++;

        double pred = 0;
        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            double x_i;
            if ((x_i = entry.getDoubleValue()) != 0.0) {
                int key = entry.getIntKey();
                double w_i = w[key];
                pred += w_i * x_i;
            }
        }

        final double negativeGrad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());

        if (Math.abs(negativeGrad) > 1e-8) {
            final double a = eta * Math.sqrt(1 / (double)iter) ;

            for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
                double x_i;
                if ((x_i = entry.getDoubleValue()) != 0.0) {
                    int key = entry.getIntKey();
                    w[key] += a * negativeGrad * x_i;
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

    public String toString() {
        String tmp = "Using SGD\n";
        tmp = tmp + "Initial learning rate = " + eta + "\n";
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
