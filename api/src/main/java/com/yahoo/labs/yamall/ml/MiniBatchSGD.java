package com.yahoo.labs.yamall.ml;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.core.SparseVector;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * Created by busafekete on 7/12/17.
 */

/*
Based on the following article
@article{DekelGSX12,
	Author = {Ofer Dekel and Ran Gilad{-}Bachrach and Ohad Shamir and Lin Xiao},
	Journal = {Journal of Machine Learning Research},
	Pages = {165--202},
	Title = {Optimal Distributed Online Prediction Using Mini-Batches},
	Volume = {13},
	Year = {2012}}
 */

public class MiniBatchSGD implements Learner {
    private static final int GATHER_GRADIENT = 1;
    private static final int UPDATE_GRADIENT = 2;

    private double eta = 1.0;
    private int step = 500;
    private double lambda = .01;

    private int backCounter = 0;


    private transient double[] w;

    private transient double[] Gbatch;

    private Loss lossFnc;
    private double iter;
    private int size_hash = 0;
    private int gatherGradIter = 0;

    public MiniBatchSGD(
            int bits) {
        size_hash = 1 << bits;
        w = new double[size_hash];

        lambda = 0.1/Math.sqrt(step);
        Gbatch = new double[size_hash];
    }

    public void setLoss(Loss lossFnc) {
        this.lossFnc = lossFnc;
    }

    public void setLearningRate(double eta) {
        this.eta = eta;
    }

    public void setRegularizationParameter(double lambda) {
        this.lambda = lambda;
    }

    public void setStep(int step) {
        this.step = step;
    }

    private double accumulateGradient( Instance sample ) {
        gatherGradIter++;

        double pred = predict(sample);

        final double grad = -lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());

        if (Math.abs(grad) > 1e-8) {
            sample.getVector().addScaledSparseVectorToDenseVector(Gbatch, grad);
        }
        return pred;
    }

    public void gradStep(){
        for (int i=0; i < size_hash; i++ ) {
            w[i] -= this.eta * ( Gbatch[i] + lambda * w[i] );
        }
    }

    private void initGatherState() {
        for (int i=0; i < size_hash; i++ ) Gbatch[i] = 0;
        gatherGradIter = 0;
    }

    private void normalizeBathGradient(){
        for (int i=0; i < size_hash; i++ ) Gbatch[i] /= (double)gatherGradIter;
    }

    public double update(Instance sample) {
        iter++;
        backCounter--;
        double pred = this.accumulateGradient(sample);

        if ( backCounter <= 0  ) {
            backCounter = step;
            this.normalizeBathGradient();
            this.gradStep();
            this.initGatherState();
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
        String tmp = "Using SVRG\n";
        tmp = tmp + "Initial learning rate = " + eta + "\n";
        tmp = tmp + "Loss function = " + getLoss().toString();
        return tmp;
    }

    private void writeObject(ObjectOutputStream o) throws IOException {
        o.defaultWriteObject();
        o.writeObject(SparseVector.dense2Sparse(w));
        o.writeObject(SparseVector.dense2Sparse(Gbatch));
    }

    private void readObject(ObjectInputStream o) throws IOException, ClassNotFoundException {
        o.defaultReadObject();
        w = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        Gbatch = ((SparseVector) o.readObject()).toDenseVector(size_hash);
    }

}
