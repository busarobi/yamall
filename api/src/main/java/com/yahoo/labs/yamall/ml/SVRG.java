package com.yahoo.labs.yamall.ml;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.core.SparseVector;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * Created by busafekete on 7/9/17.
 */
public class SVRG implements Learner {
    private boolean averaging = false;

    private static final int GATHER_GRADIENT = 1;
    private static final int UPDATE_GRADIENT = 2;

    private double eta = .01;
    private int step = 500;
    private double lambda = .01;

    private int backCounter = 0;
    private int gradStep = 0;

    private transient double[] w;

    private transient double[] Gbatch;
    private transient double[] w_prev;
    private transient double[] w_avg;

    private double N = 0;
    private Loss lossFnc;
    private double iter = 0;
    private int size_hash = 0;

    private int gatherGradIter = 0;

    private int state = 2;

    public SVRG(
            int bits) {
        size_hash = 1 << bits;
        w = new double[size_hash];

        lambda = 0.1/Math.sqrt(step);

        Gbatch = new double[size_hash];
        w_prev = new double[size_hash];
        // we can get rif of this by using online update
        w_avg = new double[size_hash];
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

    public void doAveraging() { this.averaging = true; }

    public void setStep(int step) {
        this.step = step;
    }

    private double accumulateGradient( Instance sample ) {
        gatherGradIter++;

        double pred = 0;
        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            double x_i;
            if ((x_i = entry.getDoubleValue()) != 0.0) {
                int key = entry.getIntKey();
                double w_i = w[key];
                pred += w_i * x_i;
            }
        }

        final double grad = -lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());

        if (Math.abs(grad) > 1e-8) {

            for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
                double x_i;
                if ((x_i = entry.getDoubleValue()) != 0.0) {
                    int key = entry.getIntKey();
                    double G_i = Gbatch[key];
                    G_i += (grad * x_i);
                    Gbatch[key] = G_i;
                }
            }
        }
        return pred;
    }

    public double gradStep( Instance sample ){
        double pred = 0;
        double pred_prev = 0;
        gradStep++;

        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            double x_i;
            if ((x_i = entry.getDoubleValue()) != 0.0) {
                int key = entry.getIntKey();

                double w_i = w[key];
                pred += w_i * x_i;

                double w_i_prev = w_prev[key];
                pred_prev += w_i_prev * x_i;

            }
        }

        final double grad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());
        final double grad_prev = lossFnc.negativeGradient(pred_prev, sample.getLabel(), sample.getWeight());
        final double grad_diff = grad_prev - grad;

        if (Math.abs(grad) > 1e-8) {

            for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
                double x_i;
                if ((x_i = entry.getDoubleValue()) != 0.0) {
                    int key = entry.getIntKey();
                    double Gb = Gbatch[key];
                    if(lambda==0.0)
                        w[key] += this.eta * ( grad_diff * x_i - Gb );
                    else {   // regularization
                        double wi = w[key];
                        double wi_prev = w_prev[key];
                        w[key] += this.eta * ( grad_diff * x_i - Gb - lambda * ( wi - wi_prev ) );
                    }

                    if (this.averaging ){
                        w_avg[key] += (w[key] - w_avg[key])/((double)gradStep);
                    }
                }
            }
        }
        return pred;
    }

    private void initGatherState() {

        if (this.averaging ) {
            for (int i = 0; i < size_hash; i++) w[i] = w_avg[i];
        }

        for (int i=0; i < size_hash; i++ ) w_prev[i] = w[i];
        for (int i=0; i < size_hash; i++ ) Gbatch[i] = 0;
        gatherGradIter = 0;
    }

    private void normalizeBathGradient() {
        for (int i=0; i < size_hash; i++ ) Gbatch[i] /= (double)gatherGradIter;

        gradStep = 0;
        if (this.averaging ) {
            for (int i = 0; i < size_hash; i++) w_avg[i] = w[i];
        }
    }

    public double update(Instance sample) {
        iter++;

        alterState();

        double pred = 0;
        if (state == SVRG.GATHER_GRADIENT) {
            pred = this.accumulateGradient(sample);
        } else if (state == SVRG.UPDATE_GRADIENT) {
            pred = this.gradStep(sample);
        }


        return pred;
    }

    private void alterState() {
        backCounter--;
        if ( backCounter <= 0  ) {
            if (state == SVRG.GATHER_GRADIENT ){ // switch to update parameters
                backCounter = step;
                normalizeBathGradient();
                state = SVRG.UPDATE_GRADIENT;
            } else if ( state == SVRG.UPDATE_GRADIENT ) { // switch to gather gradient
                backCounter = step;
                initGatherState();
                state = SVRG.GATHER_GRADIENT;
            }
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
        o.writeObject(SparseVector.dense2Sparse(w_prev));
    }

    private void readObject(ObjectInputStream o) throws IOException, ClassNotFoundException {
        o.defaultReadObject();
        w = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        Gbatch = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        w_prev = ((SparseVector) o.readObject()).toDenseVector(size_hash);
    }
}
