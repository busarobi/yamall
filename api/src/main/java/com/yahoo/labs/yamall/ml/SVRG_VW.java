package com.yahoo.labs.yamall.ml;

import com.yahoo.labs.yamall.core.Instance;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

/**
 * Created by busafekete on 7/21/17.
 */
public class SVRG_VW extends SVRG {
    private transient double[] theta;
    private transient double[] sumSqGrads;
    private transient double[] s;
    private double N = 0;
    private double totalGradIter = 0;

    public SVRG_VW(int bits) {
        super(bits);

        theta = new double[size_hash];
        sumSqGrads = new double[size_hash];
        s = new double[size_hash];
    }

    public double accumulateGradient( Instance sample ) {
        gatherGradIter++;

        double pred = predict(sample);

        final double grad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());

        if (Math.abs(grad) > 1e-8) {
            sample.getVector().addScaledSparseVectorToDenseVector(Gbatch, grad);
        }
        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            if (s[key] < Math.abs(entry.getDoubleValue()))
                s[key] = Math.abs(entry.getDoubleValue());
        }
        return pred;
    }

    public double gradStep( Instance sample ){
        double pred = 0;
        double pred_prev = 0;
        gradStep++;

        totalGradIter++;

        if (lambda != 0.0) {
            System.exit(-1);
        }



        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();

            //update s and N
            double s_i = s[key];
            double x_i = entry.getDoubleValue();

            s_i = Math.max(s_i,Math.abs(x_i));

            s[key] = s_i;
            N += x_i*x_i / s_i*s_i;

            // deal with missed steps
            int missed_steps = gradStep - last_updated[key] - 2;

            if (missed_steps > 0) {
                theta[key] += missed_steps * Gbatch[key];
                sumSqGrads[key] += (missed_steps * Gbatch[key] * Gbatch[key]);

                if ( sumSqGrads[key]> 0.0) {
                    w[key] = eta * Math.sqrt( (totalGradIter - 1) / (N * s[key] * s[key]) ) * theta[key] / Math.sqrt(sumSqGrads[key]);
                    last_updated[key] = gradStep;
                }
            }
        }


        pred = predict(sample);
        pred_prev = predict_prev(sample);

        final double grad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());
        final double grad_prev = lossFnc.negativeGradient(pred_prev, sample.getLabel(), sample.getWeight());
        final double grad_diff = grad - grad_prev;


        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();

            double x_i = entry.getDoubleValue();
            double term = (x_i * grad_diff + Gbatch[key]);
            theta[key] +=  term;
            sumSqGrads[key] += (term * term);

            if ( sumSqGrads[key]> 0.0) {
                w[key] = eta * Math.sqrt( (totalGradIter) / (N * s[key]* s[key]) ) * theta[key] / Math.sqrt(sumSqGrads[key]);
                last_updated[key] = gradStep;
            }
        }

        return pred;
    }

    protected void initGatherState() {
        for (int i = 0; i < size_hash; i++) {
            int missed_steps = gradStep - last_updated[i];
            if (missed_steps > 0) {
                theta[i] += missed_steps * Gbatch[i];
                sumSqGrads[i] += (missed_steps * Gbatch[i] * Gbatch[i]);

                if ( sumSqGrads[i]> 0.0) {
                    w[i] = eta * Math.sqrt( (totalGradIter) / (N * s[i] * s[i]) ) * theta[i] / Math.sqrt(sumSqGrads[i]);
                    last_updated[i] = gradStep;
                }
            }
        }


        for (int i=0; i < size_hash; i++ ) w_prev[i] = w[i];
        for (int i=0; i < size_hash; i++ ) Gbatch[i] = 0;
        gatherGradIter = 0;
    }



}
