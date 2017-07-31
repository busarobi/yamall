package com.yahoo.labs.yamall.ml;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.core.SparseVector;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

/**
 * Created by busafekete on 7/25/17.
 */
public class SVRG_FR extends SVRG {
    protected PerCoordinateFreeRex freerex = null;
    protected static final int BURN_IN = 0;

    public SVRG_FR(int bits) {
        super(bits);
        this.freerex = new PerCoordinateFreeRex(bits);
        //this.freerex.use_scaling(true);
        state=0;
        backCounter=-2;
    }

    public int getGradientStep(){
        return (int) (step/10.0);
    }

    public double freeRexUpdate(Instance instance ) {
        return this.freerex.update(instance);
    }

    public double gradStep( Instance sample ){
        double pred = 0;
        double pred_prev = 0;
        gradStep++;

        if (lambda != 0.0) {
            System.exit(-1);
        }

        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            int missed_steps = gradStep - last_updated[key] - 1;

            if (missed_steps > 0) {
                freerex.batch_update_coord( key, Gbatch[key], missed_steps);
                last_updated[key] = gradStep;
            }
        }


        pred = predict(sample);
        pred_prev = predict_prev(sample);

        final double grad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());
        final double grad_prev = lossFnc.negativeGradient(pred_prev, sample.getLabel(), sample.getWeight());
        final double grad_diff = grad - grad_prev;

        int[] skeys = new int[sample.getVector().size()];
        double[] svalues = new double[sample.getVector().size()];
        int si = 0;
        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();

            double x_i = entry.getDoubleValue();
            double term = (x_i * grad_diff + Gbatch[key]);

            skeys[si] = key;
            svalues[si] = term;
            si++;

            last_updated[key] = gradStep;
        }

        freerex.updateFromNegativeGrad(new SparseVector(skeys, svalues));
        return pred;
    }

    public void setGBatch( double[] arr ){
        for (int i=0; i < size_hash; i++ ) Gbatch[i] = arr[i];
    }


    public double update(Instance sample) {
        iter++;

        if (backCounter<0)
            backCounter = getGradientStep();

        alterState();

        double pred = 0;
        if (state == SVRG.GATHER_GRADIENT) {
            pred = this.accumulateGradient(sample);
        } else if (state == SVRG.UPDATE_GRADIENT) {
            pred = this.gradStep(sample);
        } else if (state == SVRG_FR.BURN_IN) {
            pred = this.freerex.update(sample);
        }



        return pred;
    }

    private void alterState() {
        backCounter--;
        if ( backCounter <= 0  ) {
            if (state == SVRG_FR.BURN_IN) {
                backCounter = step;
                initGatherState();
                state = SVRG.GATHER_GRADIENT;
            } else if (state == SVRG.GATHER_GRADIENT ){ // switch to update parameters
                //backCounter = (int) Math.sqrt((double)step);
                backCounter = getGradientStep();
                normalizeBatchGradient();
                state = SVRG.UPDATE_GRADIENT;
            } else if ( state == SVRG.UPDATE_GRADIENT ) { // switch to gather gradient
                backCounter = step;
                initGatherState();
                state = SVRG.GATHER_GRADIENT;
            }
        }
    }

    public void initGatherState() {
        double wsum = freerex.getWeights().squaredL2Norm();
        for (int i = 0; i < size_hash; i++) {
            int missed_steps = gradStep - last_updated[i];
            if(missed_steps > 0) {
                freerex.batch_update_coord(i, Gbatch[i], missed_steps);
            }
            last_updated[i] = 0;
        }

        double[] w_tmp = freerex.getDenseWeights();
        for (int i=0; i < size_hash; i++ ) w_prev[i] = w_tmp[i];
        for (int i=0; i < size_hash; i++ ) Gbatch[i] = 0;
        gatherGradIter = 0;
        gradStep = 0;
        // centering
        this.freerex.setCenter(w_prev);
        this.freerex.reset();
    }

    public double[] getDenseWeights() {
        return freerex.getDenseWeights();
    }

    public double predict( Instance sample ){
        return freerex.predict(sample);
    }



    public void setLoss(Loss lossFnc) {
        this.lossFnc = lossFnc;
        this.freerex.setLoss(lossFnc);
    }

    public void setLearningRate(double eta) {
        this.eta = eta;
        this.freerex.setLearningRate(eta);
    }
}
