package com.yahoo.labs.yamall.ml;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.core.SparseVector;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

/**
 * Created by acutkosky 8/3/2017.
 */
public class SVRG_FR_B extends SVRG {
    protected FreeRexBatched freerex = null;
    protected static final int BURN_IN = 0;
    protected double batch_grad_norm_sq = 0;
    protected double cumulative_grad_norm_sq = 0;
    protected double gradsum[];
    protected double cumLoss = 0.0;
    protected double miniCumLoss = 0.0;
    protected double cumBatchLoss = 0.0;

    private double minPrediction = -50;
    private double maxPrediction = 50;
    public SVRG_FR_B(int bits) {
        super(bits);
        this.freerex = new FreeRexBatched(bits);
        this.freerex.useScaling(false);
        this.freerex.useWeightScaling(true);
        state=0;
        backCounter=-2;
        gradsum = new double[size_hash];

    }

    public void resetGradSum() {
        for (int i=0; i<size_hash;i ++) {
            gradsum[i] = 0;
        }
        cumulative_grad_norm_sq = 0;
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

        freerex.lazySparseUpdateSumNegativeGrads(sample.getVector(), gradStep);
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
            double term = x_i * grad_diff;
            double new_gradsum = gradsum[key] + x_i * grad;
            cumulative_grad_norm_sq += new_gradsum * new_gradsum - gradsum[key] * gradsum[key];
            gradsum[key] = new_gradsum;


            skeys[si] = key;
            svalues[si] = term;
            si++;

            last_updated[key] = gradStep;
        }
        freerex.updateWeightScaling(sample);
        freerex.updateFromSparseNegativeGrad(new SparseVector(skeys, svalues), gradStep);
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
            double score = Math.min(Math.max(pred, minPrediction), maxPrediction);
            cumBatchLoss += lossFnc.lossValue(score, sample.getLabel()) * sample.getWeight();
        } else if (state == SVRG.UPDATE_GRADIENT) {
            pred = this.gradStep(sample);
            double score = Math.min(Math.max(pred, minPrediction), maxPrediction);

            cumLoss += lossFnc.lossValue(score, sample.getLabel()) * sample.getWeight();
            miniCumLoss += lossFnc.lossValue(score, sample.getLabel()) * sample.getWeight();
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
                System.out.printf( "-->#1 Norm of weight vector: %f\n", this.freerex.getWeights().squaredL2Norm() );
                backCounter = getGradientStep();
                normalizeBatchGradient();
                batch_grad_norm_sq = 0;
                for (int i=0; i < size_hash; i++) {
                    batch_grad_norm_sq += this.Gbatch[i] * this.Gbatch[i];
                }
                System.out.printf( "-->#2 Norm of weight vector: %f\n", this.freerex.getWeights().squaredL2Norm() );
                System.out.printf( "--> Norm of batch vector: %f\n", batch_grad_norm_sq );
                this.freerex.updateWeightScalingVector(this.scale);
                this.freerex.setNegativeBatchGrad(this.Gbatch);
                state = SVRG.UPDATE_GRADIENT;
            } else if ( state == SVRG.UPDATE_GRADIENT ) { // switch to gather gradient
                if (cumLoss/gradStep > cumBatchLoss/step * 0.99 && miniCumLoss/getGradientStep() > cumBatchLoss/step * 0.99) {
                    System.out.printf("Not good enough: cumLoss: %f, miniCumLoss: %f, batch: %f, gradstep %d \n", cumLoss/gradStep, miniCumLoss/getGradientStep(), cumBatchLoss/step, gradStep);
                    backCounter = getGradientStep();
                    miniCumLoss = 0.0;
                } else {
                    System.out.printf("Ending SGD phase: cumLoss: %f, miniCumLoss: %f, batch: %f, gradstep %d \n", cumLoss/gradStep, miniCumLoss/getGradientStep(), cumBatchLoss/step, gradStep);
                    resetGradSum();
                    System.out.printf("-->#3 Norm of weight vector: %f\n", this.freerex.getWeights().squaredL2Norm());
                    step *= 1.1;
                    backCounter = step;
                    initGatherState();
                    state = SVRG.GATHER_GRADIENT;
                    cumBatchLoss = 0.0;
                    cumLoss = 0.0;
                    System.out.printf("-->#4 Norm of weight vector: %f\n", this.freerex.getWeights().squaredL2Norm());
                }
            }
        }
    }

    public void initGatherState() {
        System.out.printf( "-->#5 Norm of weight vector: %f\n", this.freerex.getWeights().squaredL2Norm() );
        int update_count = 0;
        for (int i=0;i<size_hash;i++) {
            update_count+=last_updated[i];
        }
        System.out.printf("last updated count: %d\n",update_count);

        freerex.lazyUpdateSumNegativeGrads(gradStep);
        System.out.printf( "-->#6 Norm of weight vector: %f\n", this.freerex.getWeights().squaredL2Norm() );
        freerex.createW();
        double[] w_tmp = freerex.getDenseWeights();
        for (int i=0; i < size_hash; i++ ) w_prev[i] = w_tmp[i];
        for (int i=0; i < size_hash; i++ ) Gbatch[i] = 0;
        gatherGradIter = 0;
        gradStep = 0;

        // centering
//        this.freerex.resetLastUpdated();
        this.freerex.setCenter(w_prev);
        this.freerex.reset();

    }

    public double[] getDenseWeights() {
        return freerex.getDenseWeights();
    }

    public double predict( Instance sample ){
        return freerex.predict(sample);
    }

    public SparseVector getWeights() {
        return this.freerex.getWeights();
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
