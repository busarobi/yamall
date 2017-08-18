// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.labs.yamall.ml;

import java.io.Serializable;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.core.SparseVector;

public interface DenseSVRGLearner extends Learner {

    /**
     * Performs an update directly from a loss gradient rather than computing
     * the gradient on a sample.
     * @param sample
     * @param negativeGrad
     */
    void updateFromNegativeGrad(Instance sample, SparseVector negativeGrad);


    /**
     * resets the learner's internal state so that the next prediction will be from the starting weight vector.
     */
    void reset();

    /**
     * sets the learner's starting weight vector.
     */
    void setCenter(double center[]);

    /**
     * return weights as a dense vector rather than a sparse one.
     * @return
     */
    double[] getDenseWeights();

    void setNegativeBatchGrad(double[] gBatch);

    /**
     * performs missed updates
     */
    void lazyUpdate();

    public void updateScalingVector(Instance sample);
}
