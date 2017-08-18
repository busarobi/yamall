package com.yahoo.labs.yamall.synthetic.helper;

/**
 * Created by busafekete on 8/17/17.
 */
import java.util.Collections;
import java.util.List;

public class AUC {

    public static double compute(List<Tuple2D> data) {

        if (data.isEmpty())
            return 0;

        Collections.sort(data);
        return sortingAUC(data);
    }

    private static double sortingAUC(List<Tuple2D> overall_set) {
        double last_score = -100.0;
        int first_flag = 0;
        double score = 0.0;
        int pos_count = 0;
        int neg_count = 0;
        int neg_buffer = 0;
        int pos_buffer = 0;
        for (Tuple2D item: overall_set) {

            double prediction = item.predicted;
            int label = item.truth;
            if (prediction != last_score) {
                if (first_flag == 0) {
                    first_flag = 1;
                } else {

                    for (int i = 0; i < pos_buffer; i++) {
                        score = score + 0.5 * neg_buffer + neg_count;
                    }
                    neg_count = neg_count + neg_buffer;
                    pos_count = pos_count + pos_buffer;
                    neg_buffer = 0;
                    pos_buffer = 0;
                }
            }

            if (label != 1)
                neg_buffer = neg_buffer + 1;

            if (label == 1)
                pos_buffer = pos_buffer + 1;

            last_score = prediction;
        }

        if ((pos_buffer !=0) || (neg_buffer != 0)) {
            for (int i = 0; i < pos_buffer; i++) {
                score = score + 0.5 * neg_buffer + neg_count;
            }
        }
        neg_count = neg_count + neg_buffer;
        pos_count = pos_count + pos_buffer;
        return score / (((double)pos_count) * ((double)neg_count));
    }

    public static class Tuple2D implements Comparable<Tuple2D> {
        public double predicted;
        public int truth;

        public Tuple2D(int truth, double pred) {
            this.truth = truth;
            this.predicted = pred;
        }

        public int compareTo(Tuple2D that) {
            return Double.compare(this.predicted, that.predicted);
        }
    }
}
