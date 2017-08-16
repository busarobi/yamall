package com.yahoo.labs.yamall.synthetic.data;

import java.util.Random;

/**
 * Created by busafekete on 8/15/17.
 */
public class DataGeneratorNormal implements DataGenerator {
    protected int N = 0;
    protected int i = 0;
    protected int dim = 0;
    protected int sparsity = 0;
    protected double[] w = null;
    protected long seed = 0;

    protected Random rand = null;

    public DataGeneratorNormal(int N, int dim, int sparsity) {
        this.N = N;
        this.dim = dim;
        this.sparsity = sparsity;
        this.rand = new Random();

        this.w = new double[this.dim];
        for(int i =0; i < this.dim; i++ ) this.w[i] = rand.nextGaussian();
    }

    public DataGeneratorNormal(int N, int dim, int sparsity, long seed ){
        this.N = N;
        this.dim = dim;
        this.sparsity = sparsity;
        this.seed = seed;
        this.rand = new Random();
        this.rand.setSeed(this.seed);

        this.w = new double[this.dim];
        for(int i =0; i < this.dim; i++ ) this.w[i] = rand.nextGaussian();
    }

    public void setNum( int  N ){
        this.N = N;
    }

    public String getNextInstance(){
        if (i>=N) return null;

        i++;

        int numPos = rand.nextInt(this.sparsity);

        int[] pos = new int[numPos];
        double[] values = new double[numPos];

        double innerProd = 0;
        for( int i=0; i<numPos; i++ ){
            pos[i] = rand.nextInt(this.dim);
            values[i] = rand.nextGaussian();

            innerProd += (values[i] * this.w[pos[i]]);
        }

        double score = 1.0 / (1.0 + Math.exp(-innerProd));
        double rs = rand.nextDouble();
        int y;
        if (rs < score) y = 1;
        else y = -1;

        String strLine = convertToString(y, pos, values);
        return strLine.toString();
    }

    protected String convertToString( int label, int[] pos, double[] values ){
        StringBuilder strLine = new StringBuilder("");
        if (label>0)
            strLine.append("1 |F");
        else
            strLine.append("-1 |F");

        for(int i = 0; i < pos.length; i++ ){
            strLine.append( " f" + pos[i] + ":" + values[i]);
        }

        return strLine.toString();
    }

    public void close() {
        i=0;
    }

    public DataGenerator copy() {
        DataGeneratorNormal dg = new DataGeneratorNormal(this.N, this.dim, this.sparsity, this.seed);
        dg.w = this.w;
        return dg;
    }

}