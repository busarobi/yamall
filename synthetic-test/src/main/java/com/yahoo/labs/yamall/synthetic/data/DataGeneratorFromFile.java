package com.yahoo.labs.yamall.synthetic.data;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * Created by busafekete on 8/15/17.
 */
public class DataGeneratorFromFile implements DataGenerator {
    String fname = null;
    BufferedReader br = null;

    public DataGeneratorFromFile( String fname ) {
        this.fname = fname;
        try {
            FileInputStream fstream = new FileInputStream(this.fname);
            br = new BufferedReader(new InputStreamReader(fstream));
            System.out.printf( "Open file: %s\n", this.fname);
        } catch (IOException e ){
            System.out.println(e.getMessage());
        }
    }

    public String getNextInstance(){
        String strLine = null;
        try{
            strLine = br.readLine();
        } catch (IOException e ){
            System.out.println(e.getMessage());
        }
        return strLine;

    }

    public void close() {
        try{
            br.close();

            FileInputStream fstream = new FileInputStream(this.fname);
            br = new BufferedReader(new InputStreamReader(fstream));
        } catch (IOException e ){
            System.out.println(e.getMessage());
        }
    }

    public DataGenerator copy(){
        return new DataGeneratorFromFile(this.fname);
    }

}