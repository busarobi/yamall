package com.yahoo.labs.yamall.local;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Properties;

/**
 * Created by busafekete on 7/17/17.
 */

public class ReadProperty {
    public static Properties readProperty(String fname) {
        Properties properties = new Properties();
        try {
            FileInputStream in = new FileInputStream(fname);
            properties.load(in);
            in.close();
        } catch (FileNotFoundException e) {
            System.err.println(e.getMessage());
            System.exit(-1);
        } catch (IOException e) {
            System.err.println(e.getMessage());
            System.exit(-1);
        }
        return properties;
    }

}