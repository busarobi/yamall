package com.yahoo.labs.yamall.spark.helper;

import com.yahoo.labs.yamall.core.Instance;
import org.testng.annotations.Test;

/**
 * Created by busafekete on 9/15/17.
 */
public class HDFSDirectoryReaderTest {
    @Test
    public void testHDFSDirectoryReader() throws Exception {
        String dirname = "/Users/busafekete/work/DistOpt/Clkb_data/";
        HDFSDirectoryReader hdfsreader = new HDFSDirectoryReader(dirname, 22);
        for(int i=0; i <2; i++ ) {
            int fileSize = 0;
            while (hdfsreader.hasNext()) {
                Instance instance = hdfsreader.getNextInstance();
                fileSize++;
            }
            System.out.println("Number of lines: " + fileSize);
            hdfsreader.reset();
        }
        hdfsreader.close();
    }
}