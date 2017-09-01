package com.yahoo.labs.yamall.spark.helper;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.parser.VWParser;
import org.apache.spark.api.java.function.Function;

/**
 * Created by busafekete on 8/29/17.
 */
public class StringToYamallInstance implements Function<String,Instance> {
    protected VWParser parser = null;

    public StringToYamallInstance( int bitsHash ){
        parser = new VWParser(bitsHash, null, false);
    }
    @Override
    public Instance call(String v1) throws Exception {
        return parser.parse(v1);
    }
}
