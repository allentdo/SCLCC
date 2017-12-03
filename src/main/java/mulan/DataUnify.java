package mulan;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.Random;

/**
 * Created by WangHong on 2017/12/2.
 */
public class DataUnify {
    public static void main(String[] args) throws Exception {
        String basePath = "F:/bigmultidata/";
        String[] datas = {"bookmarks","Corel5k","eurlex-dc-leaves","eurlex-ev","eurlex-sm","mediamill","nus-wide-full_BoW_l2","nus-wide-full-cVLADplus","rcv1subset2","tmc2007"};
        for (int i = 0; i < datas.length; i++) {
            String name = datas[i];
            System.out.println(name);
            MultiLabelInstances train = new MultiLabelInstances(basePath+name+"-train.arff", basePath+name+".xml");
            MultiLabelInstances test = new MultiLabelInstances(basePath+name+"-test.arff", basePath+name+".xml");
        }
    }
}
