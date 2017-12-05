package mulan.classifier.transformation;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

import java.util.HashMap;

/**
 * Created by WangHong on 2017/11/19.
 */
public class LocalClassifieChainOnlyPredict extends TransformationBasedMultiLabelLearner{
    /**
     * The new chain ordering of the label indices
     */
    private int[] chain;
    /**
     * The ensemble of binary relevance models. These are Weka
     * FilteredClassifier objects, where the filter corresponds to removing all
     * label apart from the one that serves as a target for the corresponding
     * model.
     */
    protected FilteredClassifier[] ensemble;



    public LocalClassifieChainOnlyPredict(int[] aChain) {
        chain = aChain;
    }

    public void setEnsemble(FilteredClassifier[] ens){
        this.ensemble = ens;

    }


    protected void buildInternal(MultiLabelInstances train) throws Exception {

    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        //长度需要修改，改为chain.length！！
        boolean[] bipartition = new boolean[chain.length];
        //长度需要修改，改为chain.length！！
        double[] confidences = new double[chain.length];

        Instance tempInstance = DataUtils.createInstance(instance, instance.weight(), instance.toDoubleArray());
        //循环结束条件需要修改 改为counter<chain.length！！
        for (int counter = 0; counter < chain.length; counter++) {
            double distribution[];
            try {
                distribution = ensemble[counter].distributionForInstance(tempInstance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
            int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

            // Ensure correct predictions both for class values {0,1} and {1,0}
            Attribute classAttribute = ensemble[counter].getFilter().getOutputFormat().classAttribute();
            bipartition[counter] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

            // The confidence of the label being equal to 1
            confidences[counter] = distribution[classAttribute.indexOfValue("1")];

            tempInstance.setValue(labelIndices[chain[counter]], maxIndex);

        }

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }

    public static void main(String[] args) throws Exception{
        String path = "./data/testData/";
        Classifier baseClassifier = new J48();
        int[] chn = {5,3,4,0};
        MultiLabelLearnerBase learner = new LocalClassifieChain(baseClassifier,chn);

        String trainDatasetPath = path + "emotions-train.arff";
        String testDatasetPath = path + "emotions-test.arff";
        String xmlLabelsDefFilePath = path + "emotions.xml";
        MultiLabelInstances trainDataSet = new MultiLabelInstances(trainDatasetPath, xmlLabelsDefFilePath);
        MultiLabelInstances testDataSet = new MultiLabelInstances(testDatasetPath, xmlLabelsDefFilePath);

        learner.build(trainDataSet);
        System.out.println(learner.makePrediction(testDataSet.getDataSet().firstInstance()));
    }
}
//Bipartion: [false, true, true, false] Confidences: [0.018072289156626505, 1.0, 1.0, 0.0] Ranking: [3, 2, 1, 4]Predicted values: null