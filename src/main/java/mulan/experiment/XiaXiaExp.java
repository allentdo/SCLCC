package mulan.experiment;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.*;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;

/**
 * 夏翠萍实验
 * Created by Xiacuiping on 2017/11/23.
 */
public class XiaXiaExp {
    public static void main(String[] args) throws Exception{
        int numFolds = 2;
        String basePath = "./data/testData/";
        String[] smalldatas={"enron"};
        String[] bigdatas={};
        for (int i = 0; i < smalldatas.length; i++) {
            String data = smalldatas[i];
            for (int j = 9; j <= 10; j++) {
                MultiLabelLearnerBase mlknn = new MLkNN(j,1.0);
                exp_cross(basePath+data+".arff", basePath+data+".xml",mlknn,data+"_"+mlknn.getClass().getSimpleName()+"_k"+j+"_folds"+numFolds,numFolds);
            }
            BinaryRelevance br_j48 = new BinaryRelevance(new J48());
            exp_cross(basePath+data+".arff", basePath+data+".xml",br_j48,data+"_"+br_j48.getClass().getSimpleName()+"_j48_folds"+numFolds,numFolds);
            BinaryRelevance br_svm = new BinaryRelevance(new SMO());
            exp_cross(basePath+data+".arff", basePath+data+".xml",br_svm,data+"_"+br_svm.getClass().getSimpleName()+"_smo_folds"+numFolds,numFolds);
        }
        for (int i = 0; i < bigdatas.length; i++) {
            String data = bigdatas[i];
            for (int j = 8; j <= 10; j++) {
                MultiLabelLearnerBase mlknn = new MLkNN(j,1.0);
                exp_test(basePath+data+"-train.arff", basePath+data+".xml",basePath+data+"-test.arff",mlknn,data+"_"+mlknn.getClass().getSimpleName()+"_k"+j);
            }
            BinaryRelevance br_j48 = new BinaryRelevance(new J48());
            exp_test(basePath+data+"-train.arff", basePath+data+".xml",basePath+data+"-test.arff",br_j48,data+"_"+br_j48.getClass().getSimpleName()+"_j48");
            BinaryRelevance br_svm = new BinaryRelevance(new SMO());
            exp_test(basePath+data+"-train.arff", basePath+data+".xml",basePath+data+"-test.arff",br_svm,data+"_"+br_svm.getClass().getSimpleName()+"_smo");
        }
    }

    public static void exp_cross(String arffStr, String xmlStr,MultiLabelLearnerBase learn, String clsName, int folds) throws Exception{
        PrintWriter w = new PrintWriter(clsName);
        String now = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(System.currentTimeMillis());
        w.println("start at "+now);
        MultiLabelInstances dataset = new MultiLabelInstances(arffStr, xmlStr);
        Evaluator eval = new Evaluator();
        long time = System.currentTimeMillis();
        MultipleEvaluation results = eval.crossValidate(learn, dataset, folds);
        time = System.currentTimeMillis()-time;
        w.println(results);
        w.println("run time:"+time);
        now = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(System.currentTimeMillis());
        w.println("finished at "+now);
        w.close();
    }

    public static void exp_test(String train,String xml,String test,MultiLabelLearnerBase learn, String clsName) throws Exception{
        PrintWriter w = new PrintWriter(clsName);
        String now = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(System.currentTimeMillis());
        w.println("start at "+now);
        long time = System.currentTimeMillis();
        MultiLabelInstances traindata = new MultiLabelInstances(train, xml);
        learn.build(traindata);
        MultiLabelInstances testdata = new MultiLabelInstances(test,xml);
        //多标签指标评估
        ArrayList measures = new ArrayList();
        int numOfLabels = traindata.getNumLabels();
        // add example-based measures
        measures.add(new HammingLoss());
        measures.add(new SubsetAccuracy());
        measures.add(new ExampleBasedPrecision());
        measures.add(new ExampleBasedRecall());
        measures.add(new ExampleBasedFMeasure());
        measures.add(new ExampleBasedAccuracy());
        measures.add(new ExampleBasedSpecificity());
        // add label-based measures
        measures.add(new MicroPrecision(numOfLabels));
        measures.add(new MicroRecall(numOfLabels));
        measures.add(new MicroFMeasure(numOfLabels));
        measures.add(new MicroSpecificity(numOfLabels));
        measures.add(new MacroPrecision(numOfLabels));
        measures.add(new MacroRecall(numOfLabels));
        measures.add(new MacroFMeasure(numOfLabels));
        measures.add(new MacroSpecificity(numOfLabels));
        // add ranking based measures
        measures.add(new AveragePrecision());
        measures.add(new Coverage());
        measures.add(new OneError());
        measures.add(new IsError());
        measures.add(new ErrorSetSize());
        measures.add(new RankingLoss());
        measures.add(new MeanAveragePrecision(numOfLabels));
        measures.add(new GeometricMeanAveragePrecision(numOfLabels));
        measures.add(new MeanAverageInterpolatedPrecision(numOfLabels, 10));
        measures.add(new GeometricMeanAverageInterpolatedPrecision(numOfLabels, 10));
        measures.add(new MicroAUC(numOfLabels));
        measures.add(new MacroAUC(numOfLabels));
        measures.add(new LogLoss());



        Evaluator eval = new Evaluator();
        Evaluation results = eval.evaluate(learn,testdata,measures);
        w.println(results);
        time = System.currentTimeMillis()-time;
        w.println("run time:"+time);
        now = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(System.currentTimeMillis());
        w.println("finished at "+now);
        w.close();
    }

}
