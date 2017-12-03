package mulan.experiment;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.HOMER;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.neural.BPMLL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClusterLocalClassifierChains;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.*;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;

/**
 * Created by WangHong on 2017/12/2.
 */
public class XiaXiaSmallDataExp {
    public static void main(String[] args) throws Exception{
        int folds = 5;
        FileWriter fw = new FileWriter("xiaxia_smalldata_fold5.txt", true);
        PrintWriter toFile = new PrintWriter(fw);
        String[] datasets = {"birds","CAL500","emotions","enron","flags","genbase","medical","scene","yeast","bibtex","Corel5k"};
        String path = "./data/";
        for (int i = 0; i < datasets.length; i++) {
            String name = datasets[i];
            System.out.println("dataset:"+name);
            toFile.println("dataset:"+name);
            toFile.println("HammingLoss  ,SubsetAccurac,ExamplePrecis,ExampleRecall,ExampleFMeasu,ExampleAccura,ExampleSpecif,MicroPrecisio,MicroRecall  ,MicroFMeasure,MicroSpecific,MacroPrecisio,MacroRecall  ,MacroFMeasure,MacroSpecific,AveragePrecis,Coverage     ,OneError     ,IsError      ,ErrorSetSize ,RankingLoss  ,MeanAvePrecis,GeoPrec,MeanAveInPrec,GeoInPr,MicroAUC     ,MacroAUCLogLoss      ,time,Classfier");
            String datasetPath = path + name+ ".arff";
            //String testDatasetPath = path + name+ "-test.arff";
            String xmlLabelsDefFilePath = path + name+ ".xml";
            MultiLabelLearnerBase br_j48 = new BinaryRelevance(new J48());
            exp_cross(datasetPath,xmlLabelsDefFilePath,br_j48,br_j48.getClass().getSimpleName()+" fun=J48",toFile,folds);

            MultiLabelLearnerBase br_smo = new BinaryRelevance(new SMO());
            exp_cross(datasetPath,xmlLabelsDefFilePath,br_smo,br_smo.getClass().getSimpleName()+" fun=SMO",toFile,folds);

            MultiLabelLearnerBase bpmll = new BPMLL();
            exp_cross(datasetPath,xmlLabelsDefFilePath,bpmll,bpmll.getClass().getSimpleName(),toFile,folds);

            MultiLabelLearnerBase rakel = new RAkEL(new BinaryRelevance(new J48()));
            exp_cross(datasetPath,xmlLabelsDefFilePath,rakel,rakel.getClass().getSimpleName(),toFile,folds);

            MultiLabelLearnerBase homer = new HOMER();
            exp_cross(datasetPath,xmlLabelsDefFilePath,homer,homer.getClass().getSimpleName(),toFile,folds);

            MultiLabelLearnerBase mlknn = new MLkNN();
            exp_cross(datasetPath,xmlLabelsDefFilePath,mlknn,mlknn.getClass().getSimpleName(),toFile,folds);

        }


        toFile.close();
    }

    private static void exp_cross(String arffStr, String xmlStr,MultiLabelLearnerBase learn, String clsName, PrintWriter w, int folds) throws Exception{
        System.out.println(clsName);
        MultiLabelInstances dataset = new MultiLabelInstances(arffStr, xmlStr);
        int numOfLabels = dataset.getNumLabels();
        Evaluator eval = new Evaluator();
        //多标签指标评估
        ArrayList measures = new ArrayList();
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
        long time = System.currentTimeMillis();
        MultipleEvaluation results = eval.crossValidate(learn, dataset, measures, folds);
        time = System.currentTimeMillis()-time;
        w.println(results.toCSV().replace(";",",")+time+","+clsName);
        w.flush();
    }
}
