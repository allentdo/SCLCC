package mulan.experiment;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.transformation.ClusterLocalClassifierChains;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.*;
import weka.classifiers.trees.J48;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;

/**
 * Created by WangHong on 2017/11/27.
 */
public class CLCCKTest {
    public static void main(String[] args) throws Exception{
        int folds = 20;
        FileWriter fw = new FileWriter("res_coskmodek.txt", true);
        PrintWriter toFile = new PrintWriter(fw);
        String[] datasets = {"medical","CAL500"};
        String path = "./data/";
        for (int i = 0; i < datasets.length; i++) {
            String name = datasets[i];
            toFile.println("dataset:"+name);
            toFile.println("MicroF,MacroF,HaLoss,OneErr,RaLoss,AvePrec,RunTime,Classfier");
            String datasetPath = path + name+ ".arff";
            //String testDatasetPath = path + name+ "-test.arff";
            String xmlLabelsDefFilePath = path + name+ ".xml";
            for (int j = 1; j < 20; j++) {
               /* MultiLabelLearnerBase CLCC1 = new ClusterLocalClassifierChains(new RandomForest(), j, 100, 0.0, false,1);
                exp_cross(datasetPath,xmlLabelsDefFilePath,CLCC1,CLCC1.getClass().getSimpleName()+" fun="+1 +" k="+ j,toFile,folds);
                MultiLabelLearnerBase CLCC2 = new ClusterLocalClassifierChains(new J48(), j, 100, 0.0, false,2);
                exp_cross(datasetPath,xmlLabelsDefFilePath,CLCC2,CLCC2.getClass().getSimpleName()+" fun="+2 +" k="+ j,toFile,folds);*/
                MultiLabelLearnerBase CLCC3 = new ClusterLocalClassifierChains(new J48(), j, 100, 0.0, false,3);
                exp_cross(datasetPath,xmlLabelsDefFilePath,CLCC3,CLCC3.getClass().getSimpleName()+" fun="+3 +" k="+ j,toFile,folds);
            }
            /*MultiLabelLearnerBase BR = new BinaryRelevance(new RandomForest());
            MultiLabelLearnerBase CC = new ClassifierChain(new RandomForest());
            //MultiLabelLearnerBase MLknn = new MLkNN();
            MultiLabelLearnerBase ECC = new EnsembleOfClassifierChains(new RandomForest(), 5, true, true);
            //MultiLabelLearnerBase Bpmll = new BPMLL();
            MultiLabelLearnerBase lp = new LabelPowerset(new RandomForest());
            MultiLabelLearnerBase homer = new HOMER(new BinaryRelevance(new RandomForest()),3,HierarchyBuilder.Method.BalancedClustering);
            MultiLabelLearnerBase rakel = new RAkEL(new BinaryRelevance(new RandomForest()));
            MultiLabelLearnerBase tscc = new TwoStageClassifierChainArchitecture(new RandomForest());


            exp_cross(datasetPath,xmlLabelsDefFilePath,BR,BR.getClass().getSimpleName(),toFile,folds);
            exp_cross(datasetPath,xmlLabelsDefFilePath,CC,CC.getClass().getSimpleName(),toFile,folds);
            //exp_cross(datasetPath,xmlLabelsDefFilePath,MLknn,MLknn.getClass().getSimpleName(),toFile,folds);
            exp_cross(datasetPath,xmlLabelsDefFilePath,ECC,ECC.getClass().getSimpleName(),toFile,folds);
            //exp_cross(datasetPath,xmlLabelsDefFilePath,Bpmll,Bpmll.getClass().getSimpleName(),toFile,folds);
            exp_cross(datasetPath,xmlLabelsDefFilePath,lp,lp.getClass().getSimpleName(),toFile,folds);
            exp_cross(datasetPath,xmlLabelsDefFilePath,homer,homer.getClass().getSimpleName(),toFile,folds);
            exp_cross(datasetPath,xmlLabelsDefFilePath,rakel,rakel.getClass().getSimpleName(),toFile,folds);
            exp_cross(datasetPath,xmlLabelsDefFilePath,tscc,tscc.getClass().getSimpleName(),toFile,folds);*/

        }


        toFile.close();
    }
    private static void exp_cross(String arffStr, String xmlStr,MultiLabelLearnerBase learn, String clsName, PrintWriter w, int folds) throws Exception{
        MultiLabelInstances dataset = new MultiLabelInstances(arffStr, xmlStr);
        Evaluator eval = new Evaluator();
        //多标签指标评估
        ArrayList measures = new ArrayList();
        Measure Fexam = new MicroFMeasure(dataset.getNumLabels());
        Measure F1 = new MacroFMeasure(dataset.getNumLabels());
        Measure hloss = new HammingLoss();
        Measure oneError = new OneError();
        Measure rloss = new RankingLoss();
        Measure avgPre= new AveragePrecision();
        measures.add(Fexam);
        measures.add(F1);
        measures.add(hloss);
        measures.add(oneError);
        measures.add(rloss);
        measures.add(avgPre);
        long time = System.currentTimeMillis();
        MultipleEvaluation results = eval.crossValidate(learn, dataset, measures, folds);
        time = System.currentTimeMillis()-time;
        w.println(results.toCSV().replace(";",",")+time+","+clsName);
        w.flush();
    }
}
