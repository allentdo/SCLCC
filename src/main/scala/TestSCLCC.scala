import java.util

import mulan.classifier.MultiLabelLearnerBase
import mulan.data.MultiLabelInstances
import mulan.evaluation.Evaluator
import mulan.evaluation.measure.{MacroFMeasure, Measure, MicroFMeasure}
import org.apache.spark.sql.SparkSession
import mulan.evaluation.measure._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.multilabel.SCLCC
import weka.core.Instances
/**
  * Created by WangHong on 2017/12/4.
  */
object TestSCLCC extends Serializable{
  Logger.getLogger("org").setLevel(Level.WARN)
  def main(args: Array[String]): Unit = {
    val sclcc = new SCLCC()
    val basePath = "./testdata/"
    val dataName = "CAL500"
    val traindata = new MultiLabelInstances(basePath+dataName+"-train.arff", basePath+dataName+".xml")
    sclcc.build(traindata)
    sclcc.makePrediction(traindata.getDataSet.firstInstance)
    exp_cross(traindata,sclcc)
    sclcc.stopSpark
  }

  @throws[Exception]
  private def exp_cross(dataset:MultiLabelInstances, learn: MultiLabelLearnerBase) {
    val test = dataset.getDataSet
    for(i<- 0 until test.size()){
      println(learn.makePrediction(test.get(i)))
    }
  }
}
