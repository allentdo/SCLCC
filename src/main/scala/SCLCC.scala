import mulan.classifier.MultiLabelOutput
import mulan.classifier.transformation.{LocalClassifieChain, TransformationBasedMultiLabelLearner}
import mulan.data.MultiLabelInstances
import org.apache.spark.SparkContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import weka.core.{DenseInstance, Instance, Instances}
import weka.filters.Filter
import weka.filters.unsupervised.attribute.Remove

import scala.util.Random

/**
  * Created by WangHong on 2017/12/3.
  */
class SCLCC
( private var sc:SparkContext,
  private var k: Int=3,
  private var fun: Int=1,
  private var iterNum: Int=100,
  private var maxChange: Double=0.0,
  private var debug: Boolean=false,
  private var rand:Random=new Random(System.nanoTime()),
  private var lccs: Array[LocalClassifieChain]=null,
  private var lcIdx: Array[Array[Int]]=null) extends TransformationBasedMultiLabelLearner with Serializable with Logging{

  def this()=this(SparkSession.builder().appName(this.getClass.getSimpleName).master("local[*]").getOrCreate().sparkContext)
  /**
    * @param trainingSet the training data set
    * @throws Exception if learner model was not created successfully
    */
  override protected def buildInternal(trainingSet: MultiLabelInstances): Unit = {
    //①聚类
    val dataSet: Instances = new Instances(trainingSet.getDataSet)
    //1、过滤出标签
    val labelsRdd = this.sc.makeRDD(getLabels(dataSet))
    //2、根据fun得到局部链序

    //②训练
  }

  private def getLabels(dataSet:Instances):Array[Array[Double]]={
    //只保留标签列
    val remove = new Remove()
    remove.setAttributeIndicesArray(labelIndices)
    remove.setInvertSelection(true)
    remove.setInputFormat(dataSet)
    val labels = Filter.useFilter(dataSet, remove)
    //标签数据转置
    val labelIns = new Array[Double](labelIndices.length).map(_=>new Array[Double](labels.size()))
    for(i <- 0 until labels.size()){
      val tmp = new DenseInstance(labels.get(i)).toString.split(",").map(_.toDouble)
      for(j <- 0 until tmp.size){
        labelIns(j)(i) = tmp(j)
      }
    }
    labelIns
  }

  /**
    * @param instance the data instance to predict on
    * @throws Exception if an error occurs while making the prediction.
    * @return the output of the learner for the given instance
    */
  override protected def makePredictionInternal(instance: Instance): MultiLabelOutput = {
    null
  }
}
