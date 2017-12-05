package org.apache.spark.mllib.multilabel

import mulan.classifier.MultiLabelOutput
import mulan.classifier.transformation.{ClusterLocalClassifierChains, LocalClassifieChainOnlyPredict, TransformationBasedMultiLabelLearner}
import mulan.data.{DataUtils, MultiLabelInstances}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import weka.classifiers.{AbstractClassifier, Classifier}
import weka.classifiers.meta.FilteredClassifier
import weka.classifiers.trees.J48
import weka.core.{DenseInstance, Instance, Instances}
import weka.filters.Filter
import weka.filters.unsupervised.attribute.Remove
import wh.Utils

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by WangHong on 2017/12/3.
  */
class SCLCC
(private var spark:SparkSession=SparkSession.builder().appName("SCLCC").master("local[*]").getOrCreate(),
 private var classifier: Classifier = new J48(),
 private var k: Int=3,
 private var fun: Int=1,
 private var iterNum: Int=100,
 private var maxChange: Double=0.0,
 private var initializationSteps:Int=2,
 private var debug: Boolean=false,
 private var rand:Random=new Random(System.nanoTime()),
 private var lccs: List[LocalClassifieChainOnlyPredict]=null,
 private var lcIdx: List[List[Int]]=null) extends TransformationBasedMultiLabelLearner with Serializable with Logging{

  /**
    * @param trainingSet the training data set
    * @throws Exception if learner model was not created successfully
    */
  override protected def buildInternal(trainingSet: MultiLabelInstances): Unit = {
    //①聚类
    val dataSet: Instances = new Instances(trainingSet.getDataSet)
    //1、过滤出标签
    val labelsRdd = this.spark.sparkContext.makeRDD(getLabels(dataSet))
    //2、根据fun得到局部链序
    val centers = fun match {
      case 1 => initkModesCossimilParallel(labelsRdd.map(_._2))
    }
    if(debug) centers.foreach(x=>println(x.mkString("[",", ","]")))
    this.k = centers.size

    val lcIdxRdd = kModesCossimilParallel(labelsRdd,centers).zipWithIndex().map(x=>(x._2.toInt,x._1))
    this.lcIdx = lcIdxRdd.collect().toList.sortBy(_._1).map(_._2)
    if(debug) this.lcIdx.foreach(println)
    //②训练
    //1、广播训练集
    val bcDataSet = spark.sparkContext.broadcast(trainingSet)
    //2、并行训练FilteredClassifier 3、归位
    this.lccs = lcIdxRdd.flatMap(x=>splitIndex(x._2).map(y=>(x._1,y._1,y._2,y._3)))
      .mapPartitions(x=>x.map(y=>(y._1,y._2,getFilteredClassifier(bcDataSet.value.getDataSet,y._4,y._3))))
      .groupBy(_._1).map(x=>(x._1,x._2.map(y=>(y._2,y._3)).toList.sortBy(_._1)))
      .collect().toList.sortBy(_._1).map(x=>{
      val lcc = new LocalClassifieChainOnlyPredict(this.lcIdx(x._1).toArray)
      lcc.setEnsemble(x._2.map(_._2).toArray)
      lcc.build(bcDataSet.value)
      lcc
    })
  }

  private def splitIndex(indexs:List[Int]):List[(Int,Int,Array[Int])]={
    val allSet = indexs.toSet
    val contantSet = new mutable.HashSet[Int]()
    val res = new ArrayBuffer[(Int,Int,Array[Int])](indexs.size)
    for(i<- 0 until indexs.size){
      contantSet.add(indexs(i))
      res += ((i,this.labelIndices(indexs(i)),allSet.diff(contantSet).toArray.map(labelIndices(_))))
    }
    res.toList
  }

  private def getFilteredClassifier(dataForm:Instances, indicesToRemove:Array[Int], classindex:Int):FilteredClassifier={
    val fclass = new FilteredClassifier
    fclass.setClassifier(AbstractClassifier.makeCopy(this.classifier))
    val remove = new Remove
    remove.setAttributeIndicesArray(indicesToRemove)
    remove.setInvertSelection(false)
    remove.setInputFormat(dataForm)
    fclass.setFilter(remove)
    dataForm.setClassIndex(classindex)
    if(debug) println(s"$classindex FilteredClassifier start build")
    fclass.buildClassifier(dataForm)
    if(debug) println(s"$classindex FilteredClassifier built finished")
    fclass
  }

  //初始化蔟中心并行算法
  private def initkModesCossimilParallel(data:RDD[List[Double]]):List[List[Double]]={
    // Initialize empty centers and point costs.
    var costs = data.map(_ => Double.PositiveInfinity)

    // Initialize the first center to a random point.
    val seed = System.nanoTime()
    val sample = data.takeSample(false, 1, seed)
    // Could be empty if data is empty; fail with a better message early:
    require(sample.nonEmpty, s"No samples available from $data")

    val centers = ArrayBuffer[List[Double]]()
    var newCenters = Seq(sample.head)
    centers ++= newCenters

    // On each step, sample 2 * k points on average with probability proportional
    // to their squared distance from the centers. Note that only distances between points
    // and new centers are computed in each iteration.
    var step = 0
    var bcNewCentersList = ArrayBuffer[Broadcast[_]]()
    while (step < initializationSteps) {
      val bcNewCenters = data.context.broadcast(newCenters)
      bcNewCentersList += bcNewCenters
      val preCosts = costs
      costs = data.zip(preCosts).map { case (point, cost) =>
        math.min(SCLCC.pointCost(bcNewCenters.value, point), cost)
      }.persist(StorageLevel.MEMORY_AND_DISK)
      val sumCosts = costs.sum()

      bcNewCenters.unpersist(blocking = false)
      preCosts.unpersist(blocking = false)

      val chosen = data.zip(costs).mapPartitionsWithIndex { (index, pointCosts) =>
        pointCosts.filter { case (_, c) => rand.nextDouble() < 2.0 * c * k / sumCosts }.map(_._1)
      }.collect()
      newCenters = chosen.map(x=>x)
      centers ++= newCenters
      newCenters = centers.map(x=>x)
      step += 1
    }

    costs.unpersist(blocking = false)
    bcNewCentersList.foreach(_.destroy(false))

    val distinctCenters = centers.distinct

    if (distinctCenters.size <= k) {
      distinctCenters.toList
    } else {
      // Finally, we might have a set of more than k distinct candidate centers; weight each
      // candidate by the number of points in the dataset mapping to it and run a local k-means++
      // on the weighted centers to pick k of them

      kModesCossimilLocal(distinctCenters.toList,k,10,0.0)
    }

  }

  def stopSpark=this.spark.stop()

  //余弦相似度kmodes聚类,得到链序
  private def kModesCossimilParallel(datas:RDD[(Int, List[Double])],initCenter:List[List[Double]]): RDD[List[Int]] ={
    var i = 0
    var cenChange = Double.PositiveInfinity
    var centers = initCenter.zipWithIndex.map(x=>(x._2,x._1))
    var Clusters:RDD[((Double, Double), Int, Int, scala.List[Double])] = null
    while (i<this.iterNum && cenChange>this.maxChange){
      //广播中心
      val bcCenters = spark.sparkContext.broadcast(centers)
      //划分类簇
      Clusters = datas.map(x=>bcCenters.value.map(y=>(SCLCC.expAbsCosSim(y._2,x._2),y._1,x._1,x._2)).sortBy(_._1._1).head)

      //重新计算中心
      centers = Clusters.map(x=>(x._2,(x._3,x._4,x._1._2>0.0))).map(x=>(x._1,if(x._2._3) x._2._2 else x._2._2.map(y=>y* -1.0)))
        .reduceByKey((x,y)=>x.zip(y).map(z=>z._1+z._2))
        .map(x=>(x._1,x._2.map(y=>if(y>=0) 1.0 else -1.0)))
        .collect().toList
      cenChange = 0.0
      val newCenMap = centers.toMap
      val cenMap = bcCenters.value.toMap
      for(i<-0 until centers.size){
        cenChange+= SCLCC.expAbsCosSim(newCenMap.get(i).get,cenMap.get(i).get)._1
      }
    }
    if(debug) centers.foreach(x=>println(x._2))
    Clusters.map(x=>(x._2,x._1._1,x._3)).groupBy(_._1).map(x=>x._2.toList.map(y=>(y._2,y._3))).map(x=>x.sortBy(_._1).reverse.map(_._2))
  }

  //串行余弦相似度kmodes聚类
  private def kModesCossimilLocal(ins:List[List[Double]],k:Int,inum:Int,maxc:Double):List[List[Double]]={
    val clcc = new ClusterLocalClassifierChains()
    Utils.kModesCossimil(ins.map(_.toArray).toArray,k,inum,maxc).map(_.toList).toList
  }

  //过滤出标签
  private def getLabels(dataSet:Instances):Array[(Int,List[Double])]={
    //只保留标签列
    val remove = new Remove()
    remove.setAttributeIndicesArray(labelIndices)
    remove.setInvertSelection(true)
    remove.setInputFormat(dataSet)
    val labels = Filter.useFilter(dataSet, remove)
    //标签数据转置
    val labelIns = new Array[Double](labelIndices.length).map(_=>new Array[Double](labels.size())).zipWithIndex.map(x=>(x._2,x._1.toBuffer))
    for(i <- 0 until labels.size()){
      val tmp = new DenseInstance(labels.get(i)).toString.split(",").map(_.toDouble)
      for(j <- 0 until tmp.size){
        labelIns(j)._2(i) = if (tmp(j)==0.0) -1.0 else 1.0
      }
    }
    labelIns.map(x=>(x._1,x._2.toList))
  }

  /**
    * @param instance the data instance to predict on
    * @throws Exception if an error occurs while making the prediction.
    * @return the output of the learner for the given instance
    */
  override protected def makePredictionInternal(instance: Instance): MultiLabelOutput = {
    val bipartition = new Array[Boolean](numLabels)
    val confidences = new Array[Double](numLabels)

    for(i<- 0 until this.lcIdx.size){
      val localChainO = this.lccs(i).makePrediction(instance)
      val bip = localChainO.getBipartition
      val conf = localChainO.getConfidences
      for(j<- 0 until this.lcIdx(i).size){
        bipartition(this.lcIdx(i)(j)) = bip(j)
        confidences(this.lcIdx(i)(j)) = conf(j)
      }
    }
    val mlo = new MultiLabelOutput(bipartition, confidences)
    if(debug) println(mlo)
    return mlo
  }

  def setIsDebug(dg: Boolean): this.type = {
    this.debug = dg
    this
  }
}

object SCLCC extends Serializable{

  def pointCost(centers: TraversableOnce[List[Double]],
                point: List[Double]):Double={
    centers.map(x=>expAbsCosSim(x,point)._1).min
  }

  /**
    * 基于余弦相似度的改进距离计算公式
    *
    * @param v1
    * @param v2
    * @return double[] -> 2元素 距离值，余弦相似度
    */
  private def expAbsCosSim(v1: List[Double], v2: List[Double]):(Double,Double) = {
    require(v1.size == v2.size && v1.size > 0, "向量计算距离维度不匹配或长度小于1")
    var cosup = 0.0
    var cosdownv1 = 0.0
    var cosdownv2 = 0.0
    for(i<- 0 until v1.size){
        cosup += v1(i) * v2(i)
        cosdownv1 += Math.pow(v1(i), 2)
        cosdownv2 += Math.pow(v2(i), 2)
    }
    var cos = cosup / (math.sqrt(cosdownv1) * math.sqrt(cosdownv2))
    if (cos == 0) cos = Double.MinPositiveValue
    (-1.0 * math.log(math.abs(cos)), cos)
  }

}