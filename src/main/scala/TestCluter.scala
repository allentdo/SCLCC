import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

/**
  * Created by WangHong on 2017/11/17.
  */
object TestCluter {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName(this.getClass.getSimpleName).master("local[2]").getOrCreate
    val sc = spark.sparkContext
    // Load and parse the data
    val data = sc.textFile("data/mllib/kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

  }
}
